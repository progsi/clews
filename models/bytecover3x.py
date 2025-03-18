import sys
import torch, math
from nnAudio import features  # type: ignore
from einops import rearrange, repeat

from lib import layers
from lib import tensor_ops as tops


class Model(torch.nn.Module):

    def __init__(self, conf, sr=16000, eps=1e-6):
        super().__init__()
        self.conf = conf
        self.sr = sr
        self.eps = eps
        self.minlen = conf.shingling.len
        # CQT
        self.cqtbins = self.conf.cqt.noctaves * self.conf.cqt.nbinsoct
        self.cqt = features.CQT1992v2(
            sr=self.sr,
            hop_length=int(self.conf.cqt.hoplen * sr),
            n_bins=self.cqtbins,
            bins_per_octave=self.conf.cqt.nbinsoct,
            trainable=False,
            verbose=False,
        )
        self.cqtpool = torch.nn.AvgPool1d(
            self.conf.cqt.pool.len, stride=self.conf.cqt.pool.hop
        )
        # Model
        nc1 = conf.ncha // 8
        nc2 = conf.ncha // 4
        nc3 = conf.ncha // 2
        nc4 = conf.ncha  # 2048
        self.frontend = torch.nn.Sequential(
            layers.Unsqueeze(1),
            torch.nn.Conv2d(1, nc1, 7, stride=(1, 2), bias=False),
            torch.nn.BatchNorm2d(nc1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, 2),
        )
        aux = [layers.ResNet50BottBlock(nc1, nc1, ibn=True)]
        for _ in range(2):
            aux += [layers.ResNet50BottBlock(nc1, nc1, ibn=True)]
        aux += [layers.ResNet50BottBlock(nc1, nc2, ibn=True, stride=2)]
        for _ in range(3):
            aux += [layers.ResNet50BottBlock(nc2, nc2, ibn=True)]
        aux += [layers.ResNet50BottBlock(nc2, nc3, ibn=True, stride=2)]
        for _ in range(5):
            aux += [layers.ResNet50BottBlock(nc3, nc3, ibn=True)]
        aux += [layers.ResNet50BottBlock(nc3, nc4)]
        for _ in range(2):
            aux += [layers.ResNet50BottBlock(nc4, nc4)]
        self.resblocks = torch.nn.Sequential(*aux)
        self.embpool = layers.GeMPool()
        self.proj = torch.nn.Sequential(
            torch.nn.BatchNorm1d(conf.ncha),
            torch.nn.Linear(conf.ncha, conf.zdim),
        )
        # Loss
        self.w = torch.nn.Parameter(
            0.076 * (2 * torch.rand(conf.maxcliques, conf.nsub, conf.zdim) - 1)
        )
        self.tau = torch.nn.Parameter(torch.ones(1))
        self.smooth = conf.smooth
        self.relu = torch.nn.ReLU()
        self.margin = conf.margin
        self.lamb = conf.lamb

    def get_shingle_params(self):
        return self.conf.shingling.len, self.conf.shingling.hop

    ###########################################################################

    def forward(
        self,
        h,  # (B,T)
        shingle_len=None,
        shingle_hop=None,
    ):
        with torch.inference_mode():
            h = self.prepare(h, shingle_len=shingle_len, shingle_hop=shingle_hop)
        h = h.clone()
        h, _ = self.embed(h)
        return h  # (B,C)

    def prepare(
        self,
        h,  # (B,T)
        shingle_len=None,
        shingle_hop=None,
    ):
        assert h.ndim == 2
        assert shingle_len is None or shingle_len > 0
        assert shingle_hop is None or shingle_hop > 0
        slen = self.conf.shingling.len if shingle_len is None else shingle_len
        shop = self.conf.shingling.hop if shingle_hop is None else shingle_hop
        # Shingle
        h = tops.get_frames(
            h, int(self.sr * slen), int(self.sr * shop), pad_mode="zeros"
        )
        # Check audio length
        h = tops.force_length(
            h, int(self.sr * self.minlen), dim=-1, pad_mode="repeat", allow_longer=True
        )
        # CQT
        s = h.size(1)
        h = rearrange(h, "b s t -> (b s) t")
        h = self.cqt(h)
        h = self.cqtpool(h)
        h = rearrange(h, "(b s) c t -> b s c t", s=s)
        return h  # (B,S,C,T)

    def embed(
        self,
        h,  # (B,S,C,T)
    ):
        assert h.ndim == 4
        s = h.size(1)
        h = rearrange(h, "b s c t -> (b s) c t")
        h = h / (h.abs().max(1, keepdim=True)[0].max(2, keepdim=True)[0] + self.eps)
        h = self.frontend(h)
        h = self.resblocks(h)
        h = self.embpool(h)
        h = self.proj(h)
        h = rearrange(h, "(b s) c -> b s c", s=s)
        return h, None  # (B,S,C)

    ###########################################################################

    def loss(
        self,
        label,  # (B)
        idx,  # (B)
        z,  # (B,S,C)
        extra=None,
    ):
        assert len(label) == len(idx) and len(label) == len(z)

        # Logits ByteCover
        logits = self.tau.exp() * self.maxmean(z, self.w)
        loss_cla = torch.nn.functional.cross_entropy(
            logits, label, label_smoothing=self.smooth
        )

        # Triplet ByteCover
        sim = self.maxmean(z, z)
        samecla = label.view(-1, 1) == label.view(1, -1)
        diffid = idx.view(-1, 1) != idx.view(1, -1)
        pos = samecla & diffid
        neg = ~samecla
        possim = torch.where(pos, sim, 2).min(1)[0]
        negsim = torch.where(neg, sim, -2).max(1)[0]
        loss_dist = self.relu((1 + self.margin) * negsim - possim).mean()

        # Reg
        pos, neg = pos.type_as(z), neg.type_as(z)
        loss_reg = (pos * (1 - sim)).sum() / (pos.sum() + self.eps)

        loss = loss_cla + loss_dist + self.lamb * loss_reg
        logdict = {
            "l_main": loss,
            "l_cent": loss_cla,
            "l_cont": loss_dist,
            "v_dpos": (pos * sim).sum() / (pos.sum() + self.eps),
            "v_dneg": (neg * sim).sum() / (neg.sum() + self.eps),
        }
        return loss, logdict

    def maxmean(self, x, y):
        assert x.ndim == 3 and y.ndim == 3 and x.size(-1) == y.size(-1)
        s, l = x.size(1), y.size(1)
        x = rearrange(x, "b s c -> (b s) c")
        y = rearrange(y, "k l c -> (k l) c")
        sim = tops.pairwise_distance_matrix(x, y, mode="cossim")
        sim = rearrange(sim, "(b s) (k l) -> b k s l", s=s, l=l)
        res = sim.max(-1)[0].mean(-1)
        return res

    ###########################################################################

    def distances(
        self,
        q,  # (B,S,C)
        c,  # (B',S',C)
        qmask=None,
        cmask=None,
        redux_strategy=None,
    ):
        assert q.ndim == 3 and c.ndim == 3 and q.size(-1) == c.size(-1)
        if redux_strategy is None:
            redux_strategy = "smeanmin"
        s1, s2 = q.size(1), c.size(1)
        q = rearrange(q, "b s c -> (b s) c")
        c = rearrange(c, "b s c -> (b s) c")
        dist = tops.pairwise_distance_matrix(q, c, mode="cos")
        dist = rearrange(dist, "(b1 s1) (b2 s2) -> b1 b2 s1 s2", s1=s1, s2=s2)
        if qmask is not None and cmask is not None:
            qmask = rearrange(qmask, "b s -> (b s)")
            cmask = rearrange(cmask, "b s -> (b s)")
            mask = qmask.view(-1, 1) | cmask.view(1, -1)
            mask = rearrange(mask, "(bq sq) (bc sc) -> bq bc sq sc", sq=s1, sc=s2)
        else:
            mask = None
        dist = tops.distance_tensor_redux(dist, redux_strategy, mask=mask)
        return dist
