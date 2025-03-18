import sys
import torch, math
from nnAudio import features  # type: ignore
from einops import rearrange

from lib import layers
from lib import tensor_ops as tops


class Model(torch.nn.Module):

    def __init__(self, conf, sr=16000, eps=1e-6, max_exp=10):
        super().__init__()
        self.sr = sr
        self.eps = eps
        # Shingling
        self.shingling_len = conf.shingling.len
        self.shingling_hop = conf.shingling.hop
        self.minlen = self.shingling_len  # set minlen to training shinglen
        # CQT
        self.cqt = torch.nn.Sequential(
            features.CQT1992v2(
                sr=self.sr,
                hop_length=int(conf.cqt.hoplen * sr),
                n_bins=conf.cqt.noctaves * conf.cqt.nbinsoct,
                bins_per_octave=conf.cqt.nbinsoct,
                filter_scale=conf.cqt.fscale,
                trainable=False,
                verbose=False,
            ),
            torch.nn.AvgPool1d(conf.cqt.pool, stride=conf.cqt.pool),
        )
        # Model - Frontend
        ncha0, ncha = conf.frontend.channels
        self.frontend = torch.nn.Sequential(
            layers.CQTPrepare(pow=conf.frontend.cqtpow),
            torch.nn.Conv2d(1, ncha0, (12, 3), stride=(1, 2), bias=False),
            torch.nn.BatchNorm2d(ncha0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ncha0, ncha, (12, 3), stride=2, bias=False),
        )
        # Model - Backbone
        aux = []
        for nb, nc, st in zip(
            conf.backbone.blocks, conf.backbone.channels, conf.backbone.down
        ):
            aux += [layers.MyIBNResBlock(ncha, nc, stride=st)]
            for _ in range(nb - 1):
                aux += [layers.MyIBNResBlock(nc, nc)]
            ncha = nc
        self.backbone = torch.nn.Sequential(*aux)
        # Pooling & projection
        self.pool = layers.GeMPool()
        self.proj = torch.nn.Sequential(
            torch.nn.BatchNorm1d(ncha),
            torch.nn.Linear(ncha, conf.zdim, bias=False),
        )
        # Loss
        self.redux = conf.loss.redux
        self.gamma = conf.loss.gamma
        self.epsilon = conf.loss.epsilon
        self.b = max_exp
        self.beta = 1 / (self.epsilon * math.exp(self.b))

    def get_shingle_params(self):
        return self.shingling_len, self.shingling_hop

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
        # Shingle
        slen = self.shingling_len if shingle_len is None else shingle_len
        shop = self.shingling_hop if shingle_hop is None else shingle_hop
        h = tops.get_frames(
            h, int(self.sr * slen), int(self.sr * shop), pad_mode="repeat"
        )
        # Check min shingle length
        h = tops.force_length(
            h, int(self.sr * self.minlen), dim=-1, pad_mode="repeat", allow_longer=True
        )
        # CQT
        s = h.size(1)
        h = rearrange(h, "b s t -> (b s) t")
        h = self.cqt(h)
        h = rearrange(h, "(b s) c t -> b s c t", s=s)
        return h  # (B,S,C,T)

    def embed(
        self,
        h,  # (B,S,C,T)
    ):
        assert h.ndim == 4
        # Prepare
        s = h.size(1)
        h = rearrange(h, "b s c t -> (b s) 1 c t")
        # Feedforward
        h = self.frontend(h)
        h = self.backbone(h)
        # Pool and project
        h = self.pool(h)
        z = self.proj(h)
        # Out
        z = rearrange(z, "(b s) c -> b s c", s=s)
        return z, None  # (B,S,C)

    ###########################################################################

    def loss(
        self,
        z_label,  # (B)
        z_idx,  # (B)
        z,  # (B,S,C)
        extra=None,
        numerically_friendly=True,
    ):
        assert len(z_label) == len(z_idx) and len(z_label) == len(z)
        assert len(z) >= 4
        # If no negatives, add label noise for loss stability
        # (we assume positives exist due to batch construction)
        if len(z_label.unique()) == 1:
            z_label[: max(2, int(len(z_label) * 0.01))] = -1

        # Prepare
        sz = z.size(1)
        z = rearrange(z, "b s c -> (b s) c")
        same_label = z_label.view(-1, 1) == z_label.view(1, -1)
        same_idx = z_idx.view(-1, 1) == z_idx.view(1, -1)
        mask_pos = (~same_label) | same_idx
        mask_neg = same_label

        # Distances
        dist = tops.pairwise_distance_matrix(z, z, mode="nsqeuc")
        dist = rearrange(dist, "(b1 s1) (b2 s2) -> b1 b2 s1 s2", s1=sz, s2=sz)
        dpos = tops.distance_tensor_redux(dist, self.redux.pos)
        dneg = tops.distance_tensor_redux(dist, self.redux.neg)

        # Losses
        loss_align = tops.mmean(dpos, mask=mask_pos, eps=self.eps)
        if numerically_friendly:
            loss_uniform = (
                self.beta
                * tops.mmean(
                    (self.b - self.gamma * dneg).exp(), mask=mask_neg, eps=self.eps
                )
            ).log1p()
        else:
            loss_uniform = (
                tops.mmean((-self.gamma * dneg).exp(), mask=mask_neg, eps=self.eps)
                + self.epsilon
            ).log()

        # Output
        loss = loss_align + loss_uniform
        logdict = {
            "l_main": loss,
            "l_cent": loss_align,
            "l_cont": loss_uniform,
            "v_dpos": tops.mmean(dpos, mask=mask_pos),
            "v_dneg": tops.mmean(dneg, mask=mask_neg),
            "v_zmax": z.abs().max(),
            "v_zmean": z.mean(),
            "v_zstd": z.std(),
        }
        return loss, logdict

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
            redux_strategy = self.redux.pos
        # Reshape and compute
        sq = q.size(1)
        sc = c.size(1)
        q = rearrange(q, "b s c -> (b s) c")
        c = rearrange(c, "b s c -> (b s) c")
        dist = tops.pairwise_distance_matrix(q, c, mode="nsqeuc")
        dist = rearrange(dist, "(bq sq) (bc sc) -> bq bc sq sc", sq=sq, sc=sc)
        if qmask is not None and cmask is not None:
            qmask = rearrange(qmask, "b s -> (b s)")
            cmask = rearrange(cmask, "b s -> (b s)")
            mask = qmask.view(-1, 1) | cmask.view(1, -1)
            mask = rearrange(mask, "(bq sq) (bc sc) -> bq bc sq sc", sq=sq, sc=sc)
        else:
            mask = None
        # Redux
        dist = tops.distance_tensor_redux(dist, redux_strategy, mask=mask)
        return dist
