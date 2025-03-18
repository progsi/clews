import sys
import torch
from nnAudio import features  # type: ignore
from einops import rearrange, repeat

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
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (12, 3), padding=(6, 0), bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (13, 3), dilation=(1, 2), bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2), (1, 2)),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, (13, 3), bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), dilation=(1, 2), bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2), (1, 2)),
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, (3, 3), bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, (3, 3), dilation=(1, 2), bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2), (1, 2)),
        )
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, (3, 3), bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, (3, 3), dilation=(1, 2), bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2), (1, 2)),
        )
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, (3, 3), bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, (3, 3), dilation=(1, 2), bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.fc0 = torch.nn.Linear(512, self.conf.zdim)
        # Loss
        self.fc1 = torch.nn.Linear(self.conf.zdim, self.conf.maxcliques)

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
            h, int(self.sr * self.minlen), dim=-1, pad_mode="zeros", allow_longer=True
        )
        # CQT
        s = h.size(1)
        h = rearrange(h, "b s t -> (b s) t")
        h = self.cqt(h)
        h = self.cqtpool(h)
        h = rearrange(h, "(b s) c t -> b s c t", s=s)
        return h

    def embed(
        self,
        h,
    ):
        assert h.ndim == 4
        s = h.size(1)
        h = rearrange(h, "b s c t -> (b s) c t")
        h = h / (h.abs().max(1, keepdim=True)[0].max(2, keepdim=True)[0] + self.eps)
        h = h.unsqueeze(1)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = h.squeeze(-1).squeeze(-1)
        h = self.fc0(h)
        h = rearrange(h, "(b s) c -> b s c", s=s)
        return h, None  # (B,C)

    ###########################################################################

    def loss(
        self,
        label,  # (B)
        idx,  # (B)
        z,  # (B,S,C)
        extra=None,
    ):
        assert len(label) == len(idx) and len(label) == len(z)
        z = rearrange(z, "b s t -> (b s) t")
        logits = self.fc1(z)
        loss = torch.nn.functional.cross_entropy(logits, label)
        logd = {
            "l_main": loss,
            "l_cent": loss,
        }
        return loss, logd

    ###########################################################################

    def distances(
        self,
        q,  # (B,C)
        c,  # (B',C)
        qmask=None,
        cmask=None,
        redux_strategy=None,
    ):
        assert q.ndim == 3 and c.ndim == 3 and q.size(-1) == c.size(-1)
        if redux_strategy is None:
            redux_strategy = "min"
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
