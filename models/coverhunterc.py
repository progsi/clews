import sys
import torch
from nnAudio import features  # type: ignore
from einops import rearrange, repeat

from lib.coverhunter import ch_conformer, ch_layers, ch_losses
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
        self.preproc = torch.nn.BatchNorm1d(self.cqtbins)
        self.backbone = ch_conformer.ConformerEncoder(
            input_size=self.cqtbins,
            output_size=self.conf.ncha,
            linear_units=self.conf.ncha_attn,
            num_blocks=self.conf.nblocks,
        )
        self.pool_layer = ch_layers.AttentiveStatisticsPooling(
            self.conf.ncha, output_channels=self.conf.ncha
        )
        self.bottleneck = torch.nn.BatchNorm1d(self.conf.ncha)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        # Loss
        self.ce_layer = torch.nn.Linear(
            self.conf.ncha, self.conf.maxcliques, bias=False
        )
        self.ce_loss = ch_losses.FocalLoss(alpha=None, gamma=self.conf.gamma)
        # self.ce_loss = torch.nn.CrossEntropyLoss()
        self.center_loss = ch_losses.CenterLoss(
            num_classes=self.conf.maxcliques, feat_dim=self.conf.ncha
        )
        self.triplet_loss = ch_losses.HardTripletLoss(margin=self.conf.margin)

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
        shop = self.conf.shingling.hop / 2 if shingle_hop is None else shingle_hop
        # Shingle
        h = tops.get_frames(
            h, int(self.sr * slen), int(self.sr * shop), pad_mode="repeat"
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
        h = self.preproc(h).transpose(1, 2)
        lens = torch.full(
            [h.size(0)], fill_value=h.size(1), dtype=torch.long, device=h.device
        )
        h, _ = self.backbone(h, xs_lens=lens, decoding_chunk_size=-1)
        f_t = self.pool_layer(h)
        f_i = self.bottleneck(f_t)
        f_t = rearrange(f_t, "(b s) c -> b s c", s=s)
        f_i = rearrange(f_i, "(b s) c -> b s c", s=s)
        return f_i, f_t  # (B,S,C) both

    ###########################################################################

    def loss(
        self,
        label,  # (B)
        idx,  # (B)
        f_i,  # (B,S,C)
        extra=None,
    ):
        f_t = extra
        assert len(label) == len(idx) and len(label) == len(f_t)
        s = f_t.size(1)
        f_t = rearrange(f_t, "b s c -> (b s) c")
        f_i = rearrange(f_i, "b s c -> (b s) c")
        label = rearrange(label.unsqueeze(-1).expand(-1, s), "b s -> (b s)")
        idx = rearrange(idx.unsqueeze(-1).expand(-1, s), "b s -> (b s)")
        loss_focal = self.ce_loss(self.ce_layer(f_i), label)
        loss_center = self.center_loss(f_t, label)
        loss_triplet = self.triplet_loss(f_t, label, ids=idx)

        loss = loss_focal + 0.01 * loss_center + 0.1 * loss_triplet
        logdict = {
            "l_main": loss,
            "l_cent": loss_focal,
            "l_cont": loss_triplet,
        }
        return loss, logdict

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
