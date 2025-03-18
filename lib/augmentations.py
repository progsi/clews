import sys
import torch, math
import torchaudio, julius
from torchvision import transforms
from einops import rearrange

from lib import tensor_ops as tops


class Augment:

    def __init__(self, conf, sr=22050, random_order=True, eps=1e-6):
        self.conf = conf
        self.sr = sr
        self.random_order = random_order
        self.eps = eps

    def waveform(
        self,
        x,  # (B,T)
        noise=None,
    ):
        assert x.ndim == 2
        # Randomize augmentations
        augs = list(self.conf.keys())
        if self.random_order:
            ids = torch.randperm(len(augs)).tolist()
        else:
            ids = list(range(len(augs)))
        # Waveform domain augmentations
        for i in ids:
            if augs[i] == "polarity" and self.conf.polarity.p > 0:

                # --- Polarity augmentation ---
                mask = torch.rand(x.size(0), 1, device=x.device) < self.conf.polarity.p
                x = torch.where(mask, -x, x)

            elif augs[i] == "gain" and self.conf.gain.p > 0:

                # --- Gain augmentation ---
                rmin, rmax = self.conf.gain.r
                r = rmin + (rmax - rmin) * torch.rand(x.size(0), 1, device=x.device)
                r /= x.abs().max(-1, keepdim=True)[0] + self.eps
                mask = torch.rand(x.size(0), 1, device=x.device) < self.conf.gain.p
                x = torch.where(mask, (x * r).clamp(min=-1, max=1), x)
                del r

            elif augs[i] == "noise" and self.conf.noise.p > 0 and noise is not None:

                # --- Noise augmentation ---
                rmin, rmax = self.conf.noise.snr
                r = rmin + (rmax - rmin) * torch.rand(x.size(0), 1, device=x.device)
                r = 10 ** (r / 20)
                xnorm = x / (x.pow(2).mean(1, keepdim=True) + self.eps)
                nnorm = noise / (noise.pow(2).mean(1, keepdim=True) + self.eps)
                xnew = r * xnorm + nnorm
                xnew /= xnew.abs().max(1, keepdim=True)[0] + self.eps
                xnew *= x.abs().max(1, keepdim=True)[0]
                mask = torch.rand(x.size(0), 1, device=x.device) < self.conf.noise.p
                x = torch.where(mask, xnew, x)
                del r, xnorm, nnorm, xnew

            elif augs[i] == "reqtime" and self.conf.reqtime.p > 0:

                # --- Random EQ (time) ---
                # TODO: Really slow. Also unchecked.
                nfmin, nfmax = self.conf.reqtime.nfreqs
                nf = torch.randint(nfmin, nfmax + 1, (1,)).item()
                gmin, gmax = self.conf.reqtime.gains
                fmin, fmax = math.log(20), math.log(self.sr * 0.5 * 0.98)
                qmin, qmax = self.conf.reqtime.qrange
                qmin, qmax = math.log(qmin), math.log(qmax)
                xeq = x.clone()
                for _ in range(nf):
                    gain = gmin + (gmax - gmin) * torch.rand(1).item()
                    freq = math.exp(fmin + (fmax - fmin) * torch.rand(1).item())
                    q = math.exp(qmin + (qmax - qmin) * torch.rand(1).item())
                    xeq = torchaudio.functional.equalizer_biquad(
                        xeq, self.sr, freq, gain, Q=q
                    )
                mask = torch.rand(x.size(0), 1, device=x.device) < self.conf.reqtime.p
                x = torch.where(mask, xeq.clamp(min=-1, max=1), x)
                del xeq

            elif augs[i] == "clipping" and self.conf.clipping.p > 0:

                # --- Clipping augmentation ---
                qtl = (
                    1
                    - torch.rand(x.size(0), 1, device=x.device)
                    * self.conf.clipping.max_qtl
                )
                thres = tops.tensor_quantile(x.abs(), qtl, dim=-1, keepdim=True)
                mask = (
                    torch.rand(x.size(0), 1, device=x.device)
                    < self.conf.clipping.p_soft
                )
                xclip = torch.tanh(x * 2 / (thres + self.eps)) * thres
                xclip = torch.where(mask, xclip, x.clamp(min=-thres, max=thres))
                mask = torch.rand(x.size(0), 1, device=x.device) < self.conf.clipping.p
                x = torch.where(mask, xclip, x)
                del qtl, thres, xclip

            elif augs[i] == "length" and self.conf.length.p > 0:

                # --- Length augmentation ---
                if torch.rand(1).item() < self.conf.length.p:
                    rmin = self.conf.length.rmin
                    r = rmin + (1 - rmin) * torch.rand(1).item()
                    x = x[:, : int(r * x.size(-1))]

            elif augs[i] == "compexp" and self.conf.compexp.p > 0:

                # --- Basic compression/expansion augmentation ---
                rmin, rmax = self.conf.compexp.r
                r = rmin + (rmax - rmin) * torch.rand(x.size(0), 1, device=x.device)
                mask = torch.rand(x.size(0), 1, device=x.device) < self.conf.compexp.p
                x = torch.where(mask, x.sign() * (x.abs() ** r), x)

        return x  # (B,T)

    def cqgram(
        self,
        y,  # (B,C,T) or (B,N,C,T)
    ):
        assert y.ndim == 3 or y.ndim == 4
        # Reshape?
        if y.ndim == 4:
            nnn = y.size(1)
            y = rearrange(y, "b n c t -> (b n) c t")
        else:
            nnn = None
        # Randomize augmentations
        augs = list(self.conf.keys())
        if self.random_order:
            ids = torch.randperm(len(augs)).tolist()
        else:
            ids = list(range(len(augs)))
        # CQT domain augmentations
        for i in ids:

            if augs[i] == "specaugment" and self.conf.specaugment.p > 0:

                # --- Specaugment ---
                ydrop = y.clone()
                n = torch.randint(1, self.conf.specaugment.n + 1, (1,)).item()
                for _ in range(n):
                    fpc = (
                        torch.rand(y.size(0), 1, 1, device=y.device)
                        * self.conf.specaugment.f_pc
                    )
                    flen = (fpc * y.size(1)).clamp(min=1).long()
                    fmax = y.size(1) - flen
                    f0 = (torch.rand_like(fpc) * fmax).long()
                    tpc = (
                        torch.rand(y.size(0), 1, 1, device=y.device)
                        * self.conf.specaugment.t_pc
                    )
                    tlen = (tpc * y.size(2)).clamp(min=1).long()
                    tmax = y.size(2) - tlen
                    t0 = (torch.rand_like(tpc) * tmax).long()
                    fids = torch.arange(0, y.size(1), device=y.device).view(1, -1, 1)
                    tids = torch.arange(0, y.size(2), device=y.device).view(1, 1, -1)
                    if self.conf.specaugment.full:
                        cond = ((fids >= f0) & (fids < f0 + flen)) | (
                            (tids >= t0) & (tids < t0 + tlen)
                        )
                    else:
                        cond = ((fids >= f0) & (fids < f0 + flen)) & (
                            (tids >= t0) & (tids < t0 + tlen)
                        )
                    ydrop = torch.where(cond, 0, ydrop)
                mask = (
                    torch.rand(y.size(0), 1, 1, device=y.device)
                    < self.conf.specaugment.p
                )
                y = torch.where(mask, ydrop, y)
                del ydrop

            elif augs[i] == "timestretch" and self.conf.timestretch.p > 0:

                # --- Time stretch ---
                rmin, rmax = self.conf.timestretch.r
                ys = y.clone()
                for j in range(ys.size(0)):
                    r = rmin + (rmax - rmin) * torch.rand(1).item()
                    length = int(ys.size(2) * r)
                    if ys.size(2) != length:
                        resize = transforms.Resize((ys.size(1), length))
                        aux = resize(ys[j : j + 1].unsqueeze(1)).squeeze(1)
                        ys[j : j + 1] = tops.force_length(
                            aux,
                            ys.size(2),
                            dim=2,
                            pad_mode=self.conf.timestretch.pad_mode,
                            cut_mode=self.conf.timestretch.cut_mode,
                        )
                mask = (
                    torch.rand(y.size(0), 1, 1, device=y.device)
                    < self.conf.timestretch.p
                )
                y = torch.where(mask, ys, y)
                del ys

            elif augs[i] == "pitchstretch" and self.conf.pitchstretch.p > 0:

                # --- Pitch stretch ---
                rmin, rmax = self.conf.pitchstretch.r
                ys = y.clone()
                for j in range(ys.size(0)):
                    r = rmin + (rmax - rmin) * torch.rand(1).item()
                    length = int(ys.size(1) * r)
                    if ys.size(1) != length:
                        resize = transforms.Resize((length, ys.size(2)))
                        aux = resize(ys[j : j + 1].unsqueeze(1)).squeeze(1)
                        ys = tops.force_length(
                            aux,
                            ys.size(1),
                            dim=1,
                            pad_mode=self.conf.pitchstretch.pad_mode,
                            cut_mode=self.conf.pitchstretch.cut_mode,
                        )
                mask = (
                    torch.rand(y.size(0), 1, 1, device=y.device)
                    < self.conf.pitchstretch.p
                )
                y = torch.where(mask, ys, y)
                del ys

            elif augs[i] == "pitchtranspose" and self.conf.pitchtranspose.p > 0:

                # --- Pitch transposition ---
                rmin, rmax = self.conf.pitchtranspose.r
                yt = torch.zeros_like(y)
                for j in range(yt.size(0)):
                    r = torch.randint(rmin, rmax + 1, (1,)).item()
                    if r == 0:
                        yt[j, :, :] = y[j, :, :]
                    elif r > 0:
                        yt[j, r:, :] = y[j, :-r, :]
                    else:
                        yt[j, : -abs(r), :] = y[j, abs(r) :, :]
                mask = (
                    torch.rand(y.size(0), 1, 1, device=y.device)
                    < self.conf.pitchtranspose.p
                )
                y = torch.where(mask, yt, y)
                del yt

            elif augs[i] == "reqcqt" and self.conf.reqcqt.p > 0:

                # --- Random EQ (CQT) ---
                rmin, rmax = self.conf.reqcqt.r
                r = torch.cumsum(torch.randn(y.size(0), y.size(1), device=y.device), 1)
                r = julius.lowpass.lowpass_filter(
                    r, self.conf.reqcqt.lpf, zeros=4
                ).unsqueeze(-1)
                r -= r.min(1, keepdim=True)[0]
                r /= r.max(1, keepdim=True)[0] + self.eps
                r *= rmin + (rmax - rmin) * torch.rand(r.size(0), 1, 1, device=y.device)
                r = 10 ** (r / 10)
                mask = torch.rand(y.size(0), 1, 1, device=y.device) < self.conf.reqcqt.p
                y = torch.where(mask, y * r, y)

        # Reshape
        if nnn is not None:
            y = rearrange(y, "(b n) c t -> b n c t", n=nnn)
        return y  # (B,C,T) or (B,N,C,T), same as input
