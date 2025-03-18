import sys
import argparse
import torch
import torchaudio
import soxr
import warnings

###################################################################################################


def get_backend(filename):
    if filename.lower().endswith(".mp3"):
        backend = "ffmpeg"
    else:
        backend = "soundfile"
    return backend


def get_info(filename, backend=None):
    if backend is None:
        backend = get_backend(filename)
    info = argparse.Namespace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aux = torchaudio.info(filename)
    info.samplerate = aux.sample_rate
    info.length = aux.num_frames / aux.sample_rate
    info.channels = aux.num_channels
    return info


def load_audio(
    filename,
    sample_rate=None,
    n_channels=None,
    start=0,  # in samples
    length=None,  # in samples
    backend=None,
    resample_method="soxr",
    pad_till_length=False,
    pad_mode="zeros",
    safe_load=True,
    return_numpy=False,
):
    # Load
    def load():
        return torchaudio.load(
            filename,
            frame_offset=start,
            num_frames=length if length is not None else -1,
            normalize=True,  # to float32
            channels_first=True,
            backend=get_backend(filename) if backend is None else backend,
        )

    if safe_load:
        try:
            x, sr = load()
        except:
            print("\nWARNING: Could not load " + filename, flush=True)
            print(start, length, flush=True)
            return None
    else:
        try:
            x, sr = load()
        except:
            print("\nERROR: Could not load " + filename, flush=True)
            print(start, length, flush=True)
            x, sr = load()
    # Adjust channels
    if n_channels is None or n_channels == x.size(0):
        pass
    elif n_channels == 1:
        x = x.mean(0, keepdim=True)
    elif n_channels == 2:
        if x.size(0) == 1:
            x = torch.cat([x, x], dim=0)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    # Adjust sample rate
    if sample_rate is None:
        sample_rate = sr
    elif sr != sample_rate:
        x = resample(x, sr, sample_rate, method=resample_method)
    # Pad length
    if pad_till_length and length > x.size(1):
        if pad_mode == "zeros":
            x = torch.nn.functional.pad(
                x, (0, length - x.size(1)), mode="constant", value=0
            )
        elif pad_mode == "repeat":
            aux = torch.cat([x, x], dim=-1)
            while aux.size(-1) < length:
                aux = torch.cat([aux, x], dim=-1)
            x = aux[:, :length]
        else:
            raise NotImplementedError
    # Done
    if return_numpy:
        return x.numpy()
    return x


###################################################################################################


def resample(audio, in_sr, out_sr, method="soxr", prevent_clip=True):
    # audio is (C,T) or (B,T)
    audio *= 0.5
    if method == "soxr":
        audio = (
            torch.FloatTensor(soxr.resample(audio.T.numpy(), in_sr, out_sr))
            .to(audio.device)
            .T
        )
    elif method == "torchaudio":
        audio = torchaudio.functional.resample(audio, orig_freg=in_sr, new_freq=out_sr)
    else:
        raise NotImplementedError
    audio *= 2
    if prevent_clip:
        mx = audio.abs().max(-1, keepdim=True)[0]
        audio /= torch.clamp(mx, min=1)
    else:
        audio = torch.clamp(audio, -1, 1)
    return audio


###################################################################################################


def get_frames(x, win=10, hop=1, dimstack=1):
    frames = []
    for i in range(0, x.size(-1) - win + 1, hop):
        frames.append(x[..., i : i + win])
    return torch.stack(frames, dim=dimstack)


###################################################################################################
