import sys, os, argparse
import importlib
from omegaconf import OmegaConf
import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from tqdm import tqdm

from utils import pytorch_utils, audio_utils

ACCEPTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--path_in", type=str, default=None)
parser.add_argument("--path_out", type=str, default=None)
parser.add_argument("--fn_in", type=str, default=None)
parser.add_argument("--fn_out", type=str, default=None)
parser.add_argument("--hop_size", type=float, default=5.0)
parser.add_argument("--win_len", type=float, default=-1)  # will use model's default
args = parser.parse_args()
if args.win_len <= 0:
    args.win_len = None
using_paths = args.path_in is not None and args.path_out is not None
using_filenames = args.fn_in is not None and args.fn_out is not None
conflicting = (using_paths and (args.fn_in is not None or args.fn_out is not None)) or (
    using_filenames and (args.path_in is not None or args.path_out is not None)
)
if (not (using_paths or using_filenames)) and conflicting:
    print(
        "ERROR: You should provide either path_in/path_out or fn_in/fn_out (and only these combinations)."
    )
    print(
        '       Use either "--path_in=xxx --path_out=yyy" or "--fn_in=xxx.wav --fn_out=yyy.pt".'
    )
    sys.exit()
if using_paths:
    args.path_in = os.path.abspath(args.path_in)
    args.path_out = os.path.abspath(args.path_out)
elif using_filenames:
    args.fn_in = os.path.abspath(args.fn_in)
    args.fn_out = os.path.abspath(args.fn_out)
print("=" * 100)
print(args)
print("=" * 100)

###############################################################################

# Init output path
if using_paths and os.path.exists(args.path_out):
    print("*** Output path exists (" + args.path_out + ") ***")
    print("By hitting enter it will be erased and the script will continue. ")
    input("[Enter to continue/CTRL-C to exit]")
    os.system("rm -rf " + args.path_out)

# Init pytorch/Fabric
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)
fabric = Fabric(
    accelerator="cuda",
    devices=1,
    num_nodes=1,
    strategy=DDPStrategy(broadcast_buffers=False),
    precision="32",
)
fabric.launch()

# Load conf
print("Load model conf...")
path_checkpoint, _ = os.path.split(args.checkpoint)
conf = OmegaConf.load(os.path.join(path_checkpoint, "configuration.yaml"))

# Init model
print("Init model...")
module = importlib.import_module("models." + conf.model.name)
with fabric.init_module():
    model = module.Model(conf.model, sr=conf.data.samplerate)
model = fabric.setup(model)

# Load model
print("Load checkpoint...")
state = pytorch_utils.get_state(model, None, None, conf, None, None, None)
fabric.load(args.checkpoint, state)
model, _, _, conf, _, _, _ = pytorch_utils.set_state(state)
model.eval()

###############################################################################

# Get all files
print("Get filenames...")
if using_paths:
    filenames = []
    for path, dirs, files in os.walk(args.path_in):
        for file in files:
            # Filter audio files
            _, ext = os.path.splitext(file)
            if ext.lower() not in ACCEPTED_AUDIO_EXTENSIONS:
                continue
            # Get full filename
            fn_in = os.path.join(path, file)
            fn_out = os.path.join(args.path_out, os.path.relpath(fn_in, args.path_in))
            path_out, _ = os.path.split(fn_out)
            filenames.append([fn_in, path_out, fn_out])
else:
    path_out, _ = os.path.split(args.fn_out)
    filenames = [[args.fn_in, path_out, args.fn_out]]

# Extract
with torch.inference_mode():
    for fn_in, path_out, fn_out in tqdm(
        filenames, ascii=True, ncols=100, desc="Extract embeddings"
    ):
        # Load mono audio
        x = audio_utils.load_audio(fn_in, sample_rate=model.sr, n_channels=1)
        if x is None:
            continue
        # Compute embeddings
        z = model(x, shingle_hop=args.hop_size, shingle_len=args.win_len)
        z = z.cpu()
        # Save
        os.makedirs(path_out, exist_ok=True)
        torch.save(z, fn_out)

###############################################################################

print("Done.")
