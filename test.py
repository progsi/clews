import sys
import os
import importlib
from omegaconf import OmegaConf
import torch, math
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy

from lib import eval, dataset
from lib import tensor_ops as tops
from utils import pytorch_utils, print_utils

# --- Get arguments (and set defaults) --- Basic ---
args = OmegaConf.from_cli()
assert "checkpoint" in args
log_path, _ = os.path.split(args.checkpoint)
if "ngpus" not in args:
    args.ngpus = 1
if "nnodes" not in args:
    args.nnodes = 1
args.precision = "32"
if "path_audio" not in args:
    args.path_audio = None
if "path_meta" not in args:
    args.path_meta = None
if "partition" not in args:
    args.partition = "test"
if "limit_num" not in args:
    args.limit_num = None

# --- Get arguments (and set defaults) --- Tunable ---
if "maxlen" not in args:  # maximum audio length
    args.maxlen = 10 * 60  # in seconds
if "redux" not in args:  # distance reduction
    args.redux = None
if "qslen" not in args:  # query shingle len
    args.qslen = None
if "qshop" not in args:  # query shingle hop (default = every 5 sec)
    args.qshop = 5
if "cslen" not in args:  # candidate shingle len
    args.cslen = None
if "cshop" not in args:  # candidate shingle hop (default = every 5 sec)
    args.cshop = 5

###############################################################################

# Init pytorch/Fabric
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)
fabric = Fabric(
    accelerator="cuda",
    devices=args.ngpus,
    num_nodes=args.nnodes,
    strategy=DDPStrategy(broadcast_buffers=False),
    precision=args.precision,
)
fabric.launch()

# Seed (random segment needs a seed)
fabric.barrier()
fabric.seed_everything(44 + fabric.global_rank, workers=True)

# Init my utils
myprint = lambda s, end="\n": print_utils.myprint(
    s, end=end, doit=fabric.is_global_zero
)
myprogbar = lambda it, desc=None, leave=False: print_utils.myprogbar(
    it, desc=desc, leave=leave, doit=fabric.is_global_zero
)
timer = print_utils.Timer()
fabric.barrier()

# Load conf
myprint(OmegaConf.to_yaml(args))
myprint("Load model conf...")
conf = OmegaConf.load(os.path.join(log_path, "configuration.yaml"))

# Init model
myprint("Init model...")
module = importlib.import_module("models." + conf.model.name)
with fabric.init_module():
    model = module.Model(conf.model, sr=conf.data.samplerate)
model = fabric.setup(model)

# Load model
myprint("  Load checkpoint")
state = pytorch_utils.get_state(model, None, None, conf, None, None, None)
fabric.load(args.checkpoint, state)
model, _, _, conf, epoch, _, best = pytorch_utils.set_state(state)
myprint(f"  ({epoch} epochs; best was {best:.3f})")
model.eval()
if args.path_audio is not None:
    conf.path.audio = args.path_audio
if args.path_meta is not None:
    conf.path.meta = args.path_meta
conf.data.path = conf.path

# Get dataset
myprint("Dataset...")
dset = dataset.Dataset(
    conf.data,
    args.partition,
    augment=False,
    fullsongs=True,
    verbose=fabric.is_global_zero,
)
dloader = torch.utils.data.DataLoader(
    dset,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    drop_last=False,
    pin_memory=False,
)
dloader = fabric.setup_dataloaders(dloader)

###############################################################################


@torch.inference_mode()
def extract_embeddings(shingle_len, shingle_hop, desc="Embed", eps=1e-6):
    # Check shingle args
    shinglen, shinghop = model.get_shingle_params()
    if shingle_len is not None:
        shinglen = shingle_len
    if shingle_hop is not None:
        shinghop = shingle_hop
    mxlen = int(args.maxlen * model.sr)
    numshingles = int((mxlen - int(shinglen * model.sr)) / int(shinghop * model.sr))
    # Extract embeddings
    all_c = []
    all_i = []
    all_z = []
    all_m = []
    for batch in myprogbar(dloader, desc=desc, leave=True):
        # Get info & audio
        c, i, x = batch[:3]
        if x.size(1) > mxlen:
            x = x[:, :mxlen]
        # Get embedding (B=1,S,C)
        z = model(
            x,
            shingle_len=int(x.size(1) / model.sr) if shinglen <= 0 else shinglen,
            shingle_hop=int(0.99 * x.size(1) / model.sr) if shinghop <= 0 else shinghop,
        )
        # Make embedding shingles same size
        z = tops.force_length(
            z,
            1 if shinglen <= 0 else numshingles,
            dim=1,
            pad_mode="zeros",
            cut_mode="start",
        )
        m = z.abs().max(-1)[0] < eps
        # Append
        all_c.append(c)
        all_i.append(i)
        all_z.append(z)
        all_m.append(m)
        # Limit number of queries/candidates?
        if args.limit_num is not None and len(all_z) >= args.limit_num / args.ngpus:
            myprint("")
            myprint("  [Max num reached]")
            break
    # Concat single-song batches
    all_c = torch.cat(all_c, dim=0)
    all_i = torch.cat(all_i, dim=0)
    all_z = torch.cat(all_z, dim=0)
    all_m = torch.cat(all_m, dim=0)
    # Return
    return all_c, all_i, all_z, all_m


###############################################################################

# Let's go
with torch.inference_mode():

    # Extract embeddings
    query_c, query_i, query_z, query_m = extract_embeddings(
        args.qslen, args.qshop, desc="Query emb"
    )
    query_c = query_c.int()
    query_i = query_i.int()
    query_z = query_z.half()
    if args.cslen == args.qslen and args.cshop == args.qshop:
        myprint("Cand emb: (copy)")
        cand_c, cand_i, cand_z, cand_m = (
            query_c.clone(),
            query_i.clone(),
            query_z.clone(),
            query_m.clone(),
        )
    else:
        cand_c, cand_i, cand_z, cand_m = extract_embeddings(
            args.cslen, args.cshop, desc="Cand emb"
        )
        cand_c = cand_c.int()
        cand_i = cand_i.int()
        cand_z = cand_z.half()

    # Collect candidates from all GPUs + collapse to batch dim
    fabric.barrier()
    cand_c = fabric.all_gather(cand_c)
    cand_i = fabric.all_gather(cand_i)
    cand_z = fabric.all_gather(cand_z)
    cand_m = fabric.all_gather(cand_m)
    cand_c = torch.cat(torch.unbind(cand_c, dim=0), dim=0)
    cand_i = torch.cat(torch.unbind(cand_i, dim=0), dim=0)
    cand_z = torch.cat(torch.unbind(cand_z, dim=0), dim=0)
    cand_m = torch.cat(torch.unbind(cand_m, dim=0), dim=0)

    # Evaluate
    aps = []
    r1s = []
    rpcs = []
    for n in myprogbar(range(len(query_z)), desc="Retrieve", leave=True):
        ap, r1, rpc = eval.compute(
            model,
            query_c[n : n + 1],
            query_i[n : n + 1],
            query_z[n : n + 1],
            cand_c,
            cand_i,
            cand_z,
            queries_m=query_m[n : n + 1],
            candidates_m=cand_m,
            redux_strategy=args.redux,
            batch_size_candidates=2**15,
        )
        aps.append(ap)
        r1s.append(r1)
        rpcs.append(rpc)
    aps = torch.stack(aps)
    r1s = torch.stack(r1s)
    rpcs = torch.stack(rpcs)

    # Collect measures from all GPUs + collapse to batch dim
    fabric.barrier()
    aps = fabric.all_gather(aps)
    r1s = fabric.all_gather(r1s)
    rpcs = fabric.all_gather(rpcs)
    aps = torch.cat(torch.unbind(aps, dim=0), dim=0)
    r1s = torch.cat(torch.unbind(r1s, dim=0), dim=0)
    rpcs = torch.cat(torch.unbind(rpcs, dim=0), dim=0)

###############################################################################

# Print
logdict_mean = {
    "MAP": aps.mean(),
    "MR1": r1s.mean(),
    "ARP": rpcs.mean(),
}
logdict_ci = {
    "MAP": 1.96 * aps.std() / math.sqrt(len(aps)),
    "MR1": 1.96 * r1s.std() / math.sqrt(len(r1s)),
    "ARP": 1.96 * rpcs.std() / math.sqrt(len(rpcs)),
}
myprint("=" * 100)
myprint("Result:")
myprint("  Avg --> " + print_utils.report(logdict_mean, clean_line=False))
myprint("  c.i. -> " + print_utils.report(logdict_ci, clean_line=False))
myprint("=" * 100)
