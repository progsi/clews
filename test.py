import sys
import os
import importlib
from omegaconf import OmegaConf
import torch, math
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler

from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy

from lib import eval, dataset
from lib import tensor_ops as tops
from utils import pytorch_utils, print_utils, file_utils

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
if "domain" not in args:
    args.domain = None
if "domain_mode" not in args:
    args.domain_mode = None
if "qsdomain" not in args:
    args.qsdomain = None
if "csdomain" not in args:
    args.csdomain = None
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
    args.qshop = None
if "cslen" not in args:  # candidate shingle len
    args.cslen = None
if "cshop" not in args:  # candidate shingle hop (default = every 5 sec)
    args.cshop = None

# Set save path
test_subset = args.jobname.split("-")[-1]
sub_path = f"hs{args.qshop}ws{args.qslen}" if not (args.qslen is None or args.qshop is None) else "full_track"
save_path = os.path.join(log_path, test_subset, sub_path)
os.makedirs(save_path, exist_ok=True)

###############################################################################

# Init pytorch/Fabric
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)
fabric = Fabric(
    accelerator="gpu",
    devices=args.ngpus,
    num_nodes=args.nnodes,
    strategy=DDPStrategy(broadcast_buffers=False, timeout=timedelta(milliseconds=18_000_000)),
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
# conf = OmegaConf.load(os.path.join(log_path, "configuration.yaml"))
conf = OmegaConf.load(args.conf)


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
if args.domain is not None:
    myprint(f"Using cross-domain dataset with domain {args.domain}...")
    dset = dataset.CrossDomainDataset(
        args.domain,
        conf.data,
        args.partition,
        augment=False,
        fullsongs=True,
        verbose=fabric.is_global_zero,
        limit_cliques=args.limit_num,
        checks=False, 
    )
    if args.domain_mode is not None:
        myprint(f"Using domain mode {args.domain_mode}...")
        cmask = dset.get_domain_mask(args.domain_mode)
        eval_name = f"{args.domain_mode}"
    elif args.qsdomain is not None and args.csdomain is not None:
        cmask = dset.get_domain_mask("pair", args.qsdomain, args.csdomain)
        eval_name = f"{args.qsdomain}-to-{args.csdomain}"
else:
    myprint("Using normal dataset...")
    dset = dataset.Dataset(
        conf.data,
        args.partition,
        augment=False,
        fullsongs=True,
        verbose=fabric.is_global_zero,
        limit_cliques=args.limit_num,
        checks=False, 
    )
    cmask = None
    eval_name = ""

sampler = DistributedSampler(dset, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=False)
dloader = torch.utils.data.DataLoader(
    dset,
    sampler=sampler,
    batch_size=1,
    num_workers=8,
    pin_memory=True,
)
dloader = fabric.setup_dataloaders(dloader)

###############################################################################

@torch.inference_mode()
def extract_embeddings(shingle_len, shingle_hop, outpath, eps=1e-6):

    mxlen = int(args.maxlen * model.sr)
    if shingle_len is None and shingle_hop is None:
        numshingles = 1
    else:
        if shingle_len is None:
            shingle_len, _ = model.get_shingle_params()
        elif shingle_hop is None:
            _, shingle_hop = model.get_shingle_params()
        numshingles = 1 + int((mxlen - int(shingle_len * model.sr)) / int(shingle_hop * model.sr))

    skipped = 0
    total_saved = 0
    buffer = {"clique": [], "index": [], "z": [], "m": []}

    for step, batch in enumerate(myprogbar(dloader, desc="Extracting embeddings...", leave=True)):
        c, i, x = batch[:3]
        if x.size(1) > mxlen:
            x = x[:, :mxlen]

        try:
            xlen = int(x.size(1) / model.sr)
            z = model(
                x,
                shingle_len=xlen if (shingle_len is None or shingle_len <= 0) else shingle_len,
                shingle_hop=xlen if (shingle_hop is None or shingle_hop <= 0) else shingle_hop,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                myprint(f"[OOM] Skipping batch of shape {x.shape}")
                skipped += 1
                torch.cuda.empty_cache()
                continue
            else:
                raise

        z = tops.force_length(z, numshingles, dim=1, pad_mode="zeros", cut_mode="start")
        m = z.abs().max(-1)[0] < eps

        # Move to CPU immediately
        buffer["clique"].append(c.cpu())
        buffer["index"].append(i.cpu())
        buffer["z"].append(z.cpu())
        buffer["m"].append(m.cpu())

    myprint(f"Skipped {skipped} items.")
    all_clique = tops.all_gather_chunks(torch.cat(buffer["clique"], dim=0), fabric, chunk_size=1024)
    all_index = tops.all_gather_chunks(torch.cat(buffer["index"], dim=0), fabric, chunk_size=1024)
    all_z = tops.all_gather_chunks(torch.cat(buffer["z"], dim=0), fabric, chunk_size=1024)
    all_m = tops.all_gather_chunks(torch.cat(buffer["m"], dim=0), fabric, chunk_size=1024)
    print("Gathered embeddings of shape:", all_z.shape)
    
    if fabric.global_rank == 0 and outpath is not None:
        file_utils.save_to_hdf5(
            outpath,
            {
                "clique": all_clique,
                "index": all_index,
                "z": all_z,
                "m": all_m,
            },
            batch_start=total_saved,
            hop=shingle_hop
        )
        myprint(f"Saved checkpoint at batch {step + 1} → total {total_saved} samples")

        # Clear buffer and memory
        for k in buffer:
            buffer[k].clear()
        torch.cuda.empty_cache()

    
###############################################################################

def evaluate(batch_size_candidates=2**15, cmask=None):
    # Let's go
    with torch.inference_mode():

        need_seperate_candidates = not args.cslen == args.qslen or not args.cshop == args.qshop
        # Extract embeddings
        if args.jobname is not None:
            outp_embs_q = os.path.join(save_path, f"embeddings_q.h5")
            if need_seperate_candidates:
                outp_embs_c = os.path.join(save_path, f"embeddings_c.h5")
            else:
                outp_embs_c = None
        else:
            outp_embs_q, outp_embs_c = None
        
        expected_len = len(dloader)
        if outp_embs_q is None or not os.path.isfile(outp_embs_q):
            # change path to make sure we do not extract for the whole dataset
            extract_embeddings(
                args.qslen, args.qshop, outpath=outp_embs_q
            )
        
        if fabric.global_rank == 0:
            query_c, query_i, query_z, query_m, qhop = file_utils.load_from_hdf5(outp_embs_q)
        else:
            query_c = query_i = query_z = query_m = qhop = None

        # Then broadcast from rank 0 to others (assuming Fabric, DDP, or Torch distributed)
        query_c = fabric.broadcast(query_c, src=0)
        query_i = fabric.broadcast(query_i, src=0)
        query_z = fabric.broadcast(query_z, src=0)
        query_m = fabric.broadcast(query_m, src=0)
        
        if len(query_i) < expected_len:
            myprint(f"Warning: expected {expected_len} queries, got {len(query_i)}")
        elif len(query_i) > expected_len:
            indices = torch.tensor(dset.get_indices(), dtype=torch.long)
            mask = torch.isin(query_i, indices)
            query_i = query_i[mask]
            query_c = query_c[mask]
            query_z = query_z[mask]
            query_m = query_m[mask]
        
        if qhop != args.qshop:
            myprint(f"Reducing query windows from {qhop} to {args.qshop} seconds...")
            query_z = tops.reduce_windows(query_z, qhop, args.qshop, dim=1)
            query_m = tops.reduce_windows(query_m, qhop, args.qshop, dim=1)

        print(f"Having query embeddings of shape: {query_z.shape}")
            
        query_c = query_c.int()
        query_i = query_i.int()
        query_z = query_z.half()
        

        if not need_seperate_candidates:
            myprint("Cand emb: (copy)")
            cand_c, cand_i, cand_z, cand_m = (
                query_c.clone(),
                query_i.clone(),
                query_z.clone(),
                query_m.clone(),
            )
        else:
            if outp_embs_q is None or not os.path.isfile(outp_embs_c):
                extract_embeddings(
                    args.qslen, args.qshop, outpath=outp_embs_c
                )
            cand_c, cand_i, cand_z, cand_m, chop = file_utils.load_from_hdf5(outp_embs_c)
            if len(cand_i) < expected_len:
                myprint(f"Warning: expected {expected_len:,} queries, got {len(query_i):,}")
            elif len(cand_i) > expected_len:
                cand_i = cand_i[mask]
                cand_c = cand_c[mask]
                cand_z = cand_z[mask]
                cand_m = cand_m[mask]
            if chop != args.cshop:
                myprint(f"Reducing candidate windows from {qhop} to {args.qshop} seconds...")
                query_z = tops.reduce_windows(query_z, qhop, args.qshop, dim=1)
                query_m = tops.reduce_windows(query_m, qhop, args.qshop, dim=1)
            print(f"Having candidate embeddings of shape: {cand_z.shape}")

            cand_c = cand_c.int()
            cand_i = cand_i.int()
            cand_z = cand_z.half()
        
        # Collect candidates from all GPUs + collapse to batch dim
        fabric.barrier()
        cand_c = tops.all_gather_chunks(cand_c.cpu(), fabric, chunk_size=1024)
        cand_i = tops.all_gather_chunks(cand_i.cpu(), fabric, chunk_size=1024)
        cand_z = tops.all_gather_chunks(cand_z.cpu(), fabric, chunk_size=1024)
        cand_m = tops.all_gather_chunks(cand_m.cpu(), fabric, chunk_size=1024)

        # Evaluate
        my_queries = range(fabric.global_rank, len(query_z), fabric.world_size)
        step = 0
        total_saved = 0
        buffer = {"clique": [], "index": [], "aps": [], "r1s": [], "rpcs": [], "ncands": []}
        if cmask is None:
            outpath = os.path.join(save_path, f"measures_{fabric.global_rank}.h5")
        else:
            outpath = os.path.join(save_path, f"measures_{args.domain_mode}_{fabric.global_rank}.h5")

        for n in myprogbar(my_queries, desc=f"Retrieve (GPU {fabric.global_rank})", leave=True):
            if cmask is not None:
                if (cmask[n].sum() == 0) or ((query_c[n : n + 1].unsqueeze(1) == cand_c[cmask[n]]).sum() <= 1).item():
                    continue  # skip if no valid or positive candidates
                
            ap, r1, rpc = eval.compute(
                model,
                query_c[n : n + 1],
                query_i[n : n + 1],
                query_z[n : n + 1],
                cand_c[cmask[n]] if cmask is not None else cand_c,
                cand_i[cmask[n]] if cmask is not None else cand_i,
                cand_z[cmask[n]] if cmask is not None else cand_z,
                queries_m=query_m[n : n + 1],
                candidates_m=cand_m[cmask[n]] if cmask is not None else cand_m,
                redux_strategy=args.redux,
                batch_size_candidates=batch_size_candidates,
            )

            # Move to CPU immediately
            buffer["clique"].append(query_i[n : n + 1].cpu())
            buffer["index"].append(query_c[n : n + 1].cpu())
            buffer["aps"].append(ap.cpu())
            buffer["r1s"].append(r1.cpu())
            buffer["rpcs"].append(rpc.cpu())
            cur_ncands = torch.tensor([torch.sum(cmask[n], dtype=torch.int32)]) if cmask is not None else torch.tensor([len(cand_i)])
            buffer["ncands"].append(cur_ncands.cpu())
        
            if outpath is not None and (step + 1) % 1000 == 0 or step == len(my_queries) - 1:
                file_utils.save_to_hdf5(
                    outpath,
                    {
                        "clique": torch.cat(buffer["clique"], dim=0),
                        "index": torch.cat(buffer["index"], dim=0),
                        "aps": torch.cat(buffer["aps"], dim=0),
                        "r1s": torch.cat(buffer["r1s"], dim=0),
                        "rpcs": torch.cat(buffer["rpcs"], dim=0),
                        "ncands": torch.cat(buffer["ncands"], dim=0),
                    },
                    batch_start=total_saved,
                    hop=args.qshop,
                )
                myprint(f"MAP {torch.stack(buffer['aps']).mean().item():.3f}; MR1 {torch.stack(buffer['r1s']).mean().item():.3f}. Saved measures at batch {step + 1}.")
            step += 1
                       
        aps = torch.stack(buffer["aps"])
        r1s = torch.stack(buffer["r1s"])
        rpcs = torch.stack(buffer["rpcs"])
        ncands = torch.stack(buffer["ncands"])

        # Collect measures from all GPUs + collapse to batch dim
        fabric.barrier()
        aps = tops.all_gather_chunks(aps.cpu(), fabric, chunk_size=1024)
        r1s = tops.all_gather_chunks(r1s.cpu(), fabric, chunk_size=1024)
        rpcs = tops.all_gather_chunks(rpcs.cpu(), fabric, chunk_size=1024)
        ncands = tops.all_gather_chunks(ncands.cpu(), fabric, chunk_size=1024)

    ###############################################################################

    # Print
    logdict_mean = {
        # evaluation measures
        "MAP": aps.mean(),
        "MR1": r1s.mean(),
        "ARP": rpcs.mean(),
    }
    logdict_stats = {
        # number of queries
        "nQs": len(aps), 
    }
    if cmask is not None:
        logdict_stats["nCs_median"] = ncands.float().median().int().item()
        logdict_stats["nCs_mean"] = ncands.float().median().item()
        logdict_stats["nCs_std"] = ncands.float().median().item()
        logdict_stats["nCs_min"] = ncands.float().median().int().item()
        logdict_stats["nCs_max"] = ncands.float().median().int().item()

    logdict_ci = {
        # confidence intervals for evaluation measures
        "MAP": 1.96 * aps.std() / math.sqrt(len(aps)),
        "MR1": 1.96 * r1s.std() / math.sqrt(len(r1s)),
        "ARP": 1.96 * rpcs.std() / math.sqrt(len(rpcs)),
    }
    myprint("=" * 100)
    myprint("Result " + eval_name)
    myprint("  Avg --> " + print_utils.report(logdict_mean, clean_line=False))
    myprint("  c.i. -> " + print_utils.report(logdict_ci, clean_line=False))
    myprint("  Stats -->" + print_utils.report(logdict_stats, sep="\n", clean_line=False))
    myprint("=" * 100)

if args.domain is not None:
    myprint(f"Evaluating cross-domain dataset with domain {args.domain}...")
    if args.domain_mode is not None:
        print(f"Results for {args.domain_mode} domains:")
    elif args.qsdomain is not None and args.csdomain is not None:
        print(f"Results for {args.qsdomain}-to-{args.csdomain}:")
else:
        print("Overall results:")

evaluate(batch_size_candidates=2**15, cmask=cmask)
