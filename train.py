import sys
import os, warnings
import math
import importlib
from omegaconf import OmegaConf
import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
import torchinfo

from lib import augmentations, eval, dataset
from utils import print_utils, pytorch_utils


#########################################################################################
# Inits
#########################################################################################

# Load config
args = OmegaConf.from_cli()
assert "jobname" in args
assert "conf" in args
conf = OmegaConf.merge(OmegaConf.load(args.conf), args)
conf.jobname = args.jobname
conf.data.path = conf.path
conf.path.logs = os.path.join(conf.path.logs, conf.jobname)
fn_ckpt_last = os.path.join(conf.path.logs, "checkpoint_last.ckpt")
fn_ckpt_best = os.path.join(conf.path.logs, "checkpoint_best.ckpt")
fn_ckpt_epoch = os.path.join(conf.path.logs, "checkpoint_$epoch$.ckpt")

if "limit_num" not in args:
    args.limit_num = None
    
# Init pytorch/Fabric
torch.backends.cudnn.benchmark = True  # seems it is same speed as False?
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)
fabric = Fabric(
    accelerator="cuda",
    devices=conf.fabric.ngpus,
    num_nodes=conf.fabric.nnodes,
    strategy=DDPStrategy(broadcast_buffers=False),
    precision=conf.fabric.precision,
    loggers=pytorch_utils.get_logger(conf.path.logs, logger = conf.logger, name = conf.jobname),
)
fabric.launch()

# Common seed to have same model everywhere
# (for different rands per GPU see re-seed below)
fabric.barrier()
fabric.seed_everything(conf.seed, workers=True)

# Init my utils
myprint = lambda s, end="\n": print_utils.myprint(
    s, end=end, doit=fabric.is_global_zero
)
myprogbar = lambda it, desc=None, leave=False: print_utils.myprogbar(
    it, desc=desc, leave=leave, doit=fabric.is_global_zero
)
timer = print_utils.Timer()

# Print config
myprint("-" * 65)
myprint(OmegaConf.to_yaml(conf)[:-1])
myprint("-" * 65)


#########################################################################################
# Model, optim, scheduler, load checkpoint...
#########################################################################################

# Init model
myprint("Init model...")
module = importlib.import_module("models." + conf.model.name)
with fabric.init_module():
    model = module.Model(conf.model, sr=conf.data.samplerate)
if fabric.is_global_zero:
    torchinfo.summary(model, depth=1)
model = fabric.setup(model)
model.mark_forward_method("prepare")
model.mark_forward_method("embed")
model.mark_forward_method("loss")

# Init optimizer & scheduler
myprint("Init optimizer...")
optim = pytorch_utils.get_optimizer(conf.training.optim, model)
optim = fabric.setup_optimizers(optim)
sched, sched_on_epoch = pytorch_utils.get_scheduler(
    conf.training.optim,
    optim,
    epochs=conf.training.numepochs,
    mode=conf.training.monitor.mode,
)

# Init local variables
myprint("Init variables...")
epoch = 0
cost_best = torch.inf if conf.training.monitor.mode == "min" else -torch.inf
if conf.training.optim.sched.startswith("plateau"):
    sched.step(cost_best)
lr = sched.get_last_lr()[0]

# Restore from previous checkpoint?
fn_ckpt = None
if conf.checkpoint is not None:
    fn_ckpt = conf.checkpoint
elif os.path.exists(fn_ckpt_last):
    fn_ckpt = fn_ckpt_last
if fn_ckpt is not None:
    myprint("Loading checkpoint...")
    state = pytorch_utils.get_state(model, optim, sched, conf, epoch, lr, cost_best)
    fabric.load(fn_ckpt, state)
    model, optim, sched, conf, epoch, lr, cost_best = pytorch_utils.set_state(state)
    myprint("  Loaded " + fn_ckpt)

# Re-seed with global_rank to have truly different augmentations
myprint("Re-seed...")
fabric.barrier()
fabric.seed_everything((epoch + 1) * (conf.seed + fabric.global_rank), workers=True)


#########################################################################################
# Data
#########################################################################################

# Dataset & augmentations
myprint("Load data...")
ds_train = dataset.Dataset(
    conf.data,
    "train",
    augment=True,
    verbose=fabric.is_global_zero,
    limit_cliques=args.limit_num,
)
ds_valid = dataset.Dataset(
    conf.data,
    "valid",
    augment=False,
    verbose=fabric.is_global_zero,
    limit_cliques=args.limit_num,
)
assert conf.training.batchsize > 1
dl_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=conf.training.batchsize,
    shuffle=True,
    num_workers=conf.data.nworkers,
    drop_last=True,
    persistent_workers=False,
    pin_memory=True,
)
dl_valid = torch.utils.data.DataLoader(
    ds_valid,
    batch_size=conf.training.batchsize,
    shuffle=False,
    num_workers=conf.data.nworkers,
    drop_last=False,
    persistent_workers=False,
    pin_memory=True,
)
dl_train, dl_valid = fabric.setup_dataloaders(dl_train, dl_valid)
augment = augmentations.Augment(conf.augmentations, sr=conf.data.samplerate)

#########################################################################################
# Main loss function
#########################################################################################


def main_loss_func(batch, logdict, training=False):
    # Prepare data
    with torch.inference_mode():
        n_per_class = (len(batch) - 1) // 2
        cc = [batch[0]] * n_per_class
        cc = torch.cat(cc, dim=0)
        ii = batch[1::2]
        ii = torch.cat(ii, dim=0)
        xx = batch[2::2]
        xx = torch.cat(xx, dim=0)
        # Export audio?
        # torch.save([cc, ii, xx], "explore.pt")
        # sys.exit()
        # Augmentations - Waveform domain
        if training:
            xx = augment.waveform(xx)
        # Model - Shingle and CQT
        xx = model.prepare(
            xx,
            shingle_hop=None if training else model.get_shingle_params()[-1] / 2,
        )
        # Augmentations - CQ domain
        if training:
            xx = augment.cqgram(xx)
    cc, ii, xx = cc.clone(), ii.clone(), xx.clone()
    # Train procedure
    if training:
        optim.zero_grad(set_to_none=True)
    zz, extra = model.embed(xx)
    loss, logdct = model.loss(cc, ii, zz, extra=extra)
    if training:
        fabric.backward(loss)
        optim.step()
        if not sched_on_epoch:
            sched.step()
    # Outputs and logdict
    with torch.inference_mode():
        clist = torch.chunk(cc, n_per_class, dim=0)
        ilist = torch.chunk(ii, n_per_class, dim=0)
        zlist = torch.chunk(zz, n_per_class, dim=0)
        outputs = [clist[0]] + [None] * (2 * n_per_class)
        outputs[1::2] = ilist
        outputs[2::2] = zlist
        logdict.append(logdct)
    return outputs, logdict


#########################################################################################
# Train/valid loops
#########################################################################################


def train_loop(desc=None):
    # Init
    model.train()
    logdict = pytorch_utils.LogDict()
    # Loop
    fabric.barrier()
    for n, batch in enumerate(myprogbar(dl_train, desc=desc)):
        if conf.limit_batches is not None and n >= conf.limit_batches:
            break
        # Regular loss calc
        _, logdict = main_loss_func(batch, logdict, training=True)
        losses = logdict.get("l_main")
        myprint(f" [L*={losses[-1]:.3f}, L={losses.mean():.3f}]", end="")
    return logdict


@torch.inference_mode()
def valid_loop(desc=None):
    # Init
    model.eval()
    logdict = pytorch_utils.LogDict()
    queries_c = []
    queries_i = []
    queries_z = []
    # Loop
    fabric.barrier()
    for n, batch in enumerate(myprogbar(dl_valid, desc=desc)):
        # if conf.limit_batches is not None and n >= conf.limit_batches:
        #     break
        # Regular loss calc
        outputs, logdict = main_loss_func(batch, logdict, training=False)
        losses = logdict.get("l_main")
        myprint(f" [L*={losses[-1]:.3f}, L={losses.mean():.3f}]", end="")
        # Keep z for evaluating MAP
        cl, i1, z1 = outputs[:3]
        queries_c.append(cl)
        queries_i.append(i1)
        queries_z.append(z1)
    queries_c = torch.cat(queries_c, dim=0)  # (B)
    queries_i = torch.cat(queries_i, dim=0)  # (B)
    queries_z = torch.cat(queries_z, dim=0)  # (B,C) or (B,S,C)
    # Gather all multi-gpu tensors
    fabric.barrier()
    all_c = fabric.all_gather(queries_c)  # (N,B)
    all_i = fabric.all_gather(queries_i)  # (N,B)
    all_z = fabric.all_gather(queries_z)  # (N,B,C) or (N,B,S,C)
    all_c = torch.cat(torch.unbind(all_c, dim=0), dim=0)
    all_i = torch.cat(torch.unbind(all_i, dim=0), dim=0)
    all_z = torch.cat(torch.unbind(all_z, dim=0), dim=0)
    # Eval kNN
    myprint("Eval... ", end="")
    aps, r1s, rpcs = eval.compute(
        model,
        queries_c,
        queries_i,
        queries_z,
        all_c,
        all_i,
        all_z,
    )
    comp = (rpcs * (1 - aps)) ** 0.5
    logdict.append({"m_MAP": aps, "m_MR1": r1s, "m_ARP": rpcs, "m_COMP": comp})
    return logdict


#########################################################################################
# Main loop
#########################################################################################

# Main loop (epoch)
myprint("Training...")
stop = None
start_epoch = epoch
for epoch in range(start_epoch, conf.training.numepochs):
    desc = f"{epoch+1:{len(str(conf.training.numepochs))}d}/{conf.training.numepochs}"
    fabric.log("hpar/epoch", epoch + 1, step=epoch + 1)
    # Train
    logdict_train = train_loop(desc="Train " + desc)
    logdict_train.sync_and_mean(fabric)
    fabric.log_dict(logdict_train.get(prefix="train/"), step=epoch + 1)
    # Valid
    logdict_valid = valid_loop(desc="Valid " + desc)
    logdict_valid.sync_and_mean(fabric)
    fabric.log_dict(logdict_valid.get(prefix="valid/"), step=epoch + 1)
    # Report & check NaN/inf
    tmp = logdict_valid.get(keys=["l_main", "m_MAP", "m_ARP", "m_COMP"])
    tmp["l_main_t"] = logdict_train.get("l_main")
    report = print_utils.report(tmp, desc=f"[{timer.time()}] Epoch {desc}")
    for aux in tmp.values():
        if math.isnan(aux) or math.isinf(aux):
            stop = "NaN or inf reached!"
            break
    # Get current cost
    cost_current = logdict_valid.get(conf.training.monitor.quantity)
    # Optimizer schedule?
    fabric.log("hpar/lr", lr, step=epoch + 1)
    if sched_on_epoch:
        if conf.training.optim.sched.startswith("plateau"):
            sched.step(cost_current)
        else:
            with warnings.catch_warnings():
                # otherwise it warns about passing the epoch number (?)
                warnings.simplefilter("ignore")
                sched.step()
        new_lr = sched.get_last_lr()[0]
        if new_lr != lr:
            if conf.training.optim.sched.startswith("plateau"):
                report += f"  (lr={new_lr:.1e})"
            lr = new_lr
    if "min_lr" in conf.training.optim and lr < conf.training.optim.min_lr:
        stop = "Min lr reached."
    # Checkpoint & best
    if (
        conf.training.save_freq is not None
        and (epoch + 1) % conf.training.save_freq == 0
    ):
        fn = fn_ckpt_epoch.replace("$epoch$", "epoch" + str(epoch + 1))
        state = pytorch_utils.get_state(
            model, optim, sched, conf, epoch + 1, lr, cost_best
        )
        fabric.save(fn, state)
    if (conf.training.monitor.mode == "max" and cost_current > cost_best) or (
        conf.training.monitor.mode == "min" and cost_current < cost_best
    ):
        cost_best = cost_current
        state = pytorch_utils.get_state(
            model, optim, sched, conf, epoch + 1, lr, cost_best
        )
        fabric.save(fn_ckpt_best, state)
        report += "  *"
    state = pytorch_utils.get_state(model, optim, sched, conf, epoch + 1, lr, cost_best)
    fabric.save(fn_ckpt_last, state)
    # Done
    myprint(report)
    if stop is not None:
        myprint(stop + " Stop.")
        break

#########################################################################################
