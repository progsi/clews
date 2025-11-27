import sys
import os
import argparse
from tqdm import tqdm
import torch
import multiprocessing

from utils import file_utils, audio_utils, print_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, choices=["SHS100K", "covers80", "DiscogsVI", "DVI", "DiscogsVI2", "DVI2"], required=True
)
parser.add_argument("--path_meta", type=str, default="data/xxx", required=True)
parser.add_argument("--path_audio", type=str, default="data/yyy", required=True)
parser.add_argument("--ext_in", type=str, default="mp3", required=True)
parser.add_argument(
    "--fn_out", type=str, default="cache/metadata-dataset-specs.pt", required=True
)
parser.add_argument("--njobs", type=int, default=-1)
args = parser.parse_args()
while args.ext_in[0] == ".":
    args.ext_in = args.ext_in[1:]
print("=" * 100)
print(args)
print("=" * 100)

###############################################################################


def load_cliques_shs100k(fn):
    cliques = {}
    _, data, _ = file_utils.load_csv(fn, sep="\t")
    for c, n in zip(data[0], data[1]):
        if c not in cliques:
            cliques[c] = []
        cliques[c].append(n)
    for c in cliques.keys():
        cliques[c] = list(set(cliques[c]))
        cliques[c].sort()
        for i in range(len(cliques[c])):
            cliques[c][i] = c + "-" + cliques[c][i]
    return cliques


def load_cliques_discogsvi(fn, i=0):
    jsoncliques = file_utils.load_json(fn)
    cliques = {}
    cliqueinfo = {}
    notfound = 0
    istart = i
    for c, versions in tqdm(jsoncliques.items(), total=len(jsoncliques), desc="Load cliques..."):
        clique = []
        for ver in versions:
            v = ver["version_id"]
            ytid = ver["youtube_id"]
            idx = c + ":" + v
            # search basename
            basename = None
            for pref in [
                ytid[:2],
                ytid[0].upper() + ytid[1].upper(),
                ytid[0].upper() + ytid[1].lower(),
                ytid[0].lower() + ytid[1].upper(),
                ytid[0].lower() + ytid[1].lower(),
            ]:
                fn_meta = os.path.join(args.path_audio, pref, ytid + ".meta")
                if os.path.exists(fn_meta):
                    basename = os.path.join(pref, ytid)
                    break
            if basename is None:
                notfound += 1
                continue
            # load metadata
            try:
                data = file_utils.load_json(fn_meta)
            except:
                data = {}
            # fill in now
            clique.append(idx)
            cliqueinfo[idx] = {
                "id": i,
                "clique": c,
                "version": v,
                "artist": data["artist"] if "artist" in data else None,
                "title": data["title"] if "title" in data else None,
                "filename": os.path.join(basename + "." + args.ext_in),
            }
            i += 1

        cliques[c] = clique

    print()
    return cliques, cliqueinfo, i, notfound


###############################################################################

timer = print_utils.Timer()

# Load cliques and splits
print(f"Load {args.dataset}")
if args.dataset in ["DVI", "DiscogsVI2", "DVI2"]:
    # ********* DiscogsVI **********
    # Splits + Info
    splits = {}
    info = {}
    i = 0
    for sp in ["train", "valid", "test"]:
        fn = os.path.join(args.path_meta, sp + ".json")
        cliques, infosp, i, notfound = load_cliques_discogsvi(fn, i=i)
        splits[sp] = cliques
        info.update(infosp)
        if notfound > 0:
            print(f"({sp}: Could not find {notfound} songs)")
        else:
            print(f"({sp}: Found all songs)")
            
elif args.dataset == "SHS100K":

    # ********** SHS100K **********
    # Info
    fn = os.path.join(args.path_meta, "list")
    _, data, numrecords = file_utils.load_csv(fn, sep="\t")
    info = {}
    for i in range(numrecords):
        c, n = data[0][i], data[1][i]
        idx = c + "-" + n
        info[idx] = {
            "id": i,
            "clique": c,
            "version": n,
            "artist": data[3][i],
            "title": data[2][i],
            "filename": os.path.join(idx[:2], idx + "." + args.ext_in),
        }
    # Splits
    splits = {}
    for sp, suff in zip(["train", "valid", "test"], ["TRAIN", "VAL", "TEST"]):
        fn = os.path.join(args.path_meta, "SHS100K-" + suff)
        splits[sp] = load_cliques_shs100k(fn)
    # *****************************

elif args.dataset == "DiscogsVI":

    # ********* DiscogsVI **********
    # Splits + Info
    splits = {}
    info = {}
    i = 0
    for sp, suff in zip(["train", "valid", "test"], [".train", ".val", ".test"]):
        fn = os.path.join(args.path_meta, "DiscogsVI-YT-20240701-light.json" + suff)
        cliques, infosp, i, notfound = load_cliques_discogsvi(fn, i=i)
        splits[sp] = cliques
        info.update(infosp)
        if notfound > 0:
            print(f"({sp}: Could not find {notfound} songs)")
        else:
            print(f"({sp}: Found all songs)")
            
elif args.dataset == "covers80":

    # ********* covers80 **********
    # Info
    info = {}
    for prefix in ["list1", "list2"]:
        fn = os.path.join(args.path_meta, prefix + ".list")
        with open(fn, "r") as fh:
            lines = fh.readlines()
        for line in lines:
            c, n = line[:-1].split(os.sep)
            idx = line[:-1]
            info[idx] = {
                "id": len(info),
                "clique": c,
                "version": n,
                "artist": n.split("+")[0],
                "title": n.split("-")[-1],
                "filename": os.path.join(c, n + "." + args.ext_in),
            }
    # Splits
    cliques = {}
    for idx, ifo in info.items():
        c = ifo["clique"]
        if c not in cliques:
            cliques[c] = []
        cliques[c].append(idx)
    splits = {
        "train": cliques,
        "valid": cliques,
        "test": cliques,
    }
    # *****************************

else:
    raise NotImplementedError
print(f"  Found {len(info)} songs")
nsongs = {}
for sp in splits.keys():
    nsongs[sp] = 0
    for cl, items in splits[sp].items():
        nsongs[sp] += len(items)
print("  Contains:", nsongs)

###############################################################################


def get_file_info(idx, info):
    try:
        fn = os.path.join(args.path_audio, info["filename"])
        x = audio_utils.load_audio(fn, sample_rate=16000, n_channels=1, start=0, length=None)
        if x is None or x.size(-1) < 16000:
            return None, None
        try:
            audio_info = audio_utils.get_info(fn)
        except Exception as e:
            print(f"Failed to get info for {fn}: {e}")
            return None, None

        info["samplerate"] = audio_info.samplerate
        info["length"] = audio_info.length
        info["channels"] = audio_info.channels
        return idx, info
    except Exception as e:
        print(f"Exception in get_file_info for {idx}: {e}")
        return None, None

def worker(idx):
    return get_file_info(idx, info[idx])
###############################################################################

# Filter existing ones
print(f"Filter existing")
keys = list(info.keys())
keys.sort()
if args.njobs == 1:
    done = []
    for idx in tqdm(keys):
        fn = os.path.join(args.path_audio, info[idx]["filename"])
        if os.path.exists(fn):
            done.append(get_file_info(idx, info[idx], args.path_audio))

else:
    # NOTE: I am running into issues with joblib
    # todo = []
    # for idx in keys:
    #     fn = os.path.join(args.path_audio, info[idx]["filename"])
    #     if os.path.exists(fn):
    #         job = delayed(get_file_info)(idx, info[idx], args.path_audio)
    #         todo.append(job)
    with multiprocessing.Pool(args.njobs) as pool:
        done = pool.map(worker, [idx for idx in keys if os.path.exists(os.path.join(args.path_audio, info[idx]["filename"]))])
print()
new_info = {}
for idx, inf in done:
    if idx is not None and inf is not None:
        new_info[idx] = inf
print(f"  Found {len(new_info)} songs ({100*len(new_info)/len(info):.1f}%)")
info = new_info

# Redo splits
print(f"Filter split")
new_split = {}
new_nsongs = {}
for sp in splits.keys():
    new_split[sp] = {}
    new_nsongs[sp] = 0
    for cl, items in splits[sp].items():
        new_items = []
        for idx in items:
            if idx in info:
                new_items.append(idx)
        if len(new_items) > 1:
            new_split[sp][cl] = new_items[:]
            new_nsongs[sp] += len(new_items)
print("  Contains:", new_nsongs)
for sp in nsongs.keys():
    nsongs[sp] = 100 * new_nsongs[sp] / nsongs[sp]
print("  Percent:", nsongs)
splits = new_split

# Save
print(f"Save {args.fn_out}")
torch.save({
        "info": info,
        "split": splits   
    }, args.fn_out)

# Done
print(f"Done!")
