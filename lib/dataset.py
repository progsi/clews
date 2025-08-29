import sys
import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from utils import audio_utils
from lib import tensor_ops as tops


NAN_LABEL = "(None)"

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        conf,
        split,
        augment=False,
        fullsongs=False,
        checks=True,
        verbose=False,
        limit_cliques=None
    ):
        assert split in ("train", "valid", "test")
        # Params
        self.augment = augment
        self.samplerate = conf.samplerate
        self.fullsongs = fullsongs
        self.audiolen = conf.audiolen if not self.fullsongs else None
        self.maxlen = conf.maxlen if not self.fullsongs else None
        self.pad_mode = conf.pad_mode
        self.n_per_class = conf.n_per_class
        self.p_samesong = conf.p_samesong
        self.verbose = verbose
        # Load metadata
        print(f"Loading metadata from {conf.path.meta}...")
        meta = torch.load(conf.path.meta)
        if isinstance(meta, dict):
            self.info = meta["info"]
            splitdict = meta["split"]
        elif isinstance(meta, list):
            self.info, splitdict = meta
        if limit_cliques is None:
            self.clique = splitdict[split]
        else:
            if self.verbose:
                print(f"[Limiting cliques to {limit_cliques}]")
            self.clique = {}
            for key, item in splitdict[split].items():
                self.clique[key] = item
                if len(self.clique) == limit_cliques:
                    break

        # Update filename with audio_path
        prefix = conf.path.audio.rstrip(os.sep) + os.sep   # guarantees exactly one final /
        for ver in tqdm(self.info.values(), desc="Updating filenames...", total=len(self.info)):
            ver["filename"] = prefix + ver["filename"]
            
        # Checks
        if checks:
            print("Performing checks...")
            self.perform_checks(splitdict, split)
        # Get clique id
        self.clique2id = {}
        if split == "train":
            offset = 0
        elif split == "valid":
            offset = len(splitdict["train"])
        else:
            offset = len(splitdict["train"]) + len(splitdict["valid"])
        for i, cl in enumerate(self.clique.keys()):
            self.clique2id[cl] = offset + i
        # Get idx2version
        self.versions = []
        for clique, vers in self.clique.items():
            if not clique in vers[0]:
                vers = [clique + ":" + v for v in vers]
            self.versions += vers
        self.info = {k: v for k, v in self.info.items() if k in self.versions}
        self.id2key = {v["id"]: k for k, v in self.info.items()}
       
        # Prints
        if self.verbose:
            print(
                f"  {split}: --- Found {len(self.clique):,} cliques, {len(self.versions):,} songs ---"
            )

    ###########################################################################

    def __len__(self):
        return len(self.versions)

    def __getitem__(self, idx):
        # Get v1 (anchor) and clique
        v1 = self.versions[idx]
        i1 = self.info[v1]["id"]
        cl = self.info[v1]["clique"]
        icl = self.clique2id[cl]
        # Get other versions from same clique
        otherversions = []
        for v in self.clique[cl]:
            if v != v1 or torch.rand(1).item() < self.p_samesong:
                if cl in v:
                    otherversions.append(v)
                else:
                    otherversions.append(cl + ":" + v)
        if self.augment:
            new_vers = []
            for k in torch.randperm(len(otherversions)).tolist():
                new_vers.append(otherversions[k])
            otherversions = new_vers
        # Construct v1..vn array (n_per_class)
        v_n = [v1]
        i_n = [i1]
        for k in range(self.n_per_class - 1):
            v = otherversions[k % len(otherversions)]
            i = self.info[v]["id"]
            v_n.append(v)
            i_n.append(i)
        # Time augment?
        s_n = []
        for v in v_n:
            if self.augment:
                dur = self.info[v]["length"]
                if self.maxlen is not None:
                    dur = min(self.maxlen, dur)
                start = max(0, torch.rand(1).item() * (dur - self.audiolen))
            else:
                start = 0
            s_n.append(start)
        # Load audio and create output
        output = [icl]
        for i, v, s in zip(i_n, v_n, s_n):
            fn = self.info[v]["filename"]
            x = self.get_audio(fn, start=s, length=self.audiolen)
            output += [i, x]
            if self.fullsongs:
                return output
        return output

    ###########################################################################

    def get_indices(self, cliques=None):
        return [
            self.info[v]["id"] for v in self.versions if cliques is None or self.info[v]["clique"] in cliques
        ]
        
    def get_audio(self, fn, start=0, length=None):
        start = int(start * self.samplerate)
        length = None if length is None else int(length * self.samplerate)
        # Load
        x = audio_utils.load_audio(
            fn,
            self.samplerate,
            n_channels=1,
            start=start,
            length=length,
            pad_till_length=False,  # will pad below
            backend="ffmpeg",
            safe_load=False,
        ).squeeze(0)
        if length is not None and length > 0:
            x = tops.force_length(
                x,
                length,
                dim=-1,
                pad_mode=self.pad_mode,
                cut_mode="random" if self.augment else "start",
            )
        return x

    ###########################################################################

    def perform_checks(self, splitdict, split):
        msg = ""
        errors = False
        # # Max id
        # mx = -1
        # for v in self.info.keys():
        #     if self.info[v]["id"] > mx:
        #         mx = self.info[v]["id"]
        # msg += f"\n  {split}: Max ID = {mx}"
        # Cliques have at least 2 versions
        for cl in self.clique.keys():
            if len(self.clique[cl]) < 2:
                msg += f"\n  {split}: Clique {cl} has < 2 versions"
                errors = True
        # No overlap between partitions
        for cl in splitdict[split].keys():
            for partition in ["train", "valid", "test"]:
                if split == partition:
                    continue
                if cl in splitdict[partition]:
                    msg += (
                        f"\n  {split}: Clique {cl} is both in {split} and {partition}"
                    )
                    # errors=True
        if self.verbose and len(msg) > 1:
            print(msg[1:])
        if errors:
            sys.exit()
         
            
class FilterableDataset(Dataset):
    """
    Initializes all auxiliary labels automatically and supports complex filtering.
    """
    def __init__(self, conf, split, augment=False, fullsongs=False, checks=True, verbose=False, limit_cliques=None):
        super().__init__(conf, split, augment, fullsongs, checks, verbose, limit_cliques)
        self.DOMAINS = [
            "matched_concepts", 
            "matched_instruments_groups",
            "matched_segments",
            "release_styles", 
            "release_genres", 
            "dvi"]
        self.verbose = verbose
        self.label2ids_map = {}
        self.domains_processed = {}
        self.init_domain_labels()
        if self.verbose:
            print(f"Cross-domain dataset initialized with domains: {self.DOMAINS}")

    @staticmethod
    def join_sublists(lst, sep=": "):
        """
        Flattens nested lists/tuples into strings joined by sep.
        """
        result = []
        for item in lst:
            if isinstance(item, (list, tuple)):
                result.append(sep.join(str(sub) for sub in item))
            elif isinstance(item, str):
                result.append(item)
            elif item is None or (isinstance(item, float) and np.isnan(item)):
                continue
            else:
                result.append(str(item))
        return result
    
    def init_domain_labels(self):
        """
        Initialize all auxiliary domain labels generically.
        - domains: list of domain keys to process; if None, defaults to standard set.
        """

        self.label2ids_map = {}
        self.domains_processed = {}

        for domain in self.DOMAINS:
            # --- Collect all values for this domain ---
            tlabels = []
            for item in self.info.values():
                val = item.get(domain, None)
                if isinstance(val, dict):
                    # convert dict to list of keys
                    tlabels.append(self.matches2labels(val))
                elif isinstance(val, (list, tuple)):
                    # flatten tuples if needed (e.g. release_styles)
                    tlabels.append([v[1] if isinstance(v, tuple) else v for v in val])
                else:
                    # boolean or single value
                    tlabels.append([val])

            # --- Build label2id mapping ---
            # NOTE: might look weird in the case of dvi (True/False) but works fine
            labels = sorted({str(l) for sublist in tlabels for l in sublist if l is not None})
            self.label2ids_map[domain] = {label: idx for idx, label in enumerate(labels)}

            # --- Build multi-hot tensor per item ---
            for item_idx, item in enumerate(self.info.values()):
                row = tlabels[item_idx]  # list of labels
                ids = [self.label2ids_map[domain][str(l)] for l in row if str(l) in self.label2ids_map[domain]]
                multi_hot = torch.zeros(len(self.label2ids_map[domain]), dtype=torch.long)
                if len(ids) > 0:
                    multi_hot[ids] = 1
                # store in aux map
                self.domains_processed.setdefault(item["id"], {})[domain] = multi_hot

        print(f"Processed domains: {list(self.domains_processed[next(iter(self.domains_processed))].keys())}")

    # ------------------- Generic utility functions -------------------
    def get_value_list(self, key: str) -> list:
        return [e.get(key, None) for e in self.info.values()]

    def get_label2id_map(self, value_list: list) -> dict:
        labels = []
        for val in value_list:
            if isinstance(val, str) and val != '':
                labels.append(val)
            elif isinstance(val, bool):
                labels.append(val)
            elif isinstance(val, (list, tuple)):
                labels.extend([v for v in val if isinstance(v, str)])
            elif isinstance(val, dict):
                labels.extend(self.matches2labels(val))
        labels = sorted(set(labels))
        return {l: i for i, l in enumerate(labels)}

    @staticmethod
    def matches2labels(match_dict: dict, ignore_cols: list = None, ignore_labels: list = None) -> list:
        labels = []
        for col, matches in match_dict.items():
            if ignore_cols and col in ignore_cols:
                continue
            for label in matches.keys():
                if ignore_labels and label in ignore_labels:
                    continue
                labels.append(label)
        return labels

    @staticmethod
    def ints2multihot(items: list, n: int) -> torch.Tensor:
        items = torch.tensor(list(set(items)))
        if len(items) > 0:
            return torch.sum(F.one_hot(items, n), dim=0)
        return torch.zeros(n)

    # ------------------- Filter by query/candidate -------------------
    @staticmethod
    def parse_filter_str(filter_str: str) -> dict:
        """
        Parse a filter string into a dictionary.
        """
        result = {}
        if not filter_str:
            return result

        # Split domains by ';'
        domain_parts = filter_str.split(";")
        for part in domain_parts:
            if ":" not in part:
                continue
            key, value_str = part.split(":", 1)
            key = key.strip()
            value_str = value_str.strip()

            # Split by ',' if multiple values
            if "," in value_str:
                values = [v.strip() for v in value_str.split(",")]
                result[key] = values
            else:
                # Try to convert to int or float if possible
                try:
                    num = int(value_str)
                    result[key] = num
                except ValueError:
                    try:
                        num = float(value_str)
                        result[key] = num
                    except ValueError:
                        result[key] = value_str
        return result

    def get_filter_mask(
            self,
            query_filter_str: dict | str | None = None,
            candidate_filter_str: dict | str | None = None,
            query_i: torch.Tensor | list[int] = None,
            cand_i: torch.Tensor | list[int] = None,
            query_c: torch.Tensor | list[int] = None,
            cand_c: torch.Tensor | list[int] = None,
            device: torch.device = "cuda"
        ) -> tuple[torch.BoolTensor, torch.Tensor]:
        """
        Generic pair-mode mask using arbitrary domains. Filters can be dicts or strings.
        Returns:
        - filtered_mask: [Q_filtered, C] boolean mask of candidates allowed by filters
        - row_indices: indices of queries in the original dataset order (torch.LongTensor)
        """

        # --- normalize filter inputs to dicts ---
        if isinstance(query_filter_str, str):
            query_filter = self.parse_filter_str(query_filter_str)
        elif isinstance(query_filter_str, dict):
            query_filter = query_filter_str
        else:
            query_filter = {}

        if isinstance(candidate_filter_str, str):
            candidate_filter = self.parse_filter_str(candidate_filter_str)
        elif isinstance(candidate_filter_str, dict):
            candidate_filter = candidate_filter_str
        else:
            candidate_filter = {}

        # prepare indexing and sizes
        all_keys = list(self.info.keys())                     # original order keys
        key2pos = {k: i for i, k in enumerate(all_keys)}
        N = len(all_keys)

        # resolve query / candidate subsets -> positions in original order
        if query_i is not None:
            # query_i are dataset ids/indices that map via id2key -> key -> pos
            q_keys = [self.id2key[int(i)] for i in query_i]
            q_pos = [key2pos[k] for k in q_keys]
        else:
            q_pos = list(range(N))

        if cand_i is not None:
            c_keys = [self.id2key[int(i)] for i in cand_i]
            c_pos = [key2pos[k] for k in c_keys]
        else:
            c_pos = list(range(N))

        Q = len(q_pos)
        C = len(c_pos)

        # start with all-True mask
        mask = torch.ones((Q, C), dtype=torch.bool, device=device)

        # domains requested by either filter
        filter_domains = set(query_filter.keys()) | set(candidate_filter.keys())

        # helper: normalize filter value into a list of tokens (handles single bool/int/str)
        def normalize_filter_values(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                return list(v)
            return [v]

        # iterate domains from filters; ignore unknown domains gracefully
        for domain in filter_domains:
            if domain not in self.label2ids_map:
                # unknown domain -> skip (alternatively: raise)
                # print(f"[get_filter_mask] unknown domain '{domain}' -> ignoring")
                continue

            # number of classes for this domain
            D = len(self.label2ids_map[domain])
            if D == 0:
                # nothing to filter on this domain
                continue

            # build domain_tensor [N, D] by stacking per-item vectors from self.domains_processed
            domain_rows = []
            for k in all_keys:
                entry = self.domains_processed.get(self.info[k]["id"], None)
                if entry and domain in entry and entry[domain] is not None:
                    vec = entry[domain]
                    # ensure vector is 1D of length D; convert to bool
                    vec = vec.to(torch.bool).to(device)
                    # if vec.numel() != D:
                    #     # if shapes mismatch, try to pad/trim
                    #     v = torch.zeros(D, dtype=torch.bool, device=device)
                    #     length = min(vec.numel(), D)
                    #     v[:length] = vec.view(-1)[:length].to(device).to(torch.bool)
                    #     vec = v
                else:
                    vec = torch.zeros(D, dtype=torch.bool, device=device)
                domain_rows.append(vec)
            domain_tensor = torch.stack(domain_rows, dim=0)  # [N, D]

            # now build pair mask for this domain for the chosen subsets q_pos / c_pos
            q_x = domain_tensor[q_pos]  # [Q, D]
            c_x = domain_tensor[c_pos]  # [C, D]

            # get requested filter values for this domain
            q_vals = normalize_filter_values(query_filter.get(domain, None))
            c_vals = normalize_filter_values(candidate_filter.get(domain, None))

            # helper: convert label token -> indices in label2ids_map[domain]
            def label_tokens_to_indices(tokens):
                if tokens is None:
                    return []
                idxs = []
                mapdict = self.label2ids_map[domain]
                for t in tokens:
                    # try direct membership (works for bools/ints/strings stored as keys)
                    if t in mapdict:
                        idxs.append(mapdict[t])
                        continue
                    # try str conversion
                    ts = str(t)
                    if ts in mapdict:
                        idxs.append(mapdict[ts])
                        continue
                    # try numeric -> string fallback (e.g. label "1")
                    # ignore if not found
                return sorted(set(idxs))

            q_idxs = label_tokens_to_indices(q_vals)
            c_idxs = label_tokens_to_indices(c_vals)

            # submask creation: if no filter values specified for side -> allow all
            def submask_matrix(x, idxs):
                # x: [rows, D]
                if not idxs:
                    return torch.ones(x.shape[0], dtype=torch.bool, device=device)
                cols = x[:, idxs]    # [rows, len(idxs)]
                return cols.any(dim=1)  # [rows]

            q_mask = submask_matrix(q_x, q_idxs)  # [Q]
            c_mask = submask_matrix(c_x, c_idxs)  # [C]

            domain_pair_mask = q_mask.unsqueeze(1) & c_mask.unsqueeze(0)  # [Q, C]
            mask &= domain_pair_mask

        # remove self matches if both query_i and cand_i were provided
        if query_i is not None and cand_i is not None:
            # use original provided ids for equality comparison (keeps prior behavior)
            q_idx_tensor = torch.as_tensor(list(query_i), device=device)
            c_idx_tensor = torch.as_tensor(list(cand_i), device=device)
            self_mask = q_idx_tensor[:, None] == c_idx_tensor[None, :]
            mask &= ~self_mask

        # relevance filtering: keep rows with at least one relevant candidate and at least one non-relevant (exclude no-rel and all-rel)
        if query_c is not None and cand_c is not None:
            query_c = query_c.to(device)
            cand_c = cand_c.to(device)
            rel_mask = query_c[:, None] == cand_c[None, :]
            # consider only candidates allowed by domain mask
            rel_mask &= mask
            # keep only queries that have at least one relevant AND at least one non-relevant
            has_rel = rel_mask.any(dim=1) & (~rel_mask).any(dim=1)
        else:
            # fallback: keep queries that have at least one allowed candidate and at least one disallowed
            has_rel = mask.any(dim=1) & (~mask).any(dim=1)

        # map local indices back to original positions
        if len(has_rel) == 0:
            row_indices = torch.empty(0, dtype=torch.long, device=device)
            filtered_mask = mask[has_rel]
        else:
            local_idxs = torch.where(has_rel)[0]  # indices into q_pos (0..Q-1)
            row_indices = torch.as_tensor(q_pos, dtype=torch.long, device=device)[local_idxs]  # positions in original order
            filtered_mask = mask[local_idxs, :]  # [Q_filtered, C]

        return filtered_mask, row_indices