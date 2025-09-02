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
        print("Getting offsets...")
        for i, cl in enumerate(self.clique.keys()):
            self.clique2id[cl] = offset + i

        # Get idx2version
        print("Getting versions...")
        self.versions = []
        for clique, vers in self.clique.items():
            if not clique in vers[0]:
                vers = [clique + ":" + v for v in vers]
            self.versions.extend(vers)
        
        # print("Post-fitering...")
        # self.info = {k: v for k, v in self.info.items() if k in self.versions}
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
        print("Initializing domains...")
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
        self.label2ids_map = {}
        self.domains_processed = {}

        num_items = len(self.info)
        items_list = list(self.info.values())  # iterate only once

        for domain in tqdm(self.DOMAINS):
            # --- Collect unique labels ---
            unique_labels = set()
            all_item_labels = []
            for item in items_list:
                val = item.get(domain, None)
                if isinstance(val, dict):
                    labels = self.matches2labels(val)
                elif isinstance(val, (list, tuple)):
                    labels = [v[1] if isinstance(v, tuple) else v for v in val]
                else:
                    labels = [val]
                labels = [str(l) for l in labels if l is not None]
                unique_labels.update(labels)
                all_item_labels.append(labels)

            # --- Map labels to IDs ---
            labels_sorted = sorted(unique_labels)
            self.label2ids_map[domain] = {label: idx for idx, label in enumerate(labels_sorted)}

            # --- Preallocate multi-hot matrix ---
            num_labels = len(labels_sorted)
            multi_hot_matrix = torch.zeros((num_items, num_labels), dtype=torch.long)

            for item_idx, labels in enumerate(all_item_labels):
                ids = [self.label2ids_map[domain][l] for l in labels if l in self.label2ids_map[domain]]
                if ids:
                    multi_hot_matrix[item_idx, ids] = 1

            # Store whole matrix instead of per-item tensors
            self.domains_processed[domain] = multi_hot_matrix

        print(f"Processed domains: {list(self.domains_processed.keys())}")

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

    def get_filter_mask_full(
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

            # prepare indexing for later
            all_keys = list(self.info.keys())                    
            key2pos = {k: i for i, k in enumerate(all_keys)}
            N = len(all_keys)

            # resolve query / candidate subsets -> positions in original order to index domain tensor later
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
                domain_tensor = self.domains_processed[domain]
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

                domain_pair_mask = q_mask.unsqueeze(1) & c_mask.unsqueeze(0).to(q_mask.device)  # [Q, C]
                mask = mask & domain_pair_mask

            # set diagonal (self-matches) to False
            if query_i is not None and cand_i is not None:
                q_idx_tensor = torch.as_tensor(list(query_i), device=device)
                c_idx_tensor = torch.as_tensor(list(cand_i), device=device)
                self_mask = q_idx_tensor[:, None] == c_idx_tensor[None, :]
                mask = mask & ~self_mask

            # relevance filtering: keep rows with at least one relevant candidate and at least one non-relevant (exclude no-rel and all-rel)
            if query_c is not None and cand_c is not None:
                query_c = query_c.to(device)
                cand_c = cand_c.to(device)
                rel_mask = query_c[:, None] == cand_c[None, :]
                # consider only candidates allowed by domain mask
                rel_mask = rel_mask & mask
                # keep only queries that have at least one relevant AND at least one non-relevant
                has_rel = rel_mask.any(dim=1) & (~rel_mask).any(dim=1)
            else:
                # fallback: keep queries that have at least one allowed candidate and at least one disallowed
                has_rel = mask.any(dim=1) & (~mask).any(dim=1)

            # map local indices back to original positions
            if len(has_rel) == 0:
                local_idxs = torch.empty(0, dtype=torch.long, device=device)
                filtered_mask = mask[has_rel]
            else:
                local_idxs = torch.where(has_rel)[0]  # indices into q_pos (0..Q-1)
                filtered_mask = mask[local_idxs, :]  # [Q_filtered, C]

            return filtered_mask, local_idxs
        
    def get_filter_mask_batchwise(
        self,
        query_filter_str=None,
        candidate_filter_str=None,
        query_i=None,
        cand_i=None,
        query_c=None,
        cand_c=None,
        device="cuda",
        batch_size: int = 4096
    ):
        """
        Batchwise version of get_filter_mask to avoid OOM.
        batch_size controls how many query rows are processed at once.
        """
        # --- normalize filter inputs ---
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

        all_keys = list(self.info.keys())
        key2pos = {k: i for i, k in enumerate(all_keys)}
        N = len(all_keys)

        q_pos = [key2pos[self.id2key[int(i)]] for i in query_i] if query_i is not None else list(range(N))
        c_pos = [key2pos[self.id2key[int(i)]] for i in cand_i] if cand_i is not None else list(range(N))

        Q, C = len(q_pos), len(c_pos)

        # Start with full mask
        mask = torch.ones((Q, C), dtype=torch.bool, device=device)

        # --- domain filtering (batchwise) ---
        filter_domains = set(query_filter.keys()) | set(candidate_filter.keys())

        def normalize_filter_values(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                return list(v)
            return [v]

        for domain in filter_domains:
            if domain not in self.label2ids_map:
                continue
            D = len(self.label2ids_map[domain])
            if D == 0:
                continue

            # build domain_tensor [N, D]
            domain_tensor = self.domains_processed[domain]

            q_x = domain_tensor[q_pos]
            c_x = domain_tensor[c_pos]

            q_vals = normalize_filter_values(query_filter.get(domain, None))
            c_vals = normalize_filter_values(candidate_filter.get(domain, None))

            def label_tokens_to_indices(tokens):
                if tokens is None:
                    return []
                idxs = []
                mapdict = self.label2ids_map[domain]
                for t in tokens:
                    if t in mapdict:
                        idxs.append(mapdict[t])
                        continue
                    ts = str(t)
                    if ts in mapdict:
                        idxs.append(mapdict[ts])
                return sorted(set(idxs))

            q_idxs = label_tokens_to_indices(q_vals)
            c_idxs = label_tokens_to_indices(c_vals)

            def submask_matrix(x, idxs):
                if not idxs:
                    return torch.ones(x.shape[0], dtype=torch.bool, device=device)
                cols = x[:, idxs]
                return cols.any(dim=1)

            q_mask = submask_matrix(q_x, q_idxs)
            c_mask = submask_matrix(c_x, c_idxs)

            # batchwise combination
            for start in range(0, Q, batch_size):
                end = min(start + batch_size, Q)
                mask[start:end] &= q_mask[start:end, None] & c_mask[None, :].to(q_mask.device)

        # --- self matches (batchwise) ---
        if query_i is not None and cand_i is not None:
            q_idx_tensor = torch.as_tensor(list(query_i), device=device)
            c_idx_tensor = torch.as_tensor(list(cand_i), device=device)
            for start in range(0, Q, batch_size):
                end = min(start + batch_size, Q)
                mask[start:end] &= ~(q_idx_tensor[start:end, None] == c_idx_tensor[None, :])

        # --- relevance filtering (batchwise) ---
        if query_c is not None and cand_c is not None:
            query_c = query_c.to(device)
            cand_c = cand_c.to(device)
            has_rel = torch.zeros(Q, dtype=torch.bool, device=device)
            for start in range(0, Q, batch_size):
                end = min(start + batch_size, Q)
                rel_chunk = (query_c[start:end, None] == cand_c[None, :]) & mask[start:end]
                has_rel[start:end] = rel_chunk.any(dim=1) & (~rel_chunk).any(dim=1)
        else:
            has_rel = torch.zeros(Q, dtype=torch.bool, device=device)
            for start in range(0, Q, batch_size):
                end = min(start + batch_size, Q)
                chunk = mask[start:end]
                has_rel[start:end] = chunk.any(dim=1) & (~chunk).any(dim=1)

        local_idxs = torch.where(has_rel)[0]
        filtered_mask = mask[local_idxs, :]

        return filtered_mask, local_idxs

    def get_filter_mask(
        self,
        query_filter_str=None,
        candidate_filter_str=None,
        query_i=None,
        cand_i=None,
        query_c=None,
        cand_c=None,
        device="cuda",
        batch_size: int = 4096,
        threshold: int = 1_000_000_000  # elements in Q*C above which we use batchwise
    ):
        """
        Chooses full or batchwise get_filter_mask depending on Q*C threshold.
        """
        # determine Q and C
        all_keys = list(self.info.keys())
        N = len(all_keys)
        Q = len(query_i) if query_i is not None else N
        C = len(cand_i) if cand_i is not None else N
        total_elements = Q * C

        if total_elements > threshold:
            print(f"[INFO] Using batchwise version: Q*C={total_elements} > threshold={threshold}")
            return self.get_filter_mask_batchwise(
                query_filter_str=query_filter_str,
                candidate_filter_str=candidate_filter_str,
                query_i=query_i,
                cand_i=cand_i,
                query_c=query_c,
                cand_c=cand_c,
                device=device,
                batch_size=batch_size
            )
        else:
            print(f"[INFO] Using full version: Q*C={total_elements} <= threshold={threshold}")
            return self.get_filter_mask_full(
                query_filter_str=query_filter_str,
                candidate_filter_str=candidate_filter_str,
                query_i=query_i,
                cand_i=cand_i,
                query_c=query_c,
                cand_c=cand_c,
                device=device
            )