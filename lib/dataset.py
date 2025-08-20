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
         
            
class CrossDomainDataset(Dataset):
    """
    Dataset for cross-domain evaluation.
    """
    def __init__(self, 
                 domain, 
                 conf, 
                 split, 
                 augment=False, 
                 fullsongs=False, 
                 checks=True, 
                 verbose=False, 
                 limit_cliques=None):
        super().__init__(conf, split, augment, fullsongs, checks, verbose, limit_cliques)
        # Cross-domain specific attributes
        assert domain is not None, "Domain must be specified for CrossDomainDataset"
        self.domain = domain
        self.init_domain_labels()
        if self.verbose:
            print(f"Cross-domain: {self.domain}")

    def get_all_multihot(self):
        return torch.stack(list(self.aux_processed.values()))
    
    def get_filter_mask(self, domain_label):
        """
        Filter versions based on query and candidate domains.
        """
        intlabel = self.label2ids[domain_label]
        return torch.stack(self.domains_processed)[:, intlabel] > 0
    
    def get_value_list(self, key: str) -> list:
        """Get all values for a given key."""
        return [e[key] for e in list(self.info.values())]
    
    def get_label2id_map(self, 
                         value_list: list, 
                         keep_nan: bool = False) -> dict:
        """Get a mapping from label to id for a given key."""
        labels = []
        for label in value_list:
            if isinstance(label, str) and not (label == ''):
                labels.append(label)
            elif (isinstance(label, str) and (label == '') or isinstance(label, float) and np.isnan(label)) or label is None:
                if keep_nan:
                    labels.append(NAN_LABEL)
                else:
                    continue
            elif isinstance(label, (list, tuple)):
                for slabel in label:
                    if isinstance(slabel, str):
                        labels.append(slabel)
                    else:
                        raise ValueError(f"Expected string, got: {type(slabel)}")
            elif isinstance(label, dict):
                slabels = self.matches2labels(label)
                for slabel in slabels:                    
                    labels.append(slabel)
            else:
                raise ValueError(f"Expected string or iterable of strings, got: {type(label)}")
        labels = sorted(set(labels))
        label2id = {label: idx for idx, label in enumerate(labels)}
        return label2id
    
    def matches2labels(self, match_dict: dict, ignore_cols: list = None, ignore_labels: list = None) -> list:
        """Transform match dict to list of matched labels."""
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
        """Encodes a list of integers as a multi-hot tensor."""
        items = torch.tensor(list(set(items)))
        if len(items) > 0:
            ids = torch.sum(F.one_hot(items, n), axis=0)
        else:
            ids = torch.zeros(n)
        return ids
    
    def init_domain_labels(self):
        """"""
        self.label2ids = {}
        self.aux_processed = {}
        self.cls_weights = {}
        self.domains_processed = {}
        tlabels = self.get_value_list(self.domain) 
        if self.domain == "release_styles":
            # Special case for release_styles, which is a list of tuples
            tlabels = self.join_sublists(tlabels, sep=": ")
        elif self.domain == "country":
            # TODO: implement
            pass
                    
        self.label2ids = self.get_label2id_map(tlabels, keep_nan=False)
        
        tlabels_processed = []
        for i, row in enumerate(tlabels):
            # transform string labels to ids
            ids = []
            if row is not None and row != np.nan:
                if isinstance(row, list):
                    slabels = row
                elif isinstance(row, dict):
                    slabels = self.matches2labels(row)
                for slabel in slabels:
                    ids.append(self.label2ids[slabel])  # -1 for unknown labels
            else:
                if self.label2ids.get(NAN_LABEL, False):
                    ids.append(self.label2ids[NAN_LABEL])  # -1 for unknown labels
            tlabels_processed.append(self.ints2multihot(ids, len(self.label2ids))) 
        # always keep multi-hot tensor
        self.domains_processed = torch.stack(tlabels_processed)        

        # transform
        item_keys = [v["id"] for v in self.info.values()]
        for idx, item_key in tqdm(enumerate(item_keys), desc="Transforming auxiliary data...", total=len(item_keys)):
            self.aux_processed[item_key] = tlabels_processed[idx]
        print(f"  Domain labels processed: {len(self.aux_processed):,} items, {len(self.label2ids)} classes")

    def filter_by_domain_pair(self, mode="same", qslabel=None, cslabel=None):
        doms = self.domains_processed  # [N, D] multi-hot
        assert mode in ["same", "overlap", "disjoint", "pair", "all"], f"Invalid mode: {mode}"
        if mode == "pair":
            assert qslabel is not None and cslabel is not None, "qslabel and cslabel must be specified for 'pair' mode"
            q_idx = self.label2ids[qslabel]
            c_idx = self.label2ids[cslabel]
            assert q_idx is not None and c_idx is not None, f"Labels {qslabel} and {cslabel} must be in label2ids"
        else:
            assert qslabel is None and cslabel is None, "qslabel and cslabel must be None for modes other than 'pair'"
        
        if mode == "same":
            mask = (doms == doms).all(dim=1)
        elif mode == "overlap":
            mask = (doms.unsqueeze(1) & doms.unsqueeze(0)).any(dim=-1).any(dim=1)
        elif mode == "disjoint":
            mask = ~((doms.unsqueeze(1) & doms.unsqueeze(0)).any(dim=-1)).any(dim=1)
        elif mode == "pair":
            mask = (doms[:, q_idx] | doms[:, c_idx]).bool()
        else:
            mask = torch.ones(len(self), dtype=torch.bool)

        return mask
    
    # def get_domain_mask(
    #     self,
    #     mode: str = "same",
    #     qslabel: str = None,
    #     cslabel: str = None,
    #     query_indices: torch.Tensor | list[int] = None,
    #     cand_indices: torch.Tensor | list[int] = None,
    # ) -> torch.BoolTensor:
    #     """
    #     Returns a [Q, C] boolean mask where Q = #queries, C = #candidates.
    #     If query_indices / cand_indices are provided, the mask is restricted
    #     to those dataset indices in that order.
    #     """
    #     x = self.domains_processed  # [N, D] multi-hot

    #     # resolve query subset
    #     if query_indices is not None:
    #         q_keys = [self.id2key[int(i)] for i in query_indices]
    #         key2pos = {k: i for i, k in enumerate(self.info.keys())}
    #         q_pos = [key2pos[k] for k in q_keys]
    #         q_x = x[q_pos]  # [Q, D]
    #     else:
    #         q_x = x  # [N, D]

    #     # resolve candidate subset
    #     if cand_indices is not None:
    #         c_keys = [self.id2key[int(i)] for i in cand_indices]
    #         key2pos = {k: i for i, k in enumerate(self.info.keys())}
    #         c_pos = [key2pos[k] for k in c_keys]
    #         c_x = x[c_pos]  # [C, D]
    #     else:
    #         c_x = x  # [N, D]

    #     assert mode in ["same", "overlap", "disjoint", "pair"], f"Invalid mode: {mode}"

    #     if mode == "same":
    #         # Exact match between queries and candidates
    #         mask = (q_x.unsqueeze(1) == c_x.unsqueeze(0)).all(dim=2).bool()  # [Q, C]

    #     elif mode == "overlap":
    #         # At least one shared domain label, not identical
    #         mask = (q_x @ c_x.T > 0) & ~(q_x.unsqueeze(1) == c_x.unsqueeze(0)).all(dim=2)

    #     elif mode == "disjoint":
    #         # No shared domain labels
    #         mask = ~(q_x[:, None, :] & c_x[None, :, :]).any(dim=-1)  # [Q, C]

    #     elif mode == "pair":
    #         assert qslabel is not None and cslabel is not None, \
    #             "qslabel and cslabel must be specified for 'pair' mode"
    #         q_idx = self.label2ids[qslabel]
    #         c_idx = self.label2ids[cslabel]

    #         q_sub = q_x[:, q_idx]  # [Q, |q_idx|]
    #         c_sub = c_x[:, c_idx]  # [C, |c_idx|]

    #         # Asymmetric overlap: query domains vs candidate domains
    #         mask = q_sub.bool().unsqueeze(1) & c_sub.bool().unsqueeze(0)  # [Q, C]

    #     else:
    #         raise ValueError(f"Unknown mode '{mode}'")

    #     return mask  # [Q, C]

    def get_domain_mask(
            self,
            mode: str = "same",
            qslabel: str = None,
            cslabel: str = None,
            query_i: torch.Tensor | list[int] = None,
            cand_i: torch.Tensor | list[int] = None,
            query_c: torch.Tensor | list[int] = None,
            cand_c: torch.Tensor | list[int] = None,
        ) -> tuple[torch.BoolTensor, torch.Tensor]:
        """
        Returns:
            - filtered_mask: [Q_filtered, C] boolean mask 
                (only rows with at least one valid candidate remain).
            - row_indices: [Q_filtered] tensor of indices referring to the
                surviving query rows in the *original* [Q, C] mask.

        Improvements:
            - self-matches (query == candidate index) are explicitly set to False.
        """
        x = self.domains_processed  # [N, D] multi-hot
        device = x.device

        # ----------------------
        # resolve query subset
        # ----------------------
        if query_i is not None:
            q_keys = [self.id2key[int(i)] for i in query_i]
            key2pos = {k: i for i, k in enumerate(self.info.keys())}
            q_pos = [key2pos[k] for k in q_keys]
            q_x = x[q_pos]  # [Q, D]
            if query_c is not None:
                query_c = torch.as_tensor(query_c, device=device)[q_pos]
        else:
            q_x = x  # [N, D]
            if query_c is not None:
                query_c = torch.as_tensor(query_c, device=device)

        # ----------------------
        # resolve candidate subset
        # ----------------------
        if cand_i is not None:
            c_keys = [self.id2key[int(i)] for i in cand_i]
            key2pos = {k: i for i, k in enumerate(self.info.keys())}
            c_pos = [key2pos[k] for k in c_keys]
            c_x = x[c_pos]  # [C, D]
            if cand_c is not None:
                cand_c = torch.as_tensor(cand_c, device=device)[c_pos]
        else:
            c_x = x  # [N, D]
            if cand_c is not None:
                cand_c = torch.as_tensor(cand_c, device=device)

        Q, _ = q_x.shape
        C, _ = c_x.shape

        # ----------------------
        # domain mask logic
        # ----------------------
        assert mode in ["same", "overlap", "disjoint", "pair"], f"Invalid mode: {mode}"

        if mode == "same":
            mask = (q_x.unsqueeze(1) == c_x.unsqueeze(0)).all(dim=2)  # [Q, C]

        elif mode == "overlap":
            mask = (q_x @ c_x.T > 0) & ~(q_x.unsqueeze(1) == c_x.unsqueeze(0)).all(dim=2)

        elif mode == "disjoint":
            mask = ~(q_x[:, None, :] & c_x[None, :, :]).any(dim=-1)  # [Q, C]

        elif mode == "pair":
            assert qslabel is not None and cslabel is not None, \
                "qslabel and cslabel must be specified for 'pair' mode"
            q_idx = self.label2ids[qslabel]
            c_idx = self.label2ids[cslabel]

            q_sub = q_x[:, q_idx]  # [Q, |q_idx|]
            c_sub = c_x[:, c_idx]  # [C, |c_idx|]

            mask = q_sub.bool().unsqueeze(1) & c_sub.bool().unsqueeze(0)  # [Q, C]

        # ----------------------
        # apply class ids if given
        # ----------------------
        if query_c is not None and cand_c is not None:
            class_mask = query_c[:, None] == cand_c[None, :]
            mask &= class_mask

        # ----------------------
        # remove self matches
        # ----------------------
        if query_i is not None and cand_i is not None:
            q_idx = torch.as_tensor(query_i, device=device)
            c_idx = torch.as_tensor(cand_i, device=device)
            # broadcast comparison [Q, C]
            self_mask = q_idx[:, None] == c_idx[None, :]
            mask &= ~self_mask  # set self matches to False

        # ----------------------
        # filter rows
        # ----------------------
        has_rel = mask.any(dim=1)  # [Q] bool
        row_indices = torch.nonzero(has_rel).squeeze(1)  # [Q_filtered]
        filtered_mask = mask[has_rel]  # [Q_filtered, C]

        return filtered_mask, row_indices

