import sys
import torch

from utils.file_utils import load_from_h5_by_index, load_from_h5_by_indices

###################################################################################################


@torch.inference_mode()
def compute_from_disk(
    model,
    h5_path_q,
    index_q,
    total_c,
    redux_strategy=None,
    h5_path_c=None,
    batch_size_c=1024,
):
    """
    Compute retrieval metrics (AP, R@1, RPC) for a single query at index_q against all candidates,
    loading candidates from disk in batches (to avoid OOM).
    """

    # Load single query
    query_z, query_m, query_i, query_c = load_from_h5_by_index(h5_path_q, index_q)
    # query_c = query_c.int()
    # query_i = query_i.int()
    # query_z = query_z.half()

    # Default to self-retrieval if no candidate path is provided
    candidate_path = h5_path_c or h5_path_q

    dist_list = []
    match_clique_all = []

    model.eval()

    for start in range(0, total_c, batch_size_c):
        end = min(start + batch_size_c, total_c)

        cand_z, cand_m, cand_i, cand_c = load_from_h5_by_indices(candidate_path, start, end)
        # cand_c = cand_c.int()
        # cand_i = cand_i.int()
        # cand_z = cand_z.half()

        dists = model.distances(
            query_z,
            cand_z,
            qmask=query_m,
            cmask=cand_m,
            redux_strategy=redux_strategy,
        ).squeeze(0)  # shape: (batch_size,)

        match_clique = (cand_c == query_c.item())  # shape: (batch_size,)
        match_query = (cand_i == query_i.item())   # shape: (batch_size,)

        dist_list.append(torch.where(match_query, torch.inf, dists))
        match_clique_all.append(torch.where(match_query, False, match_clique))

    dist = torch.cat(dist_list, dim=0)
    match_clique = torch.cat(match_clique_all, dim=0)

    # Compute metrics
    ap = average_precision(dist, match_clique)
    r1 = rank_of_first_correct(dist, match_clique)
    rpc = rank_percentile(dist, match_clique)

    return ap.unsqueeze(0), r1.unsqueeze(0), rpc.unsqueeze(0)

@torch.inference_mode()
def compute(
    model,
    queries_c,  # clique index (B)
    queries_i,  # song index (B)
    queries_z,  # embedding (B,S,C)
    candidates_c,  # clique index (B')
    candidates_i,  # song index (B')
    candidates_z,  # embedding (B',S,C)
    queries_m=None,
    candidates_m=None,
    redux_strategy=None,
    batch_size_candidates=None,
    device="cuda",
):
    # Prepare
    aps = []
    r1s = []
    rpcs = []
    model.eval()
    for n in range(len(queries_i)):
        # Compute distance between query and everything
        if batch_size_candidates is None or batch_size_candidates >= len(candidates_i):
            dist = model.distances(
                queries_z[n : n + 1].float().to(device),
                candidates_z.float().to(device),
                qmask=queries_m[n : n + 1].to(device) if queries_m is not None else None,
                cmask=candidates_m.to(device),
                redux_strategy=redux_strategy.to(device),
            ).squeeze(0)
        else:
            dist = []
            for mstart in range(0, len(candidates_i), batch_size_candidates):
                mend = min(mstart + batch_size_candidates, len(candidates_i))
                ddd = model.distances(
                    queries_z[n : n + 1].float().to(device),
                    candidates_z[mstart:mend].float().to(device),
                    qmask=queries_m[n : n + 1].to(device) if queries_m is not None else None,
                    cmask=(
                        candidates_m[mstart:mend].to(device) if candidates_m is not None else None
                    ),
                    redux_strategy=redux_strategy,
                ).squeeze(0)
                dist.append(ddd)
            dist = torch.cat(dist, dim=-1)
        # Get ground truth
        match_clique = (candidates_c == queries_c[n]).to(dist.device)
        # Remove query from candidates if present
        match_query = (candidates_i == queries_i[n]).to(dist.device)
        dist = torch.where(match_query, torch.inf, dist)
        match_clique = torch.where(match_query, False, match_clique)
        # Compute AP and R1
        aps.append(average_precision(dist, match_clique))
        r1s.append(rank_of_first_correct(dist, match_clique))
        rpcs.append(rank_percentile(dist, match_clique))
        
        del dist
        # torch.cuda.empty_cache()  
        
    # Return as vector
    aps = torch.stack(aps)
    r1s = torch.stack(r1s)
    rpcs = torch.stack(rpcs)
    return aps, r1s, rpcs


###################################################################################################


@torch.inference_mode()
def average_precision(distances, ismatch):
    assert distances.ndim == 1 and ismatch.ndim == 1 and len(distances) == len(ismatch)
    rel = ismatch.type_as(distances)
    assert rel.sum() >= 1, "There should be at least 1 relevant item"
    rel = rel[torch.argsort(distances)]
    rank = torch.arange(len(rel), device=distances.device) + 1
    prec = torch.cumsum(rel, 0) / rank
    ap = torch.sum(prec * rel) / torch.sum(rel)
    return ap


@torch.inference_mode()
def rank_of_first_correct(distances, ismatch):
    assert distances.ndim == 1 and ismatch.ndim == 1 and len(distances) == len(ismatch)
    rel = ismatch.type_as(distances)
    assert rel.sum() >= 1, "There should be at least 1 relevant item"
    rel = rel[torch.argsort(distances)]
    # argmax returns index of first occurrence
    r1 = (torch.argmax(rel) + 1).type_as(distances)
    return r1


@torch.inference_mode()
def rank_percentile(distances, ismatch, biased=False):
    # https://publications.hevs.ch/index.php/publications/show/125
    assert distances.ndim == 1 and ismatch.ndim == 1 and len(distances) == len(ismatch)
    rel = ismatch.type_as(distances)
    assert rel.sum() >= 1, "There should be at least 1 relevant item"
    rel = rel[torch.argsort(distances)]
    if biased:
        # Size of the clique affects the measure, that is, you do not get a
        # perfect 0 score if clique size>1
        normrank = torch.linspace(0, 1, len(rel), device=distances.device)
    else:
        # counting number of zeros preceding rels allows to get perfect 0 score
        normrank = torch.cumsum(1 - rel, 0) / torch.sum(1 - rel)
    rpc = torch.sum(rel * normrank) / torch.sum(rel)
    return 100 * rpc


###################################################################################################
