import sys
import torch

###################################################################################################


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
                queries_z[n : n + 1].float(),
                candidates_z.float(),
                qmask=queries_m[n : n + 1] if queries_m is not None else None,
                cmask=candidates_m,
                redux_strategy=redux_strategy,
            ).squeeze(0)
        else:
            dist = []
            for mstart in range(0, len(candidates_i), batch_size_candidates):
                mend = min(mstart + batch_size_candidates, len(candidates_i))
                ddd = model.distances(
                    queries_z[n : n + 1].float(),
                    candidates_z[mstart:mend].float(),
                    qmask=queries_m[n : n + 1] if queries_m is not None else None,
                    cmask=(
                        candidates_m[mstart:mend] if candidates_m is not None else None
                    ),
                    redux_strategy=redux_strategy,
                ).squeeze(0)
                dist.append(ddd)
            dist = torch.cat(dist, dim=-1)
        # Get ground truth
        match_clique = candidates_c == queries_c[n]
        # Remove query from candidates if present
        match_query = candidates_i == queries_i[n]
        dist = torch.where(match_query, torch.inf, dist)
        match_clique = torch.where(match_query, False, match_clique)
        # Compute AP and R1
        aps.append(average_precision(dist, match_clique))
        r1s.append(rank_of_first_correct(dist, match_clique))
        rpcs.append(rank_percentile(dist, match_clique))
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
