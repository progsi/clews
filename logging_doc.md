# Train
- train/l_main  Main training loss
- hpar/lr       Learning rate
- l_cent	    Alignment loss term	Encourages same-label (positive) examples to be close
- l_cont	    Uniformity / contrastive repulsion loss	Encourages different-label samples to spread apart
- v_dpos	    Mean positive-pair distance	Average distance between positive pairs (same source except index)
- v_dneg	    Mean negative-pair distance	Average distance between negative pairs (different labels)
- v_zmax	    Max absolute embedding value (Detects exploding embeddings)
- v_zmean	    Mean of embedding values (Detects drift or bias)
- v_zstd	    Standard deviation of embeddings (Detects representation collapse)
# Valid
- valid/l_main	Main validation loss
- valid/m_MAP	Mean Average Precision from kNN evaluation
- valid/m_MR1	Mean Recall@1 (1-nearest-neighbor accuracy)
- valid/m_ARP	Average Rank Percentile
- valid/m_COMP	Composite metric (rpcs * (1 - aps))**0.5