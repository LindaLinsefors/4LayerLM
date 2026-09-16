# CI co-activation per matrix — alive components, co-CI>0.9 clusters

Same heatmaps as [report_alive.md](report_alive.md) (Pearson r of per-token CI; see there for data and definitions), but with threshold-rule clusters (alternative to [report_alive_clustered.md](report_alive_clustered.md)): **(1)** two components with co-CI r > 0.9 are in the same cluster (pairs processed in descending r, chains allowed); **(2)** a component does not join a cluster if it has r ≤ 0.0 (or undefined r) with any existing member — two clusters only merge if every cross pair is > 0; such joins are skipped (count noted per matrix when > 0). No mean-CI-ratio constraint. Clusters are placed by descending mean member CI — singletons land where the plain mean-CI sort would put them — and members within a cluster are sorted by descending mean CI. **Black outlines** mark the multi-member clusters on the diagonal.

## pile_4l

### Layer 0

#### h.0.attn.q_proj — 111 components, 9 clusters with ≥ 2 members (largest 26)

![h.0.attn.q_proj](../hide/figures/pile_4l/alive_clustered09/h0_attn_q_proj.png)

#### h.0.attn.k_proj — 126 components, 14 clusters with ≥ 2 members (largest 16)

![h.0.attn.k_proj](../hide/figures/pile_4l/alive_clustered09/h0_attn_k_proj.png)

#### h.0.attn.v_proj — 514 components, 33 clusters with ≥ 2 members (largest 9)

![h.0.attn.v_proj](../hide/figures/pile_4l/alive_clustered09/h0_attn_v_proj.png)

#### h.0.attn.o_proj — 446 components, 39 clusters with ≥ 2 members (largest 12)

![h.0.attn.o_proj](../hide/figures/pile_4l/alive_clustered09/h0_attn_o_proj.png)

#### h.0.mlp.c_fc — 1380 components, 64 clusters with ≥ 2 members (largest 21)

![h.0.mlp.c_fc](../hide/figures/pile_4l/alive_clustered09/h0_mlp_c_fc.png)

#### h.0.mlp.down_proj — 1133 components, 18 clusters with ≥ 2 members (largest 29)

![h.0.mlp.down_proj](../hide/figures/pile_4l/alive_clustered09/h0_mlp_down_proj.png)

### Layer 1

#### h.1.attn.q_proj — 15 components, 2 clusters with ≥ 2 members (largest 3)

![h.1.attn.q_proj](../hide/figures/pile_4l/alive_clustered09/h1_attn_q_proj.png)

#### h.1.attn.k_proj — 48 components, 4 clusters with ≥ 2 members (largest 27)

![h.1.attn.k_proj](../hide/figures/pile_4l/alive_clustered09/h1_attn_k_proj.png)

#### h.1.attn.v_proj — 226 components, 9 clusters with ≥ 2 members (largest 68)

![h.1.attn.v_proj](../hide/figures/pile_4l/alive_clustered09/h1_attn_v_proj.png)

#### h.1.attn.o_proj — 97 components, 4 clusters with ≥ 2 members (largest 3)

![h.1.attn.o_proj](../hide/figures/pile_4l/alive_clustered09/h1_attn_o_proj.png)

#### h.1.mlp.c_fc — 251 components, 11 clusters with ≥ 2 members (largest 40)

![h.1.mlp.c_fc](../hide/figures/pile_4l/alive_clustered09/h1_mlp_c_fc.png)

#### h.1.mlp.down_proj — 211 components, 5 clusters with ≥ 2 members (largest 20)

![h.1.mlp.down_proj](../hide/figures/pile_4l/alive_clustered09/h1_mlp_down_proj.png)

### Layer 2

#### h.2.attn.q_proj — 92 components, 0 clusters with ≥ 2 members (largest 1)

![h.2.attn.q_proj](../hide/figures/pile_4l/alive_clustered09/h2_attn_q_proj.png)

#### h.2.attn.k_proj — 167 components, 5 clusters with ≥ 2 members (largest 29)

![h.2.attn.k_proj](../hide/figures/pile_4l/alive_clustered09/h2_attn_k_proj.png)

#### h.2.attn.v_proj — 508 components, 3 clusters with ≥ 2 members (largest 20)

![h.2.attn.v_proj](../hide/figures/pile_4l/alive_clustered09/h2_attn_v_proj.png)

#### h.2.attn.o_proj — 525 components, 2 clusters with ≥ 2 members (largest 2)

![h.2.attn.o_proj](../hide/figures/pile_4l/alive_clustered09/h2_attn_o_proj.png)

#### h.2.mlp.c_fc — 292 components, 4 clusters with ≥ 2 members (largest 5)

![h.2.mlp.c_fc](../hide/figures/pile_4l/alive_clustered09/h2_mlp_c_fc.png)

#### h.2.mlp.down_proj — 359 components, 4 clusters with ≥ 2 members (largest 7)

![h.2.mlp.down_proj](../hide/figures/pile_4l/alive_clustered09/h2_mlp_down_proj.png)

### Layer 3

#### h.3.attn.q_proj — 36 components, 2 clusters with ≥ 2 members (largest 2)

![h.3.attn.q_proj](../hide/figures/pile_4l/alive_clustered09/h3_attn_q_proj.png)

#### h.3.attn.k_proj — 60 components, 2 clusters with ≥ 2 members (largest 8)

![h.3.attn.k_proj](../hide/figures/pile_4l/alive_clustered09/h3_attn_k_proj.png)

#### h.3.attn.v_proj — 249 components, 3 clusters with ≥ 2 members (largest 6)

![h.3.attn.v_proj](../hide/figures/pile_4l/alive_clustered09/h3_attn_v_proj.png)

#### h.3.attn.o_proj — 246 components, 2 clusters with ≥ 2 members (largest 14)

![h.3.attn.o_proj](../hide/figures/pile_4l/alive_clustered09/h3_attn_o_proj.png)

#### h.3.mlp.c_fc — 1031 components, 3 clusters with ≥ 2 members (largest 3)

![h.3.mlp.c_fc](../hide/figures/pile_4l/alive_clustered09/h3_mlp_c_fc.png)

#### h.3.mlp.down_proj — 1850 components, 3 clusters with ≥ 2 members (largest 5)

![h.3.mlp.down_proj](../hide/figures/pile_4l/alive_clustered09/h3_mlp_down_proj.png)

## simple_2l

### Layer 0

#### h.0.attn.q_proj — 251 components, 0 clusters with ≥ 2 members (largest 1)

![h.0.attn.q_proj](../hide/figures/simple_2l/alive_clustered09/h0_attn_q_proj.png)

#### h.0.attn.k_proj — 248 components, 0 clusters with ≥ 2 members (largest 1)

![h.0.attn.k_proj](../hide/figures/simple_2l/alive_clustered09/h0_attn_k_proj.png)

#### h.0.attn.v_proj — 344 components, 0 clusters with ≥ 2 members (largest 1)

![h.0.attn.v_proj](../hide/figures/simple_2l/alive_clustered09/h0_attn_v_proj.png)

#### h.0.attn.o_proj — 432 components, 0 clusters with ≥ 2 members (largest 1)

![h.0.attn.o_proj](../hide/figures/simple_2l/alive_clustered09/h0_attn_o_proj.png)

#### h.0.mlp.c_fc — 1068 components, 1 clusters with ≥ 2 members (largest 2)

![h.0.mlp.c_fc](../hide/figures/simple_2l/alive_clustered09/h0_mlp_c_fc.png)

#### h.0.mlp.down_proj — 886 components, 0 clusters with ≥ 2 members (largest 1)

![h.0.mlp.down_proj](../hide/figures/simple_2l/alive_clustered09/h0_mlp_down_proj.png)

### Layer 1

#### h.1.attn.q_proj — 268 components, 1 clusters with ≥ 2 members (largest 2)

![h.1.attn.q_proj](../hide/figures/simple_2l/alive_clustered09/h1_attn_q_proj.png)

#### h.1.attn.k_proj — 256 components, 0 clusters with ≥ 2 members (largest 1)

![h.1.attn.k_proj](../hide/figures/simple_2l/alive_clustered09/h1_attn_k_proj.png)

#### h.1.attn.v_proj — 358 components, 0 clusters with ≥ 2 members (largest 1)

![h.1.attn.v_proj](../hide/figures/simple_2l/alive_clustered09/h1_attn_v_proj.png)

#### h.1.attn.o_proj — 442 components, 0 clusters with ≥ 2 members (largest 1)

![h.1.attn.o_proj](../hide/figures/simple_2l/alive_clustered09/h1_attn_o_proj.png)

#### h.1.mlp.c_fc — 1076 components, 0 clusters with ≥ 2 members (largest 1)

![h.1.mlp.c_fc](../hide/figures/simple_2l/alive_clustered09/h1_mlp_c_fc.png)

#### h.1.mlp.down_proj — 906 components, 0 clusters with ≥ 2 members (largest 1)

![h.1.mlp.down_proj](../hide/figures/simple_2l/alive_clustered09/h1_mlp_down_proj.png)
