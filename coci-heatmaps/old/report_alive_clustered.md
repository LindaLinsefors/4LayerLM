# CI co-activation per matrix — alive components, CI-clustered order

Same heatmaps as [report_alive.md](report_alive.md) (Pearson r of per-token CI; see there for data and definitions), but the component order trades off mean-CI sorting against CI-similarity clustering: constrained complete-linkage clustering on d = 1 − r, with pairs whose harvest mean CIs differ by **more than a factor 2** (or with undefined r) forbidden (d = ∞) — so no cluster ever mixes components of >2× different average CI. Flat clusters at complete-linkage d ≤ 0.7 (⇒ every within-cluster pair has r ≥ 0.3). Clusters are placed by descending mean member CI — singletons land where the plain mean-CI sort would put them — and members within a cluster are sorted by descending mean CI. **Black outlines** mark the multi-member clusters on the diagonal.

## pile_4l

### Layer 0

#### h.0.attn.q_proj — 111 components, 24 clusters with ≥ 2 members (largest 24)

![h.0.attn.q_proj](../hide/figures/pile_4l/alive_clustered/h0_attn_q_proj.png)

#### h.0.attn.k_proj — 126 components, 21 clusters with ≥ 2 members (largest 16)

![h.0.attn.k_proj](../hide/figures/pile_4l/alive_clustered/h0_attn_k_proj.png)

#### h.0.attn.v_proj — 514 components, 88 clusters with ≥ 2 members (largest 11)

![h.0.attn.v_proj](../hide/figures/pile_4l/alive_clustered/h0_attn_v_proj.png)

#### h.0.attn.o_proj — 446 components, 84 clusters with ≥ 2 members (largest 12)

![h.0.attn.o_proj](../hide/figures/pile_4l/alive_clustered/h0_attn_o_proj.png)

#### h.0.mlp.c_fc — 1380 components, 288 clusters with ≥ 2 members (largest 25)

![h.0.mlp.c_fc](../hide/figures/pile_4l/alive_clustered/h0_mlp_c_fc.png)

#### h.0.mlp.down_proj — 1133 components, 256 clusters with ≥ 2 members (largest 30)

![h.0.mlp.down_proj](../hide/figures/pile_4l/alive_clustered/h0_mlp_down_proj.png)

### Layer 1

#### h.1.attn.q_proj — 15 components, 3 clusters with ≥ 2 members (largest 3)

![h.1.attn.q_proj](../hide/figures/pile_4l/alive_clustered/h1_attn_q_proj.png)

#### h.1.attn.k_proj — 48 components, 5 clusters with ≥ 2 members (largest 27)

![h.1.attn.k_proj](../hide/figures/pile_4l/alive_clustered/h1_attn_k_proj.png)

#### h.1.attn.v_proj — 226 components, 25 clusters with ≥ 2 members (largest 62)

![h.1.attn.v_proj](../hide/figures/pile_4l/alive_clustered/h1_attn_v_proj.png)

#### h.1.attn.o_proj — 97 components, 12 clusters with ≥ 2 members (largest 4)

![h.1.attn.o_proj](../hide/figures/pile_4l/alive_clustered/h1_attn_o_proj.png)

#### h.1.mlp.c_fc — 251 components, 26 clusters with ≥ 2 members (largest 40)

![h.1.mlp.c_fc](../hide/figures/pile_4l/alive_clustered/h1_mlp_c_fc.png)

#### h.1.mlp.down_proj — 211 components, 29 clusters with ≥ 2 members (largest 29)

![h.1.mlp.down_proj](../hide/figures/pile_4l/alive_clustered/h1_mlp_down_proj.png)

### Layer 2

#### h.2.attn.q_proj — 92 components, 18 clusters with ≥ 2 members (largest 3)

![h.2.attn.q_proj](../hide/figures/pile_4l/alive_clustered/h2_attn_q_proj.png)

#### h.2.attn.k_proj — 167 components, 33 clusters with ≥ 2 members (largest 29)

![h.2.attn.k_proj](../hide/figures/pile_4l/alive_clustered/h2_attn_k_proj.png)

#### h.2.attn.v_proj — 508 components, 92 clusters with ≥ 2 members (largest 19)

![h.2.attn.v_proj](../hide/figures/pile_4l/alive_clustered/h2_attn_v_proj.png)

#### h.2.attn.o_proj — 525 components, 79 clusters with ≥ 2 members (largest 9)

![h.2.attn.o_proj](../hide/figures/pile_4l/alive_clustered/h2_attn_o_proj.png)

#### h.2.mlp.c_fc — 292 components, 46 clusters with ≥ 2 members (largest 5)

![h.2.mlp.c_fc](../hide/figures/pile_4l/alive_clustered/h2_mlp_c_fc.png)

#### h.2.mlp.down_proj — 359 components, 61 clusters with ≥ 2 members (largest 8)

![h.2.mlp.down_proj](../hide/figures/pile_4l/alive_clustered/h2_mlp_down_proj.png)

### Layer 3

#### h.3.attn.q_proj — 36 components, 3 clusters with ≥ 2 members (largest 2)

![h.3.attn.q_proj](../hide/figures/pile_4l/alive_clustered/h3_attn_q_proj.png)

#### h.3.attn.k_proj — 60 components, 9 clusters with ≥ 2 members (largest 13)

![h.3.attn.k_proj](../hide/figures/pile_4l/alive_clustered/h3_attn_k_proj.png)

#### h.3.attn.v_proj — 249 components, 40 clusters with ≥ 2 members (largest 6)

![h.3.attn.v_proj](../hide/figures/pile_4l/alive_clustered/h3_attn_v_proj.png)

#### h.3.attn.o_proj — 246 components, 31 clusters with ≥ 2 members (largest 17)

![h.3.attn.o_proj](../hide/figures/pile_4l/alive_clustered/h3_attn_o_proj.png)

#### h.3.mlp.c_fc — 1031 components, 224 clusters with ≥ 2 members (largest 24)

![h.3.mlp.c_fc](../hide/figures/pile_4l/alive_clustered/h3_mlp_c_fc.png)

#### h.3.mlp.down_proj — 1850 components, 398 clusters with ≥ 2 members (largest 52)

![h.3.mlp.down_proj](../hide/figures/pile_4l/alive_clustered/h3_mlp_down_proj.png)

## simple_2l

### Layer 0

#### h.0.attn.q_proj — 251 components, 14 clusters with ≥ 2 members (largest 4)

![h.0.attn.q_proj](../hide/figures/simple_2l/alive_clustered/h0_attn_q_proj.png)

#### h.0.attn.k_proj — 248 components, 19 clusters with ≥ 2 members (largest 3)

![h.0.attn.k_proj](../hide/figures/simple_2l/alive_clustered/h0_attn_k_proj.png)

#### h.0.attn.v_proj — 344 components, 19 clusters with ≥ 2 members (largest 4)

![h.0.attn.v_proj](../hide/figures/simple_2l/alive_clustered/h0_attn_v_proj.png)

#### h.0.attn.o_proj — 432 components, 22 clusters with ≥ 2 members (largest 6)

![h.0.attn.o_proj](../hide/figures/simple_2l/alive_clustered/h0_attn_o_proj.png)

#### h.0.mlp.c_fc — 1068 components, 54 clusters with ≥ 2 members (largest 7)

![h.0.mlp.c_fc](../hide/figures/simple_2l/alive_clustered/h0_mlp_c_fc.png)

#### h.0.mlp.down_proj — 886 components, 47 clusters with ≥ 2 members (largest 6)

![h.0.mlp.down_proj](../hide/figures/simple_2l/alive_clustered/h0_mlp_down_proj.png)

### Layer 1

#### h.1.attn.q_proj — 268 components, 17 clusters with ≥ 2 members (largest 5)

![h.1.attn.q_proj](../hide/figures/simple_2l/alive_clustered/h1_attn_q_proj.png)

#### h.1.attn.k_proj — 256 components, 12 clusters with ≥ 2 members (largest 5)

![h.1.attn.k_proj](../hide/figures/simple_2l/alive_clustered/h1_attn_k_proj.png)

#### h.1.attn.v_proj — 358 components, 16 clusters with ≥ 2 members (largest 4)

![h.1.attn.v_proj](../hide/figures/simple_2l/alive_clustered/h1_attn_v_proj.png)

#### h.1.attn.o_proj — 442 components, 26 clusters with ≥ 2 members (largest 5)

![h.1.attn.o_proj](../hide/figures/simple_2l/alive_clustered/h1_attn_o_proj.png)

#### h.1.mlp.c_fc — 1076 components, 82 clusters with ≥ 2 members (largest 7)

![h.1.mlp.c_fc](../hide/figures/simple_2l/alive_clustered/h1_mlp_c_fc.png)

#### h.1.mlp.down_proj — 906 components, 71 clusters with ≥ 2 members (largest 5)

![h.1.mlp.down_proj](../hide/figures/simple_2l/alive_clustered/h1_mlp_down_proj.png)
