# CI co-activation per matrix — all components, CI-clustered order

Same heatmaps as [report_all.md](report_all.md) (Pearson r of per-token CI; see there for data and definitions), but the component order trades off mean-CI sorting against CI-similarity clustering: constrained complete-linkage clustering on d = 1 − r, with pairs whose harvest mean CIs differ by **more than a factor 2** (or with undefined r) forbidden (d = ∞) — so no cluster ever mixes components of >2× different average CI. Flat clusters at complete-linkage d ≤ 0.7 (⇒ every within-cluster pair has r ≥ 0.3). Clusters are placed by descending mean member CI — singletons land where the plain mean-CI sort would put them — and members within a cluster are sorted by descending mean CI. **Black outlines** mark the multi-member clusters on the diagonal.

## pile_4l

### Layer 0

#### h.0.attn.q_proj — 512 components, 25 clusters with ≥ 2 members (largest 24)

![h.0.attn.q_proj](../hide/figures/pile_4l/all_clustered/h0_attn_q_proj.png)

#### h.0.attn.k_proj — 512 components, 22 clusters with ≥ 2 members (largest 16)

![h.0.attn.k_proj](../hide/figures/pile_4l/all_clustered/h0_attn_k_proj.png)

#### h.0.attn.v_proj — 1024 components, 88 clusters with ≥ 2 members (largest 11)

![h.0.attn.v_proj](../hide/figures/pile_4l/all_clustered/h0_attn_v_proj.png)

#### h.0.attn.o_proj — 1024 components, 85 clusters with ≥ 2 members (largest 12)

![h.0.attn.o_proj](../hide/figures/pile_4l/all_clustered/h0_attn_o_proj.png)

#### h.0.mlp.c_fc — 3072 components, 294 clusters with ≥ 2 members (largest 25)

![h.0.mlp.c_fc](../hide/figures/pile_4l/all_clustered/h0_mlp_c_fc.png)

#### h.0.mlp.down_proj — 3584 components, 262 clusters with ≥ 2 members (largest 30)

![h.0.mlp.down_proj](../hide/figures/pile_4l/all_clustered/h0_mlp_down_proj.png)

### Layer 1

#### h.1.attn.q_proj — 512 components, 3 clusters with ≥ 2 members (largest 3)

![h.1.attn.q_proj](../hide/figures/pile_4l/all_clustered/h1_attn_q_proj.png)

#### h.1.attn.k_proj — 512 components, 6 clusters with ≥ 2 members (largest 27)

![h.1.attn.k_proj](../hide/figures/pile_4l/all_clustered/h1_attn_k_proj.png)

#### h.1.attn.v_proj — 1024 components, 31 clusters with ≥ 2 members (largest 62)

![h.1.attn.v_proj](../hide/figures/pile_4l/all_clustered/h1_attn_v_proj.png)

#### h.1.attn.o_proj — 1024 components, 19 clusters with ≥ 2 members (largest 7)

![h.1.attn.o_proj](../hide/figures/pile_4l/all_clustered/h1_attn_o_proj.png)

#### h.1.mlp.c_fc — 3072 components, 40 clusters with ≥ 2 members (largest 40)

![h.1.mlp.c_fc](../hide/figures/pile_4l/all_clustered/h1_mlp_c_fc.png)

#### h.1.mlp.down_proj — 3584 components, 40 clusters with ≥ 2 members (largest 29)

![h.1.mlp.down_proj](../hide/figures/pile_4l/all_clustered/h1_mlp_down_proj.png)

### Layer 2

#### h.2.attn.q_proj — 512 components, 18 clusters with ≥ 2 members (largest 3)

![h.2.attn.q_proj](../hide/figures/pile_4l/all_clustered/h2_attn_q_proj.png)

#### h.2.attn.k_proj — 512 components, 33 clusters with ≥ 2 members (largest 29)

![h.2.attn.k_proj](../hide/figures/pile_4l/all_clustered/h2_attn_k_proj.png)

#### h.2.attn.v_proj — 1024 components, 99 clusters with ≥ 2 members (largest 19)

![h.2.attn.v_proj](../hide/figures/pile_4l/all_clustered/h2_attn_v_proj.png)

#### h.2.attn.o_proj — 1024 components, 86 clusters with ≥ 2 members (largest 9)

![h.2.attn.o_proj](../hide/figures/pile_4l/all_clustered/h2_attn_o_proj.png)

#### h.2.mlp.c_fc — 3072 components, 53 clusters with ≥ 2 members (largest 5)

![h.2.mlp.c_fc](../hide/figures/pile_4l/all_clustered/h2_mlp_c_fc.png)

#### h.2.mlp.down_proj — 3584 components, 72 clusters with ≥ 2 members (largest 8)

![h.2.mlp.down_proj](../hide/figures/pile_4l/all_clustered/h2_mlp_down_proj.png)

### Layer 3

#### h.3.attn.q_proj — 512 components, 3 clusters with ≥ 2 members (largest 2)

![h.3.attn.q_proj](../hide/figures/pile_4l/all_clustered/h3_attn_q_proj.png)

#### h.3.attn.k_proj — 512 components, 11 clusters with ≥ 2 members (largest 13)

![h.3.attn.k_proj](../hide/figures/pile_4l/all_clustered/h3_attn_k_proj.png)

#### h.3.attn.v_proj — 1024 components, 50 clusters with ≥ 2 members (largest 15)

![h.3.attn.v_proj](../hide/figures/pile_4l/all_clustered/h3_attn_v_proj.png)

#### h.3.attn.o_proj — 1024 components, 31 clusters with ≥ 2 members (largest 17)

![h.3.attn.o_proj](../hide/figures/pile_4l/all_clustered/h3_attn_o_proj.png)

#### h.3.mlp.c_fc — 3072 components, 227 clusters with ≥ 2 members (largest 24)

![h.3.mlp.c_fc](../hide/figures/pile_4l/all_clustered/h3_mlp_c_fc.png)

#### h.3.mlp.down_proj — 3584 components, 399 clusters with ≥ 2 members (largest 52)

![h.3.mlp.down_proj](../hide/figures/pile_4l/all_clustered/h3_mlp_down_proj.png)

## simple_2l

### Layer 0

#### h.0.attn.q_proj — 288 components, 14 clusters with ≥ 2 members (largest 4)

![h.0.attn.q_proj](../hide/figures/simple_2l/all_clustered/h0_attn_q_proj.png)

#### h.0.attn.k_proj — 288 components, 19 clusters with ≥ 2 members (largest 3)

![h.0.attn.k_proj](../hide/figures/simple_2l/all_clustered/h0_attn_k_proj.png)

#### h.0.attn.v_proj — 384 components, 19 clusters with ≥ 2 members (largest 4)

![h.0.attn.v_proj](../hide/figures/simple_2l/all_clustered/h0_attn_v_proj.png)

#### h.0.attn.o_proj — 480 components, 22 clusters with ≥ 2 members (largest 6)

![h.0.attn.o_proj](../hide/figures/simple_2l/all_clustered/h0_attn_o_proj.png)

#### h.0.mlp.c_fc — 1152 components, 54 clusters with ≥ 2 members (largest 7)

![h.0.mlp.c_fc](../hide/figures/simple_2l/all_clustered/h0_mlp_c_fc.png)

#### h.0.mlp.down_proj — 960 components, 47 clusters with ≥ 2 members (largest 6)

![h.0.mlp.down_proj](../hide/figures/simple_2l/all_clustered/h0_mlp_down_proj.png)

### Layer 1

#### h.1.attn.q_proj — 288 components, 17 clusters with ≥ 2 members (largest 5)

![h.1.attn.q_proj](../hide/figures/simple_2l/all_clustered/h1_attn_q_proj.png)

#### h.1.attn.k_proj — 288 components, 12 clusters with ≥ 2 members (largest 5)

![h.1.attn.k_proj](../hide/figures/simple_2l/all_clustered/h1_attn_k_proj.png)

#### h.1.attn.v_proj — 384 components, 16 clusters with ≥ 2 members (largest 4)

![h.1.attn.v_proj](../hide/figures/simple_2l/all_clustered/h1_attn_v_proj.png)

#### h.1.attn.o_proj — 480 components, 26 clusters with ≥ 2 members (largest 5)

![h.1.attn.o_proj](../hide/figures/simple_2l/all_clustered/h1_attn_o_proj.png)

#### h.1.mlp.c_fc — 1152 components, 82 clusters with ≥ 2 members (largest 7)

![h.1.mlp.c_fc](../hide/figures/simple_2l/all_clustered/h1_mlp_c_fc.png)

#### h.1.mlp.down_proj — 960 components, 71 clusters with ≥ 2 members (largest 5)

![h.1.mlp.down_proj](../hide/figures/simple_2l/all_clustered/h1_mlp_down_proj.png)
