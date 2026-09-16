# newB (`p-4d9a6a12`) — CI co-activation, all components, co-CI>0.9 clusters

Decomposition **newB = `p-4d9a6a12`** (800k-step JAX decomposition of the pile_4l target, ImportanceMinimalityLoss frequency coeff 6.6e-6; see CLAUDE.md "New 800k-step decompositions"). Heatmaps of **Pearson r of per-token causal importance** (clip(preact, 0, 1) = lower_leaky, continuous sampling) between every pair of a matrix's subcomponents, over the 4,000 cached Pile rows (2.05M tokens) — the same co-CI measure and threshold-rule clusters as the old decomposition's [report_all_clustered09.md](../old/report_all_clustered09.md): **(1)** two components with co-CI r > 0.9 are in the same cluster (pairs processed in descending r, chains allowed); **(2)** a component does not join a cluster if it has r ≤ 0.0 (or undefined r) with any existing member — such joins are skipped (count noted per matrix when > 0). Clusters are placed by descending mean member CI (singletons land where the plain mean-CI sort would put them), members within a cluster by descending mean CI. **Black outlines** mark multi-member clusters on the diagonal. **Gray** = zero CI variance in the sample (r undefined).

No harvest DB exists for the new runs, so mean CI is the **sample mean over the same 2.05M tokens** and alive is the proxy **sample mean CI > 1e-6**; the **dashed green line** sits at that alive count in plain mean-CI order (cluster order can put a few components on the wrong side of it).

### Layer 0

#### h.0.attn.q_proj — 768 components, 6 clusters with ≥ 2 members (largest 3)

![h.0.attn.q_proj](../hide/figures/newB/all_clustered09/h0_attn_q_proj.png)

#### h.0.attn.k_proj — 768 components, 17 clusters with ≥ 2 members (largest 8)

![h.0.attn.k_proj](../hide/figures/newB/all_clustered09/h0_attn_k_proj.png)

#### h.0.attn.v_proj — 768 components, 19 clusters with ≥ 2 members (largest 17)

![h.0.attn.v_proj](../hide/figures/newB/all_clustered09/h0_attn_v_proj.png)

#### h.0.attn.o_proj — 768 components, 18 clusters with ≥ 2 members (largest 23)

![h.0.attn.o_proj](../hide/figures/newB/all_clustered09/h0_attn_o_proj.png)

#### h.0.mlp.c_fc — 3072 components, 65 clusters with ≥ 2 members (largest 89)

![h.0.mlp.c_fc](../hide/figures/newB/all_clustered09/h0_mlp_c_fc.png)

#### h.0.mlp.down_proj — 3072 components, 54 clusters with ≥ 2 members (largest 33)

![h.0.mlp.down_proj](../hide/figures/newB/all_clustered09/h0_mlp_down_proj.png)

### Layer 1

#### h.1.attn.q_proj — 768 components, 0 clusters with ≥ 2 members (largest 1)

![h.1.attn.q_proj](../hide/figures/newB/all_clustered09/h1_attn_q_proj.png)

#### h.1.attn.k_proj — 768 components, 10 clusters with ≥ 2 members (largest 5)

![h.1.attn.k_proj](../hide/figures/newB/all_clustered09/h1_attn_k_proj.png)

#### h.1.attn.v_proj — 768 components, 21 clusters with ≥ 2 members (largest 24)

![h.1.attn.v_proj](../hide/figures/newB/all_clustered09/h1_attn_v_proj.png)

#### h.1.attn.o_proj — 768 components, 7 clusters with ≥ 2 members (largest 16)

![h.1.attn.o_proj](../hide/figures/newB/all_clustered09/h1_attn_o_proj.png)

#### h.1.mlp.c_fc — 3072 components, 19 clusters with ≥ 2 members (largest 80)

![h.1.mlp.c_fc](../hide/figures/newB/all_clustered09/h1_mlp_c_fc.png)

#### h.1.mlp.down_proj — 3072 components, 24 clusters with ≥ 2 members (largest 48)

![h.1.mlp.down_proj](../hide/figures/newB/all_clustered09/h1_mlp_down_proj.png)

### Layer 2

#### h.2.attn.q_proj — 768 components, 3 clusters with ≥ 2 members (largest 3)

![h.2.attn.q_proj](../hide/figures/newB/all_clustered09/h2_attn_q_proj.png)

#### h.2.attn.k_proj — 768 components, 8 clusters with ≥ 2 members (largest 66)

![h.2.attn.k_proj](../hide/figures/newB/all_clustered09/h2_attn_k_proj.png)

#### h.2.attn.v_proj — 768 components, 4 clusters with ≥ 2 members (largest 22)

![h.2.attn.v_proj](../hide/figures/newB/all_clustered09/h2_attn_v_proj.png)

#### h.2.attn.o_proj — 768 components, 1 clusters with ≥ 2 members (largest 2)

![h.2.attn.o_proj](../hide/figures/newB/all_clustered09/h2_attn_o_proj.png)

#### h.2.mlp.c_fc — 3072 components, 19 clusters with ≥ 2 members (largest 25)

![h.2.mlp.c_fc](../hide/figures/newB/all_clustered09/h2_mlp_c_fc.png)

#### h.2.mlp.down_proj — 3072 components, 19 clusters with ≥ 2 members (largest 21)

![h.2.mlp.down_proj](../hide/figures/newB/all_clustered09/h2_mlp_down_proj.png)

### Layer 3

#### h.3.attn.q_proj — 768 components, 7 clusters with ≥ 2 members (largest 2)

![h.3.attn.q_proj](../hide/figures/newB/all_clustered09/h3_attn_q_proj.png)

#### h.3.attn.k_proj — 768 components, 9 clusters with ≥ 2 members (largest 85)

![h.3.attn.k_proj](../hide/figures/newB/all_clustered09/h3_attn_k_proj.png)

#### h.3.attn.v_proj — 768 components, 5 clusters with ≥ 2 members (largest 46)

![h.3.attn.v_proj](../hide/figures/newB/all_clustered09/h3_attn_v_proj.png)

#### h.3.attn.o_proj — 768 components, 4 clusters with ≥ 2 members (largest 27)

![h.3.attn.o_proj](../hide/figures/newB/all_clustered09/h3_attn_o_proj.png)

#### h.3.mlp.c_fc — 3072 components, 19 clusters with ≥ 2 members (largest 14)

![h.3.mlp.c_fc](../hide/figures/newB/all_clustered09/h3_mlp_c_fc.png)

#### h.3.mlp.down_proj — 3072 components, 6 clusters with ≥ 2 members (largest 11)

![h.3.mlp.down_proj](../hide/figures/newB/all_clustered09/h3_mlp_down_proj.png)
