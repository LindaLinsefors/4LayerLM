# newB (`p-4d9a6a12`) — CI co-activation, alive components, co-CI>0.9 clusters

Decomposition **newB = `p-4d9a6a12`** (800k-step JAX decomposition of the pile_4l target, ImportanceMinimalityLoss frequency coeff 6.6e-6; see CLAUDE.md "New 800k-step decompositions"). Heatmaps of **Pearson r of per-token causal importance** (clip(preact, 0, 1) = lower_leaky, continuous sampling) between every pair of a matrix's subcomponents, over the 4,000 cached Pile rows (2.05M tokens) — the same co-CI measure and threshold-rule clusters as the old decomposition's [report_alive_clustered09.md](../old/report_alive_clustered09.md): **(1)** two components with co-CI r > 0.9 are in the same cluster (pairs processed in descending r, chains allowed); **(2)** a component does not join a cluster if it has r ≤ 0.0 (or undefined r) with any existing member — such joins are skipped (count noted per matrix when > 0). Clusters are placed by descending mean member CI (singletons land where the plain mean-CI sort would put them), members within a cluster by descending mean CI. **Black outlines** mark multi-member clusters on the diagonal. **Gray** = zero CI variance in the sample (r undefined).

No harvest DB exists for the new runs, so mean CI is the **sample mean over the same 2.05M tokens** and alive is the proxy **sample mean CI > 1e-6**.

Each matrix shows **three heatmaps in the same cluster order**: co-CI r, then the **absolute cosine similarity between the components' write vectors |cos(U_a, U_b)|** (U = the output (d_out) factor of the rank-one subcomponent V_c U_c^T) and **between their read-in vectors |cos(V_a, V_b)|** (V = the input (d_in) factor). Absolute value because a component's sign is gauge ((V_c, U_c) -> (-V_c, -U_c) is the same component); Reds, 0 -> 1; random baseline E|cos| = sqrt(2/(pi d)) ~= 0.03 for d = 768, 0.014 for d = 3072.

### Layer 0

#### h.0.attn.q_proj — 271 components, 6 clusters with ≥ 2 members (largest 3)

![h.0.attn.q_proj](../hide/figures/newB/alive_clustered09/h0_attn_q_proj.png)

![h.0.attn.q_proj cosU](../hide/figures/newB/alive_clustered09/h0_attn_q_proj_cosU.png)

![h.0.attn.q_proj cosV](../hide/figures/newB/alive_clustered09/h0_attn_q_proj_cosV.png)

#### h.0.attn.k_proj — 338 components, 14 clusters with ≥ 2 members (largest 8)

![h.0.attn.k_proj](../hide/figures/newB/alive_clustered09/h0_attn_k_proj.png)

![h.0.attn.k_proj cosU](../hide/figures/newB/alive_clustered09/h0_attn_k_proj_cosU.png)

![h.0.attn.k_proj cosV](../hide/figures/newB/alive_clustered09/h0_attn_k_proj_cosV.png)

#### h.0.attn.v_proj — 530 components, 18 clusters with ≥ 2 members (largest 17)

![h.0.attn.v_proj](../hide/figures/newB/alive_clustered09/h0_attn_v_proj.png)

![h.0.attn.v_proj cosU](../hide/figures/newB/alive_clustered09/h0_attn_v_proj_cosU.png)

![h.0.attn.v_proj cosV](../hide/figures/newB/alive_clustered09/h0_attn_v_proj_cosV.png)

#### h.0.attn.o_proj — 540 components, 18 clusters with ≥ 2 members (largest 23)

![h.0.attn.o_proj](../hide/figures/newB/alive_clustered09/h0_attn_o_proj.png)

![h.0.attn.o_proj cosU](../hide/figures/newB/alive_clustered09/h0_attn_o_proj_cosU.png)

![h.0.attn.o_proj cosV](../hide/figures/newB/alive_clustered09/h0_attn_o_proj_cosV.png)

#### h.0.mlp.c_fc — 2075 components, 60 clusters with ≥ 2 members (largest 40)

![h.0.mlp.c_fc](../hide/figures/newB/alive_clustered09/h0_mlp_c_fc.png)

![h.0.mlp.c_fc cosU](../hide/figures/newB/alive_clustered09/h0_mlp_c_fc_cosU.png)

![h.0.mlp.c_fc cosV](../hide/figures/newB/alive_clustered09/h0_mlp_c_fc_cosV.png)

#### h.0.mlp.down_proj — 2394 components, 51 clusters with ≥ 2 members (largest 13)

![h.0.mlp.down_proj](../hide/figures/newB/alive_clustered09/h0_mlp_down_proj.png)

![h.0.mlp.down_proj cosU](../hide/figures/newB/alive_clustered09/h0_mlp_down_proj_cosU.png)

![h.0.mlp.down_proj cosV](../hide/figures/newB/alive_clustered09/h0_mlp_down_proj_cosV.png)

### Layer 1

#### h.1.attn.q_proj — 74 components, 0 clusters with ≥ 2 members (largest 1)

![h.1.attn.q_proj](../hide/figures/newB/alive_clustered09/h1_attn_q_proj.png)

![h.1.attn.q_proj cosU](../hide/figures/newB/alive_clustered09/h1_attn_q_proj_cosU.png)

![h.1.attn.q_proj cosV](../hide/figures/newB/alive_clustered09/h1_attn_q_proj_cosV.png)

#### h.1.attn.k_proj — 129 components, 9 clusters with ≥ 2 members (largest 5)

![h.1.attn.k_proj](../hide/figures/newB/alive_clustered09/h1_attn_k_proj.png)

![h.1.attn.k_proj cosU](../hide/figures/newB/alive_clustered09/h1_attn_k_proj_cosU.png)

![h.1.attn.k_proj cosV](../hide/figures/newB/alive_clustered09/h1_attn_k_proj_cosV.png)

#### h.1.attn.v_proj — 453 components, 19 clusters with ≥ 2 members (largest 24)

![h.1.attn.v_proj](../hide/figures/newB/alive_clustered09/h1_attn_v_proj.png)

![h.1.attn.v_proj cosU](../hide/figures/newB/alive_clustered09/h1_attn_v_proj_cosU.png)

![h.1.attn.v_proj cosV](../hide/figures/newB/alive_clustered09/h1_attn_v_proj_cosV.png)

#### h.1.attn.o_proj — 374 components, 4 clusters with ≥ 2 members (largest 16)

![h.1.attn.o_proj](../hide/figures/newB/alive_clustered09/h1_attn_o_proj.png)

![h.1.attn.o_proj cosU](../hide/figures/newB/alive_clustered09/h1_attn_o_proj_cosU.png)

![h.1.attn.o_proj cosV](../hide/figures/newB/alive_clustered09/h1_attn_o_proj_cosV.png)

#### h.1.mlp.c_fc — 1290 components, 16 clusters with ≥ 2 members (largest 29)

![h.1.mlp.c_fc](../hide/figures/newB/alive_clustered09/h1_mlp_c_fc.png)

![h.1.mlp.c_fc cosU](../hide/figures/newB/alive_clustered09/h1_mlp_c_fc_cosU.png)

![h.1.mlp.c_fc cosV](../hide/figures/newB/alive_clustered09/h1_mlp_c_fc_cosV.png)

#### h.1.mlp.down_proj — 1742 components, 22 clusters with ≥ 2 members (largest 7)

![h.1.mlp.down_proj](../hide/figures/newB/alive_clustered09/h1_mlp_down_proj.png)

![h.1.mlp.down_proj cosU](../hide/figures/newB/alive_clustered09/h1_mlp_down_proj_cosU.png)

![h.1.mlp.down_proj cosV](../hide/figures/newB/alive_clustered09/h1_mlp_down_proj_cosV.png)

### Layer 2

#### h.2.attn.q_proj — 310 components, 3 clusters with ≥ 2 members (largest 3)

![h.2.attn.q_proj](../hide/figures/newB/alive_clustered09/h2_attn_q_proj.png)

![h.2.attn.q_proj cosU](../hide/figures/newB/alive_clustered09/h2_attn_q_proj_cosU.png)

![h.2.attn.q_proj cosV](../hide/figures/newB/alive_clustered09/h2_attn_q_proj_cosV.png)

#### h.2.attn.k_proj — 386 components, 6 clusters with ≥ 2 members (largest 66)

![h.2.attn.k_proj](../hide/figures/newB/alive_clustered09/h2_attn_k_proj.png)

![h.2.attn.k_proj cosU](../hide/figures/newB/alive_clustered09/h2_attn_k_proj_cosU.png)

![h.2.attn.k_proj cosV](../hide/figures/newB/alive_clustered09/h2_attn_k_proj_cosV.png)

#### h.2.attn.v_proj — 634 components, 4 clusters with ≥ 2 members (largest 22)

![h.2.attn.v_proj](../hide/figures/newB/alive_clustered09/h2_attn_v_proj.png)

![h.2.attn.v_proj cosU](../hide/figures/newB/alive_clustered09/h2_attn_v_proj_cosU.png)

![h.2.attn.v_proj cosV](../hide/figures/newB/alive_clustered09/h2_attn_v_proj_cosV.png)

#### h.2.attn.o_proj — 627 components, 1 clusters with ≥ 2 members (largest 2)

![h.2.attn.o_proj](../hide/figures/newB/alive_clustered09/h2_attn_o_proj.png)

![h.2.attn.o_proj cosU](../hide/figures/newB/alive_clustered09/h2_attn_o_proj_cosU.png)

![h.2.attn.o_proj cosV](../hide/figures/newB/alive_clustered09/h2_attn_o_proj_cosV.png)

#### h.2.mlp.c_fc — 1752 components, 16 clusters with ≥ 2 members (largest 25)

![h.2.mlp.c_fc](../hide/figures/newB/alive_clustered09/h2_mlp_c_fc.png)

![h.2.mlp.c_fc cosU](../hide/figures/newB/alive_clustered09/h2_mlp_c_fc_cosU.png)

![h.2.mlp.c_fc cosV](../hide/figures/newB/alive_clustered09/h2_mlp_c_fc_cosV.png)

#### h.2.mlp.down_proj — 2302 components, 16 clusters with ≥ 2 members (largest 21)

![h.2.mlp.down_proj](../hide/figures/newB/alive_clustered09/h2_mlp_down_proj.png)

![h.2.mlp.down_proj cosU](../hide/figures/newB/alive_clustered09/h2_mlp_down_proj_cosU.png)

![h.2.mlp.down_proj cosV](../hide/figures/newB/alive_clustered09/h2_mlp_down_proj_cosV.png)

### Layer 3

#### h.3.attn.q_proj — 255 components, 7 clusters with ≥ 2 members (largest 2)

![h.3.attn.q_proj](../hide/figures/newB/alive_clustered09/h3_attn_q_proj.png)

![h.3.attn.q_proj cosU](../hide/figures/newB/alive_clustered09/h3_attn_q_proj_cosU.png)

![h.3.attn.q_proj cosV](../hide/figures/newB/alive_clustered09/h3_attn_q_proj_cosV.png)

#### h.3.attn.k_proj — 329 components, 7 clusters with ≥ 2 members (largest 85)

![h.3.attn.k_proj](../hide/figures/newB/alive_clustered09/h3_attn_k_proj.png)

![h.3.attn.k_proj cosU](../hide/figures/newB/alive_clustered09/h3_attn_k_proj_cosU.png)

![h.3.attn.k_proj cosV](../hide/figures/newB/alive_clustered09/h3_attn_k_proj_cosV.png)

#### h.3.attn.v_proj — 608 components, 2 clusters with ≥ 2 members (largest 46)

![h.3.attn.v_proj](../hide/figures/newB/alive_clustered09/h3_attn_v_proj.png)

![h.3.attn.v_proj cosU](../hide/figures/newB/alive_clustered09/h3_attn_v_proj_cosU.png)

![h.3.attn.v_proj cosV](../hide/figures/newB/alive_clustered09/h3_attn_v_proj_cosV.png)

#### h.3.attn.o_proj — 619 components, 4 clusters with ≥ 2 members (largest 27)

![h.3.attn.o_proj](../hide/figures/newB/alive_clustered09/h3_attn_o_proj.png)

![h.3.attn.o_proj cosU](../hide/figures/newB/alive_clustered09/h3_attn_o_proj_cosU.png)

![h.3.attn.o_proj cosV](../hide/figures/newB/alive_clustered09/h3_attn_o_proj_cosV.png)

#### h.3.mlp.c_fc — 2226 components, 15 clusters with ≥ 2 members (largest 5)

![h.3.mlp.c_fc](../hide/figures/newB/alive_clustered09/h3_mlp_c_fc.png)

![h.3.mlp.c_fc cosU](../hide/figures/newB/alive_clustered09/h3_mlp_c_fc_cosU.png)

![h.3.mlp.c_fc cosV](../hide/figures/newB/alive_clustered09/h3_mlp_c_fc_cosV.png)

#### h.3.mlp.down_proj — 2744 components, 3 clusters with ≥ 2 members (largest 6)

![h.3.mlp.down_proj](../hide/figures/newB/alive_clustered09/h3_mlp_down_proj.png)

![h.3.mlp.down_proj cosU](../hide/figures/newB/alive_clustered09/h3_mlp_down_proj_cosU.png)

![h.3.mlp.down_proj cosV](../hide/figures/newB/alive_clustered09/h3_mlp_down_proj_cosV.png)
