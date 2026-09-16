# C (`p-d60af588`) — CI co-activation, alive components, co-CI>0.9 clusters

Decomposition **C = `p-d60af588`** (seed 0; 100k-step JAX decomposition of the attention-sink/untied-head target `t-87f91319` = "sink seed 45", newA recipe: ImportanceMinimalityLoss frequency coeff 6.6e-5; see CLAUDE.md "Attention-sink models & decompositions"). Heatmaps of **Pearson r of per-token causal importance** (clip(preact, 0, 1) = lower_leaky, continuous sampling) between every pair of a matrix's subcomponents, over the 4,000 cached Pile rows (2.05M tokens) — the same co-CI measure and threshold-rule clusters as the old decomposition's [report_alive_clustered09.md](../old/report_alive_clustered09.md): **(1)** two components with co-CI r > 0.9 are in the same cluster (pairs processed in descending r, chains allowed); **(2)** a component does not join a cluster if it has r ≤ 0.0 (or undefined r) with any existing member — such joins are skipped (count noted per matrix when > 0). Clusters are placed by descending mean member CI (singletons land where the plain mean-CI sort would put them), members within a cluster by descending mean CI. **Black outlines** mark multi-member clusters on the diagonal. **Gray** = zero CI variance in the sample (r undefined).

No harvest DB exists for these runs, so mean CI is the **sample mean over the same 2.05M tokens** and alive is the proxy **sample mean CI > 1e-6**.

Each matrix shows **three heatmaps in the same cluster order**: co-CI r, then the **signed cosine similarity between the components' write vectors cos(U_a, U_b)** (U = the output (d_out) factor of the rank-one subcomponent V_c U_c^T) and **between their read-in vectors cos(V_a, V_b)** (V = the input (d_in) factor). A component's sign is gauge ((V_c, U_c) -> (-V_c, -U_c) is the same component), so each component's (U_c, V_c) is flipped jointly to make its **input activation V_c·x positive on the majority of tokens where it fires (CI > 0.1)** (ties by summed firing activation, never-firing components by the all-token activation sum; stats over the same 4,000 Pile rows). Same white-centered scale as co-CI r; random baseline E|cos| = sqrt(2/(pi d)) ~= 0.03 for d = 768, 0.014 for d = 3072.

### Layer 0

#### h.0.attn.q_proj — 410 components, 29 clusters with ≥ 2 members (largest 11)

![h.0.attn.q_proj](../hide/figures/C/alive_clustered09/h0_attn_q_proj.png)

![h.0.attn.q_proj cosU](../hide/figures/C/alive_clustered09/h0_attn_q_proj_cosU.png)

![h.0.attn.q_proj cosV](../hide/figures/C/alive_clustered09/h0_attn_q_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.attn.k_proj — 555 components, 41 clusters with ≥ 2 members (largest 8)

![h.0.attn.k_proj](../hide/figures/C/alive_clustered09/h0_attn_k_proj.png)

![h.0.attn.k_proj cosU](../hide/figures/C/alive_clustered09/h0_attn_k_proj_cosU.png)

![h.0.attn.k_proj cosV](../hide/figures/C/alive_clustered09/h0_attn_k_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.attn.v_proj — 666 components, 25 clusters with ≥ 2 members (largest 10)

![h.0.attn.v_proj](../hide/figures/C/alive_clustered09/h0_attn_v_proj.png)

![h.0.attn.v_proj cosU](../hide/figures/C/alive_clustered09/h0_attn_v_proj_cosU.png)

![h.0.attn.v_proj cosV](../hide/figures/C/alive_clustered09/h0_attn_v_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.attn.o_proj — 551 components, 36 clusters with ≥ 2 members (largest 8)

![h.0.attn.o_proj](../hide/figures/C/alive_clustered09/h0_attn_o_proj.png)

![h.0.attn.o_proj cosU](../hide/figures/C/alive_clustered09/h0_attn_o_proj_cosU.png)

![h.0.attn.o_proj cosV](../hide/figures/C/alive_clustered09/h0_attn_o_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.mlp.c_fc — 2064 components, 90 clusters with ≥ 2 members (largest 20)

![h.0.mlp.c_fc](../hide/figures/C/alive_clustered09/h0_mlp_c_fc.png)

![h.0.mlp.c_fc cosU](../hide/figures/C/alive_clustered09/h0_mlp_c_fc_cosU.png)

![h.0.mlp.c_fc cosV](../hide/figures/C/alive_clustered09/h0_mlp_c_fc_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.mlp.down_proj — 1838 components, 65 clusters with ≥ 2 members (largest 21)

![h.0.mlp.down_proj](../hide/figures/C/alive_clustered09/h0_mlp_down_proj.png)

![h.0.mlp.down_proj cosU](../hide/figures/C/alive_clustered09/h0_mlp_down_proj_cosU.png)

![h.0.mlp.down_proj cosV](../hide/figures/C/alive_clustered09/h0_mlp_down_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

### Layer 1

#### h.1.attn.q_proj — 83 components, 7 clusters with ≥ 2 members (largest 5)

![h.1.attn.q_proj](../hide/figures/C/alive_clustered09/h1_attn_q_proj.png)

![h.1.attn.q_proj cosU](../hide/figures/C/alive_clustered09/h1_attn_q_proj_cosU.png)

![h.1.attn.q_proj cosV](../hide/figures/C/alive_clustered09/h1_attn_q_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.attn.k_proj — 185 components, 14 clusters with ≥ 2 members (largest 17)

![h.1.attn.k_proj](../hide/figures/C/alive_clustered09/h1_attn_k_proj.png)

![h.1.attn.k_proj cosU](../hide/figures/C/alive_clustered09/h1_attn_k_proj_cosU.png)

![h.1.attn.k_proj cosV](../hide/figures/C/alive_clustered09/h1_attn_k_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.attn.v_proj — 401 components, 15 clusters with ≥ 2 members (largest 38)

![h.1.attn.v_proj](../hide/figures/C/alive_clustered09/h1_attn_v_proj.png)

![h.1.attn.v_proj cosU](../hide/figures/C/alive_clustered09/h1_attn_v_proj_cosU.png)

![h.1.attn.v_proj cosV](../hide/figures/C/alive_clustered09/h1_attn_v_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.attn.o_proj — 217 components, 6 clusters with ≥ 2 members (largest 14)

![h.1.attn.o_proj](../hide/figures/C/alive_clustered09/h1_attn_o_proj.png)

![h.1.attn.o_proj cosU](../hide/figures/C/alive_clustered09/h1_attn_o_proj_cosU.png)

![h.1.attn.o_proj cosV](../hide/figures/C/alive_clustered09/h1_attn_o_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.mlp.c_fc — 1049 components, 42 clusters with ≥ 2 members (largest 39)

![h.1.mlp.c_fc](../hide/figures/C/alive_clustered09/h1_mlp_c_fc.png)

![h.1.mlp.c_fc cosU](../hide/figures/C/alive_clustered09/h1_mlp_c_fc_cosU.png)

![h.1.mlp.c_fc cosV](../hide/figures/C/alive_clustered09/h1_mlp_c_fc_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.mlp.down_proj — 1034 components, 22 clusters with ≥ 2 members (largest 19)

![h.1.mlp.down_proj](../hide/figures/C/alive_clustered09/h1_mlp_down_proj.png)

![h.1.mlp.down_proj cosU](../hide/figures/C/alive_clustered09/h1_mlp_down_proj_cosU.png)

![h.1.mlp.down_proj cosV](../hide/figures/C/alive_clustered09/h1_mlp_down_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

### Layer 2

#### h.2.attn.q_proj — 144 components, 5 clusters with ≥ 2 members (largest 2)

![h.2.attn.q_proj](../hide/figures/C/alive_clustered09/h2_attn_q_proj.png)

![h.2.attn.q_proj cosU](../hide/figures/C/alive_clustered09/h2_attn_q_proj_cosU.png)

![h.2.attn.q_proj cosV](../hide/figures/C/alive_clustered09/h2_attn_q_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.attn.k_proj — 362 components, 16 clusters with ≥ 2 members (largest 9)

![h.2.attn.k_proj](../hide/figures/C/alive_clustered09/h2_attn_k_proj.png)

![h.2.attn.k_proj cosU](../hide/figures/C/alive_clustered09/h2_attn_k_proj_cosU.png)

![h.2.attn.k_proj cosV](../hide/figures/C/alive_clustered09/h2_attn_k_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.attn.v_proj — 700 components, 16 clusters with ≥ 2 members (largest 6)

![h.2.attn.v_proj](../hide/figures/C/alive_clustered09/h2_attn_v_proj.png)

![h.2.attn.v_proj cosU](../hide/figures/C/alive_clustered09/h2_attn_v_proj_cosU.png)

![h.2.attn.v_proj cosV](../hide/figures/C/alive_clustered09/h2_attn_v_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.attn.o_proj — 490 components, 7 clusters with ≥ 2 members (largest 3)

![h.2.attn.o_proj](../hide/figures/C/alive_clustered09/h2_attn_o_proj.png)

![h.2.attn.o_proj cosU](../hide/figures/C/alive_clustered09/h2_attn_o_proj_cosU.png)

![h.2.attn.o_proj cosV](../hide/figures/C/alive_clustered09/h2_attn_o_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.mlp.c_fc — 1103 components, 32 clusters with ≥ 2 members (largest 19)

![h.2.mlp.c_fc](../hide/figures/C/alive_clustered09/h2_mlp_c_fc.png)

![h.2.mlp.c_fc cosU](../hide/figures/C/alive_clustered09/h2_mlp_c_fc_cosU.png)

![h.2.mlp.c_fc cosV](../hide/figures/C/alive_clustered09/h2_mlp_c_fc_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.mlp.down_proj — 1452 components, 31 clusters with ≥ 2 members (largest 18)

![h.2.mlp.down_proj](../hide/figures/C/alive_clustered09/h2_mlp_down_proj.png)

![h.2.mlp.down_proj cosU](../hide/figures/C/alive_clustered09/h2_mlp_down_proj_cosU.png)

![h.2.mlp.down_proj cosV](../hide/figures/C/alive_clustered09/h2_mlp_down_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

### Layer 3

#### h.3.attn.q_proj — 188 components, 4 clusters with ≥ 2 members (largest 4)

![h.3.attn.q_proj](../hide/figures/C/alive_clustered09/h3_attn_q_proj.png)

![h.3.attn.q_proj cosU](../hide/figures/C/alive_clustered09/h3_attn_q_proj_cosU.png)

![h.3.attn.q_proj cosV](../hide/figures/C/alive_clustered09/h3_attn_q_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.attn.k_proj — 450 components, 20 clusters with ≥ 2 members (largest 10)

![h.3.attn.k_proj](../hide/figures/C/alive_clustered09/h3_attn_k_proj.png)

![h.3.attn.k_proj cosU](../hide/figures/C/alive_clustered09/h3_attn_k_proj_cosU.png)

![h.3.attn.k_proj cosV](../hide/figures/C/alive_clustered09/h3_attn_k_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.attn.v_proj — 745 components, 5 clusters with ≥ 2 members (largest 5)

![h.3.attn.v_proj](../hide/figures/C/alive_clustered09/h3_attn_v_proj.png)

![h.3.attn.v_proj cosU](../hide/figures/C/alive_clustered09/h3_attn_v_proj_cosU.png)

![h.3.attn.v_proj cosV](../hide/figures/C/alive_clustered09/h3_attn_v_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.attn.o_proj — 623 components, 8 clusters with ≥ 2 members (largest 3)

![h.3.attn.o_proj](../hide/figures/C/alive_clustered09/h3_attn_o_proj.png)

![h.3.attn.o_proj cosU](../hide/figures/C/alive_clustered09/h3_attn_o_proj_cosU.png)

![h.3.attn.o_proj cosV](../hide/figures/C/alive_clustered09/h3_attn_o_proj_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.mlp.c_fc — 1507 components, 16 clusters with ≥ 2 members (largest 4)

![h.3.mlp.c_fc](../hide/figures/C/alive_clustered09/h3_mlp_c_fc.png)

![h.3.mlp.c_fc cosU](../hide/figures/C/alive_clustered09/h3_mlp_c_fc_cosU.png)

![h.3.mlp.c_fc cosV](../hide/figures/C/alive_clustered09/h3_mlp_c_fc_cosV.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.mlp.down_proj — 2586 components, 18 clusters with ≥ 2 members (largest 3)

![h.3.mlp.down_proj](../hide/figures/C/alive_clustered09/h3_mlp_down_proj.png)

![h.3.mlp.down_proj cosU](../hide/figures/C/alive_clustered09/h3_mlp_down_proj_cosU.png)

![h.3.mlp.down_proj cosV](../hide/figures/C/alive_clustered09/h3_mlp_down_proj_cosV.png)
