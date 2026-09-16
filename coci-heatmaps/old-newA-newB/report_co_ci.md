# Cross-decomposition CI co-activation: old vs newA vs newB

For each matrix of the pile_4l target, heatmaps of the **Pearson r between the per-token causal importance of a component of one decomposition and a component of another** — pairs (old, newA), (old, newB), (newA, newB); the first-named decomposition is the y axis. Same CI measure (lower_leaky / clip(preact, 0, 1), continuous sampling) and the same 4,000 cached Pile rows (2.05M tokens) as the per-decomposition reports. **old** = `s-55ea3f9b` (the paper's decomposition), **newA** = `p-8383f5e5`, **newB** = `p-4d9a6a12` (see CLAUDE.md "New 800k-step decompositions").

Alive components only (old: harvest mean CI > 1e-6; newA/newB: sample mean CI > 1e-6). **Ordering, chosen to make matches diagonal:** the y axis is that decomposition's alive components by descending mean CI; each x component is placed at the position of its best-matching y component (max r, when that r ≥ 0.3), ties broken by its own mean CI — so shared mechanisms line up along a diagonal band (with vertical stripes where several x components match one y component). x components with no match ≥ 0.3 follow at the right, by descending mean CI. **Gray** = zero CI variance in the sample (r undefined).

### Layer 0

#### h.0.attn.q_proj

![h.0.attn.q_proj old_newA](../hide/figures/cross/old_newA/h0_attn_q_proj.png)

![h.0.attn.q_proj old_newB](../hide/figures/cross/old_newB/h0_attn_q_proj.png)

![h.0.attn.q_proj newA_newB](../hide/figures/cross/newA_newB/h0_attn_q_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.attn.k_proj

![h.0.attn.k_proj old_newA](../hide/figures/cross/old_newA/h0_attn_k_proj.png)

![h.0.attn.k_proj old_newB](../hide/figures/cross/old_newB/h0_attn_k_proj.png)

![h.0.attn.k_proj newA_newB](../hide/figures/cross/newA_newB/h0_attn_k_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.attn.v_proj

![h.0.attn.v_proj old_newA](../hide/figures/cross/old_newA/h0_attn_v_proj.png)

![h.0.attn.v_proj old_newB](../hide/figures/cross/old_newB/h0_attn_v_proj.png)

![h.0.attn.v_proj newA_newB](../hide/figures/cross/newA_newB/h0_attn_v_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.attn.o_proj

![h.0.attn.o_proj old_newA](../hide/figures/cross/old_newA/h0_attn_o_proj.png)

![h.0.attn.o_proj old_newB](../hide/figures/cross/old_newB/h0_attn_o_proj.png)

![h.0.attn.o_proj newA_newB](../hide/figures/cross/newA_newB/h0_attn_o_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

#### h.0.mlp.c_fc

![h.0.mlp.c_fc old_newA](../hide/figures/cross/old_newA/h0_mlp_c_fc.png)

![h.0.mlp.c_fc old_newB](../hide/figures/cross/old_newB/h0_mlp_c_fc.png)

![h.0.mlp.c_fc newA_newB](../hide/figures/cross/newA_newB/h0_mlp_c_fc.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.mlp.down_proj

![h.0.mlp.down_proj old_newA](../hide/figures/cross/old_newA/h0_mlp_down_proj.png)

![h.0.mlp.down_proj old_newB](../hide/figures/cross/old_newB/h0_mlp_down_proj.png)

![h.0.mlp.down_proj newA_newB](../hide/figures/cross/newA_newB/h0_mlp_down_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

### Layer 1

#### h.1.attn.q_proj

![h.1.attn.q_proj old_newA](../hide/figures/cross/old_newA/h1_attn_q_proj.png)

![h.1.attn.q_proj old_newB](../hide/figures/cross/old_newB/h1_attn_q_proj.png)

![h.1.attn.q_proj newA_newB](../hide/figures/cross/newA_newB/h1_attn_q_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.attn.k_proj

![h.1.attn.k_proj old_newA](../hide/figures/cross/old_newA/h1_attn_k_proj.png)

![h.1.attn.k_proj old_newB](../hide/figures/cross/old_newB/h1_attn_k_proj.png)

![h.1.attn.k_proj newA_newB](../hide/figures/cross/newA_newB/h1_attn_k_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.attn.v_proj

![h.1.attn.v_proj old_newA](../hide/figures/cross/old_newA/h1_attn_v_proj.png)

![h.1.attn.v_proj old_newB](../hide/figures/cross/old_newB/h1_attn_v_proj.png)

![h.1.attn.v_proj newA_newB](../hide/figures/cross/newA_newB/h1_attn_v_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.attn.o_proj

![h.1.attn.o_proj old_newA](../hide/figures/cross/old_newA/h1_attn_o_proj.png)

![h.1.attn.o_proj old_newB](../hide/figures/cross/old_newB/h1_attn_o_proj.png)

![h.1.attn.o_proj newA_newB](../hide/figures/cross/newA_newB/h1_attn_o_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

#### h.1.mlp.c_fc

![h.1.mlp.c_fc old_newA](../hide/figures/cross/old_newA/h1_mlp_c_fc.png)

![h.1.mlp.c_fc old_newB](../hide/figures/cross/old_newB/h1_mlp_c_fc.png)

![h.1.mlp.c_fc newA_newB](../hide/figures/cross/newA_newB/h1_mlp_c_fc.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.mlp.down_proj

![h.1.mlp.down_proj old_newA](../hide/figures/cross/old_newA/h1_mlp_down_proj.png)

![h.1.mlp.down_proj old_newB](../hide/figures/cross/old_newB/h1_mlp_down_proj.png)

![h.1.mlp.down_proj newA_newB](../hide/figures/cross/newA_newB/h1_mlp_down_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

### Layer 2

#### h.2.attn.q_proj

![h.2.attn.q_proj old_newA](../hide/figures/cross/old_newA/h2_attn_q_proj.png)

![h.2.attn.q_proj old_newB](../hide/figures/cross/old_newB/h2_attn_q_proj.png)

![h.2.attn.q_proj newA_newB](../hide/figures/cross/newA_newB/h2_attn_q_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.attn.k_proj

![h.2.attn.k_proj old_newA](../hide/figures/cross/old_newA/h2_attn_k_proj.png)

![h.2.attn.k_proj old_newB](../hide/figures/cross/old_newB/h2_attn_k_proj.png)

![h.2.attn.k_proj newA_newB](../hide/figures/cross/newA_newB/h2_attn_k_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.attn.v_proj

![h.2.attn.v_proj old_newA](../hide/figures/cross/old_newA/h2_attn_v_proj.png)

![h.2.attn.v_proj old_newB](../hide/figures/cross/old_newB/h2_attn_v_proj.png)

![h.2.attn.v_proj newA_newB](../hide/figures/cross/newA_newB/h2_attn_v_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.attn.o_proj

![h.2.attn.o_proj old_newA](../hide/figures/cross/old_newA/h2_attn_o_proj.png)

![h.2.attn.o_proj old_newB](../hide/figures/cross/old_newB/h2_attn_o_proj.png)

![h.2.attn.o_proj newA_newB](../hide/figures/cross/newA_newB/h2_attn_o_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

#### h.2.mlp.c_fc

![h.2.mlp.c_fc old_newA](../hide/figures/cross/old_newA/h2_mlp_c_fc.png)

![h.2.mlp.c_fc old_newB](../hide/figures/cross/old_newB/h2_mlp_c_fc.png)

![h.2.mlp.c_fc newA_newB](../hide/figures/cross/newA_newB/h2_mlp_c_fc.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.mlp.down_proj

![h.2.mlp.down_proj old_newA](../hide/figures/cross/old_newA/h2_mlp_down_proj.png)

![h.2.mlp.down_proj old_newB](../hide/figures/cross/old_newB/h2_mlp_down_proj.png)

![h.2.mlp.down_proj newA_newB](../hide/figures/cross/newA_newB/h2_mlp_down_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

### Layer 3

#### h.3.attn.q_proj

![h.3.attn.q_proj old_newA](../hide/figures/cross/old_newA/h3_attn_q_proj.png)

![h.3.attn.q_proj old_newB](../hide/figures/cross/old_newB/h3_attn_q_proj.png)

![h.3.attn.q_proj newA_newB](../hide/figures/cross/newA_newB/h3_attn_q_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.attn.k_proj

![h.3.attn.k_proj old_newA](../hide/figures/cross/old_newA/h3_attn_k_proj.png)

![h.3.attn.k_proj old_newB](../hide/figures/cross/old_newB/h3_attn_k_proj.png)

![h.3.attn.k_proj newA_newB](../hide/figures/cross/newA_newB/h3_attn_k_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.attn.v_proj

![h.3.attn.v_proj old_newA](../hide/figures/cross/old_newA/h3_attn_v_proj.png)

![h.3.attn.v_proj old_newB](../hide/figures/cross/old_newB/h3_attn_v_proj.png)

![h.3.attn.v_proj newA_newB](../hide/figures/cross/newA_newB/h3_attn_v_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.attn.o_proj

![h.3.attn.o_proj old_newA](../hide/figures/cross/old_newA/h3_attn_o_proj.png)

![h.3.attn.o_proj old_newB](../hide/figures/cross/old_newB/h3_attn_o_proj.png)

![h.3.attn.o_proj newA_newB](../hide/figures/cross/newA_newB/h3_attn_o_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

#### h.3.mlp.c_fc

![h.3.mlp.c_fc old_newA](../hide/figures/cross/old_newA/h3_mlp_c_fc.png)

![h.3.mlp.c_fc old_newB](../hide/figures/cross/old_newB/h3_mlp_c_fc.png)

![h.3.mlp.c_fc newA_newB](../hide/figures/cross/newA_newB/h3_mlp_c_fc.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.mlp.down_proj

![h.3.mlp.down_proj old_newA](../hide/figures/cross/old_newA/h3_mlp_down_proj.png)

![h.3.mlp.down_proj old_newB](../hide/figures/cross/old_newB/h3_mlp_down_proj.png)

![h.3.mlp.down_proj newA_newB](../hide/figures/cross/newA_newB/h3_mlp_down_proj.png)
