# Cross-decomposition U-cosine similarity: old vs newA vs newB

Same structure as the co-CI report [report_co_ci.md](report_co_ci.md), but instead of co-CI each heatmap shows the **absolute cosine similarity |cos(U_a, U_b)| between the write vectors of an alive component of one decomposition and one of another** — U is the output (d_out) factor of the rank-one subcomponent V_c U_c^T. Pairs (old, newA), (old, newB), (newA, newB); the first-named decomposition is the y axis. Absolute value because a component's sign is gauge ((V_c, U_c) -> (-V_c, -U_c) is the same component). Color white -> red spans 0 -> 1; the random baseline is E|cos| = sqrt(2/(pi d)) ~= 0.03 for d = 768, 0.014 for d = 3072.

Alive components only, in **exactly the axis orders of [report_co_ci.md](report_co_ci.md)** (y: descending mean CI; x: each component at its best-co-CI-matching y component's position when that r >= 0.3, unmatched at the right by mean CI — the matching comes from co-CI r, NOT from the cosines), so every cell is directly comparable across the three reports: a diagonal-band cell that is red both there and here is a matched mechanism whose factors also geometrically agree.

### Layer 0

#### h.0.attn.q_proj

![h.0.attn.q_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h0_attn_q_proj.png)

![h.0.attn.q_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h0_attn_q_proj.png)

![h.0.attn.q_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h0_attn_q_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.attn.k_proj

![h.0.attn.k_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h0_attn_k_proj.png)

![h.0.attn.k_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h0_attn_k_proj.png)

![h.0.attn.k_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h0_attn_k_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.attn.v_proj

![h.0.attn.v_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h0_attn_v_proj.png)

![h.0.attn.v_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h0_attn_v_proj.png)

![h.0.attn.v_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h0_attn_v_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.attn.o_proj

![h.0.attn.o_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h0_attn_o_proj.png)

![h.0.attn.o_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h0_attn_o_proj.png)

![h.0.attn.o_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h0_attn_o_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

#### h.0.mlp.c_fc

![h.0.mlp.c_fc U old_newA](../hide/figures/cross_cos/U/old_newA/h0_mlp_c_fc.png)

![h.0.mlp.c_fc U old_newB](../hide/figures/cross_cos/U/old_newB/h0_mlp_c_fc.png)

![h.0.mlp.c_fc U newA_newB](../hide/figures/cross_cos/U/newA_newB/h0_mlp_c_fc.png)

<hr style="height:10px;background:#555;border:none;">

#### h.0.mlp.down_proj

![h.0.mlp.down_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h0_mlp_down_proj.png)

![h.0.mlp.down_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h0_mlp_down_proj.png)

![h.0.mlp.down_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h0_mlp_down_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

### Layer 1

#### h.1.attn.q_proj

![h.1.attn.q_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h1_attn_q_proj.png)

![h.1.attn.q_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h1_attn_q_proj.png)

![h.1.attn.q_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h1_attn_q_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.attn.k_proj

![h.1.attn.k_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h1_attn_k_proj.png)

![h.1.attn.k_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h1_attn_k_proj.png)

![h.1.attn.k_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h1_attn_k_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.attn.v_proj

![h.1.attn.v_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h1_attn_v_proj.png)

![h.1.attn.v_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h1_attn_v_proj.png)

![h.1.attn.v_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h1_attn_v_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.attn.o_proj

![h.1.attn.o_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h1_attn_o_proj.png)

![h.1.attn.o_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h1_attn_o_proj.png)

![h.1.attn.o_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h1_attn_o_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

#### h.1.mlp.c_fc

![h.1.mlp.c_fc U old_newA](../hide/figures/cross_cos/U/old_newA/h1_mlp_c_fc.png)

![h.1.mlp.c_fc U old_newB](../hide/figures/cross_cos/U/old_newB/h1_mlp_c_fc.png)

![h.1.mlp.c_fc U newA_newB](../hide/figures/cross_cos/U/newA_newB/h1_mlp_c_fc.png)

<hr style="height:10px;background:#555;border:none;">

#### h.1.mlp.down_proj

![h.1.mlp.down_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h1_mlp_down_proj.png)

![h.1.mlp.down_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h1_mlp_down_proj.png)

![h.1.mlp.down_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h1_mlp_down_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

### Layer 2

#### h.2.attn.q_proj

![h.2.attn.q_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h2_attn_q_proj.png)

![h.2.attn.q_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h2_attn_q_proj.png)

![h.2.attn.q_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h2_attn_q_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.attn.k_proj

![h.2.attn.k_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h2_attn_k_proj.png)

![h.2.attn.k_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h2_attn_k_proj.png)

![h.2.attn.k_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h2_attn_k_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.attn.v_proj

![h.2.attn.v_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h2_attn_v_proj.png)

![h.2.attn.v_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h2_attn_v_proj.png)

![h.2.attn.v_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h2_attn_v_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.attn.o_proj

![h.2.attn.o_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h2_attn_o_proj.png)

![h.2.attn.o_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h2_attn_o_proj.png)

![h.2.attn.o_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h2_attn_o_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

#### h.2.mlp.c_fc

![h.2.mlp.c_fc U old_newA](../hide/figures/cross_cos/U/old_newA/h2_mlp_c_fc.png)

![h.2.mlp.c_fc U old_newB](../hide/figures/cross_cos/U/old_newB/h2_mlp_c_fc.png)

![h.2.mlp.c_fc U newA_newB](../hide/figures/cross_cos/U/newA_newB/h2_mlp_c_fc.png)

<hr style="height:10px;background:#555;border:none;">

#### h.2.mlp.down_proj

![h.2.mlp.down_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h2_mlp_down_proj.png)

![h.2.mlp.down_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h2_mlp_down_proj.png)

![h.2.mlp.down_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h2_mlp_down_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

### Layer 3

#### h.3.attn.q_proj

![h.3.attn.q_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h3_attn_q_proj.png)

![h.3.attn.q_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h3_attn_q_proj.png)

![h.3.attn.q_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h3_attn_q_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.attn.k_proj

![h.3.attn.k_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h3_attn_k_proj.png)

![h.3.attn.k_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h3_attn_k_proj.png)

![h.3.attn.k_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h3_attn_k_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.attn.v_proj

![h.3.attn.v_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h3_attn_v_proj.png)

![h.3.attn.v_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h3_attn_v_proj.png)

![h.3.attn.v_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h3_attn_v_proj.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.attn.o_proj

![h.3.attn.o_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h3_attn_o_proj.png)

![h.3.attn.o_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h3_attn_o_proj.png)

![h.3.attn.o_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h3_attn_o_proj.png)

<hr style="height:10px;background:#555;border:none;">

<hr style="height:10px;background:#555;border:none;">

#### h.3.mlp.c_fc

![h.3.mlp.c_fc U old_newA](../hide/figures/cross_cos/U/old_newA/h3_mlp_c_fc.png)

![h.3.mlp.c_fc U old_newB](../hide/figures/cross_cos/U/old_newB/h3_mlp_c_fc.png)

![h.3.mlp.c_fc U newA_newB](../hide/figures/cross_cos/U/newA_newB/h3_mlp_c_fc.png)

<hr style="height:10px;background:#555;border:none;">

#### h.3.mlp.down_proj

![h.3.mlp.down_proj U old_newA](../hide/figures/cross_cos/U/old_newA/h3_mlp_down_proj.png)

![h.3.mlp.down_proj U old_newB](../hide/figures/cross_cos/U/old_newB/h3_mlp_down_proj.png)

![h.3.mlp.down_proj U newA_newB](../hide/figures/cross_cos/U/newA_newB/h3_mlp_down_proj.png)
