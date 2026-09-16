# CI co-activation per matrix — all components

For each model and each decomposed weight matrix: **Pearson r of the per-token causal importance** (lower_leaky, `sampling="continuous"`) between every pair of the matrix's subcomponents, i.e. the co-CI measure of the pile-qk-comps reports, here for all 6 matrix types. Samples: pile_4l over 4,000 cached Pile training rows (2.05M tokens) / simple_2l over 6,000 cached SimpleStories stories (1.72M tokens) (SimpleStories batches padded; pad positions excluded).

Axes: components sorted by **harvest-DB mean CI, descending** (ties by component id); id tick labels only where they fit (n ≤ 120), otherwise the axis is the rank. Diagonal = 1. **Gray** = component with zero CI variance in the sample (r undefined). Alive = harvest mean CI > 1e-6 (project convention: 9,973 / 6,535 components).

## pile_4l

### Layer 0

#### h.0.attn.q_proj — all 512 components (111 alive, 34 zero-variance)

![h.0.attn.q_proj](../hide/figures/pile_4l/all/h0_attn_q_proj.png)

#### h.0.attn.k_proj — all 512 components (126 alive, 30 zero-variance)

![h.0.attn.k_proj](../hide/figures/pile_4l/all/h0_attn_k_proj.png)

#### h.0.attn.v_proj — all 1024 components (514 alive, 58 zero-variance)

![h.0.attn.v_proj](../hide/figures/pile_4l/all/h0_attn_v_proj.png)

#### h.0.attn.o_proj — all 1024 components (446 alive, 60 zero-variance)

![h.0.attn.o_proj](../hide/figures/pile_4l/all/h0_attn_o_proj.png)

#### h.0.mlp.c_fc — all 3072 components (1380 alive, 132 zero-variance)

![h.0.mlp.c_fc](../hide/figures/pile_4l/all/h0_mlp_c_fc.png)

#### h.0.mlp.down_proj — all 3584 components (1133 alive, 239 zero-variance)

![h.0.mlp.down_proj](../hide/figures/pile_4l/all/h0_mlp_down_proj.png)

### Layer 1

#### h.1.attn.q_proj — all 512 components (15 alive, 38 zero-variance)

![h.1.attn.q_proj](../hide/figures/pile_4l/all/h1_attn_q_proj.png)

#### h.1.attn.k_proj — all 512 components (48 alive, 44 zero-variance)

![h.1.attn.k_proj](../hide/figures/pile_4l/all/h1_attn_k_proj.png)

#### h.1.attn.v_proj — all 1024 components (226 alive, 53 zero-variance)

![h.1.attn.v_proj](../hide/figures/pile_4l/all/h1_attn_v_proj.png)

#### h.1.attn.o_proj — all 1024 components (97 alive, 50 zero-variance)

![h.1.attn.o_proj](../hide/figures/pile_4l/all/h1_attn_o_proj.png)

#### h.1.mlp.c_fc — all 3072 components (251 alive, 245 zero-variance)

![h.1.mlp.c_fc](../hide/figures/pile_4l/all/h1_mlp_c_fc.png)

#### h.1.mlp.down_proj — all 3584 components (211 alive, 364 zero-variance)

![h.1.mlp.down_proj](../hide/figures/pile_4l/all/h1_mlp_down_proj.png)

### Layer 2

#### h.2.attn.q_proj — all 512 components (92 alive, 27 zero-variance)

![h.2.attn.q_proj](../hide/figures/pile_4l/all/h2_attn_q_proj.png)

#### h.2.attn.k_proj — all 512 components (167 alive, 23 zero-variance)

![h.2.attn.k_proj](../hide/figures/pile_4l/all/h2_attn_k_proj.png)

#### h.2.attn.v_proj — all 1024 components (508 alive, 34 zero-variance)

![h.2.attn.v_proj](../hide/figures/pile_4l/all/h2_attn_v_proj.png)

#### h.2.attn.o_proj — all 1024 components (525 alive, 31 zero-variance)

![h.2.attn.o_proj](../hide/figures/pile_4l/all/h2_attn_o_proj.png)

#### h.2.mlp.c_fc — all 3072 components (292 alive, 232 zero-variance)

![h.2.mlp.c_fc](../hide/figures/pile_4l/all/h2_mlp_c_fc.png)

#### h.2.mlp.down_proj — all 3584 components (359 alive, 317 zero-variance)

![h.2.mlp.down_proj](../hide/figures/pile_4l/all/h2_mlp_down_proj.png)

### Layer 3

#### h.3.attn.q_proj — all 512 components (36 alive, 33 zero-variance)

![h.3.attn.q_proj](../hide/figures/pile_4l/all/h3_attn_q_proj.png)

#### h.3.attn.k_proj — all 512 components (60 alive, 37 zero-variance)

![h.3.attn.k_proj](../hide/figures/pile_4l/all/h3_attn_k_proj.png)

#### h.3.attn.v_proj — all 1024 components (249 alive, 62 zero-variance)

![h.3.attn.v_proj](../hide/figures/pile_4l/all/h3_attn_v_proj.png)

#### h.3.attn.o_proj — all 1024 components (246 alive, 51 zero-variance)

![h.3.attn.o_proj](../hide/figures/pile_4l/all/h3_attn_o_proj.png)

#### h.3.mlp.c_fc — all 3072 components (1031 alive, 187 zero-variance)

![h.3.mlp.c_fc](../hide/figures/pile_4l/all/h3_mlp_c_fc.png)

#### h.3.mlp.down_proj — all 3584 components (1850 alive, 189 zero-variance)

![h.3.mlp.down_proj](../hide/figures/pile_4l/all/h3_mlp_down_proj.png)

## simple_2l

### Layer 0

#### h.0.attn.q_proj — all 288 components (251 alive, 5 zero-variance)

![h.0.attn.q_proj](../hide/figures/simple_2l/all/h0_attn_q_proj.png)

#### h.0.attn.k_proj — all 288 components (248 alive, 11 zero-variance)

![h.0.attn.k_proj](../hide/figures/simple_2l/all/h0_attn_k_proj.png)

#### h.0.attn.v_proj — all 384 components (344 alive, 8 zero-variance)

![h.0.attn.v_proj](../hide/figures/simple_2l/all/h0_attn_v_proj.png)

#### h.0.attn.o_proj — all 480 components (432 alive, 15 zero-variance)

![h.0.attn.o_proj](../hide/figures/simple_2l/all/h0_attn_o_proj.png)

#### h.0.mlp.c_fc — all 1152 components (1068 alive, 22 zero-variance)

![h.0.mlp.c_fc](../hide/figures/simple_2l/all/h0_mlp_c_fc.png)

#### h.0.mlp.down_proj — all 960 components (886 alive, 24 zero-variance)

![h.0.mlp.down_proj](../hide/figures/simple_2l/all/h0_mlp_down_proj.png)

### Layer 1

#### h.1.attn.q_proj — all 288 components (268 alive, 8 zero-variance)

![h.1.attn.q_proj](../hide/figures/simple_2l/all/h1_attn_q_proj.png)

#### h.1.attn.k_proj — all 288 components (256 alive, 8 zero-variance)

![h.1.attn.k_proj](../hide/figures/simple_2l/all/h1_attn_k_proj.png)

#### h.1.attn.v_proj — all 384 components (358 alive, 6 zero-variance)

![h.1.attn.v_proj](../hide/figures/simple_2l/all/h1_attn_v_proj.png)

#### h.1.attn.o_proj — all 480 components (442 alive, 12 zero-variance)

![h.1.attn.o_proj](../hide/figures/simple_2l/all/h1_attn_o_proj.png)

#### h.1.mlp.c_fc — all 1152 components (1076 alive, 29 zero-variance)

![h.1.mlp.c_fc](../hide/figures/simple_2l/all/h1_mlp_c_fc.png)

#### h.1.mlp.down_proj — all 960 components (906 alive, 16 zero-variance)

![h.1.mlp.down_proj](../hide/figures/simple_2l/all/h1_mlp_down_proj.png)
