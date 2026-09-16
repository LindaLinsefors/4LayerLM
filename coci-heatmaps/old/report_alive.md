# CI co-activation per matrix — alive components

For each model and each decomposed weight matrix: **Pearson r of the per-token causal importance** (lower_leaky, `sampling="continuous"`) between every pair of the matrix's subcomponents, i.e. the co-CI measure of the pile-qk-comps reports, here for all 6 matrix types. Samples: pile_4l over 4,000 cached Pile training rows (2.05M tokens) / simple_2l over 6,000 cached SimpleStories stories (1.72M tokens) (SimpleStories batches padded; pad positions excluded).

Axes: components sorted by **harvest-DB mean CI, descending** (ties by component id); id tick labels only where they fit (n ≤ 120), otherwise the axis is the rank. Diagonal = 1. **Gray** = component with zero CI variance in the sample (r undefined). Alive = harvest mean CI > 1e-6 (project convention: 9,973 / 6,535 components).

## pile_4l

### Layer 0

#### h.0.attn.q_proj — 111 alive of 512

![h.0.attn.q_proj](../hide/figures/pile_4l/alive/h0_attn_q_proj.png)

#### h.0.attn.k_proj — 126 alive of 512

![h.0.attn.k_proj](../hide/figures/pile_4l/alive/h0_attn_k_proj.png)

#### h.0.attn.v_proj — 514 alive of 1024

![h.0.attn.v_proj](../hide/figures/pile_4l/alive/h0_attn_v_proj.png)

#### h.0.attn.o_proj — 446 alive of 1024

![h.0.attn.o_proj](../hide/figures/pile_4l/alive/h0_attn_o_proj.png)

#### h.0.mlp.c_fc — 1380 alive of 3072

![h.0.mlp.c_fc](../hide/figures/pile_4l/alive/h0_mlp_c_fc.png)

#### h.0.mlp.down_proj — 1133 alive of 3584

![h.0.mlp.down_proj](../hide/figures/pile_4l/alive/h0_mlp_down_proj.png)

### Layer 1

#### h.1.attn.q_proj — 15 alive of 512

![h.1.attn.q_proj](../hide/figures/pile_4l/alive/h1_attn_q_proj.png)

#### h.1.attn.k_proj — 48 alive of 512

![h.1.attn.k_proj](../hide/figures/pile_4l/alive/h1_attn_k_proj.png)

#### h.1.attn.v_proj — 226 alive of 1024

![h.1.attn.v_proj](../hide/figures/pile_4l/alive/h1_attn_v_proj.png)

#### h.1.attn.o_proj — 97 alive of 1024

![h.1.attn.o_proj](../hide/figures/pile_4l/alive/h1_attn_o_proj.png)

#### h.1.mlp.c_fc — 251 alive of 3072

![h.1.mlp.c_fc](../hide/figures/pile_4l/alive/h1_mlp_c_fc.png)

#### h.1.mlp.down_proj — 211 alive of 3584

![h.1.mlp.down_proj](../hide/figures/pile_4l/alive/h1_mlp_down_proj.png)

### Layer 2

#### h.2.attn.q_proj — 92 alive of 512

![h.2.attn.q_proj](../hide/figures/pile_4l/alive/h2_attn_q_proj.png)

#### h.2.attn.k_proj — 167 alive of 512

![h.2.attn.k_proj](../hide/figures/pile_4l/alive/h2_attn_k_proj.png)

#### h.2.attn.v_proj — 508 alive of 1024

![h.2.attn.v_proj](../hide/figures/pile_4l/alive/h2_attn_v_proj.png)

#### h.2.attn.o_proj — 525 alive of 1024

![h.2.attn.o_proj](../hide/figures/pile_4l/alive/h2_attn_o_proj.png)

#### h.2.mlp.c_fc — 292 alive of 3072

![h.2.mlp.c_fc](../hide/figures/pile_4l/alive/h2_mlp_c_fc.png)

#### h.2.mlp.down_proj — 359 alive of 3584

![h.2.mlp.down_proj](../hide/figures/pile_4l/alive/h2_mlp_down_proj.png)

### Layer 3

#### h.3.attn.q_proj — 36 alive of 512

![h.3.attn.q_proj](../hide/figures/pile_4l/alive/h3_attn_q_proj.png)

#### h.3.attn.k_proj — 60 alive of 512

![h.3.attn.k_proj](../hide/figures/pile_4l/alive/h3_attn_k_proj.png)

#### h.3.attn.v_proj — 249 alive of 1024

![h.3.attn.v_proj](../hide/figures/pile_4l/alive/h3_attn_v_proj.png)

#### h.3.attn.o_proj — 246 alive of 1024

![h.3.attn.o_proj](../hide/figures/pile_4l/alive/h3_attn_o_proj.png)

#### h.3.mlp.c_fc — 1031 alive of 3072

![h.3.mlp.c_fc](../hide/figures/pile_4l/alive/h3_mlp_c_fc.png)

#### h.3.mlp.down_proj — 1850 alive of 3584

![h.3.mlp.down_proj](../hide/figures/pile_4l/alive/h3_mlp_down_proj.png)

## simple_2l

### Layer 0

#### h.0.attn.q_proj — 251 alive of 288

![h.0.attn.q_proj](../hide/figures/simple_2l/alive/h0_attn_q_proj.png)

#### h.0.attn.k_proj — 248 alive of 288

![h.0.attn.k_proj](../hide/figures/simple_2l/alive/h0_attn_k_proj.png)

#### h.0.attn.v_proj — 344 alive of 384

![h.0.attn.v_proj](../hide/figures/simple_2l/alive/h0_attn_v_proj.png)

#### h.0.attn.o_proj — 432 alive of 480

![h.0.attn.o_proj](../hide/figures/simple_2l/alive/h0_attn_o_proj.png)

#### h.0.mlp.c_fc — 1068 alive of 1152

![h.0.mlp.c_fc](../hide/figures/simple_2l/alive/h0_mlp_c_fc.png)

#### h.0.mlp.down_proj — 886 alive of 960

![h.0.mlp.down_proj](../hide/figures/simple_2l/alive/h0_mlp_down_proj.png)

### Layer 1

#### h.1.attn.q_proj — 268 alive of 288

![h.1.attn.q_proj](../hide/figures/simple_2l/alive/h1_attn_q_proj.png)

#### h.1.attn.k_proj — 256 alive of 288

![h.1.attn.k_proj](../hide/figures/simple_2l/alive/h1_attn_k_proj.png)

#### h.1.attn.v_proj — 358 alive of 384

![h.1.attn.v_proj](../hide/figures/simple_2l/alive/h1_attn_v_proj.png)

#### h.1.attn.o_proj — 442 alive of 480

![h.1.attn.o_proj](../hide/figures/simple_2l/alive/h1_attn_o_proj.png)

#### h.1.mlp.c_fc — 1076 alive of 1152

![h.1.mlp.c_fc](../hide/figures/simple_2l/alive/h1_mlp_c_fc.png)

#### h.1.mlp.down_proj — 906 alive of 960

![h.1.mlp.down_proj](../hide/figures/simple_2l/alive/h1_mlp_down_proj.png)
