#%%

import torch

import importlib, load, circles
importlib.reload(load)
importlib.reload(circles)  # after load: circles does `import load` at its top

from load import load_pile_4l, load_simple_2l, pile_samples, simple_samples
from circles import text_change, text_jacobian


pile_model, pile_parameter_components, pile_tokenizer = load_pile_4l()
simple_model, simple_parameter_components, simple_tokenizer = load_simple_2l()

pile_sample_rows = pile_samples(5)

pile_sample_texts = [
    pile_tokenizer.decode(row) for row in pile_sample_rows
]

simple_sample_rows = simple_samples(5)

simple_sample_texts = [
    simple_tokenizer.decode(row) for row in simple_sample_rows
]

#%%
print(simple_sample_texts[0])

# %%
print(simple_sample_texts[1])

# %%
for token_id in range(100):
    print(simple_tokenizer.decode([token_id]))

# %%
for token_id in range(100, 200):
    print(simple_tokenizer.decode([token_id]))
# %%

for token_id in range(200, 300):
    print(simple_tokenizer.decode([token_id]))
# %%
for token_id in range(3919, 4019):
    print(simple_tokenizer.decode([token_id]))
# %%
