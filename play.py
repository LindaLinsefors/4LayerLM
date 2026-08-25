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

# %%
text_change(pile_model, pile_sample_rows[2][:5], 
            max_dist=1.0, n_points=10, in_layer=0, out_layer=4, 
            vary='last')
# %%

t = 5

logits = pile_model(pile_sample_rows[2][:t].unsqueeze(0))
predictions = torch.argmax(logits, dim=-1)
print("Predicted token IDs:", predictions)
print("Predicted tokens:", [pile_tokenizer.decode([token_id]) for token_id in predictions[0].tolist()])
print("Actual token IDs:", pile_sample_rows[2][1:t+1].tolist())
print("Actual tokens:", [pile_tokenizer.decode([token_id]) for token_id in pile_sample_rows[2][1:t+1].tolist()])
# %%
for in_layer in [0, 1, 2, 3]:
    for max_dist in [0.1, 1.0, 10.0]:
        text_change(pile_model, pile_sample_rows[2][:5], 
                    max_dist=max_dist, n_points=10, in_layer=in_layer, out_layer=4, 
                    vary='last', final_norm=False,
                    title=f"pile_model, in_layer={in_layer}, max_dist={max_dist}")

# %%
t = 10

logits = simple_model(simple_sample_rows[0][:t].unsqueeze(0))
predictions = torch.argmax(logits, dim=-1)
print("Predicted token IDs:", predictions)
print("Predicted tokens:", [simple_tokenizer.decode([token_id]) for token_id in predictions[0].tolist()])
print("Actual token IDs:", simple_sample_rows[0][1:t+1].tolist())
print("Actual tokens:", [simple_tokenizer.decode([token_id]) for token_id in simple_sample_rows[0][1:t+1].tolist()])
# %%

for in_layer in [0, 1]:
    for max_dist in [0.1, 1.0, 10.0]:
        text_change(simple_model, simple_sample_rows[0][:10], 
                    max_dist=max_dist, n_points=10, in_layer=in_layer, out_layer=2, 
                    vary='last', final_norm=False,
                    title=f"simple_model, in_layer={in_layer}, max_dist={max_dist}")

# %%
