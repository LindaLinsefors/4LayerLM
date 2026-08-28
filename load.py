"""Load the target models and their VPD decompositions (standalone, Python 3.11 OK).

Reads everything directly from the local prev_paper/models/ folder; no param_decomp import
needed. (The param-decomp-vpd library + its 3.13 venv are only required for the
decomposition *machinery* — masked forward passes, the causal-importance
function — not for loading.)

Each loader returns:
  model                 -- the target LlamaSimpleMLP transformer (eval mode)
  parameter_components  -- ParameterComponents: for each decomposed weight matrix
                           (e.g. "h.0.mlp.c_fc") a Subcomponents object with
                           factors V (d_in, C) and U (C, d_out), where the C
                           rank-one subcomponents reconstruct the weight as
                           W ~= (V @ U)^T   [nn.Linear stores W as (d_out, d_in)].
                           The causal-importance function's weights are kept as a
                           raw state dict in .ci_fn_state_dict.

Also fetches training-data samples: pile_samples() / simple_samples()
return rows of each model's training set as 1-D token-id tensors.
"""

#%%
#run load.py

import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent / "prev_paper"))
from model_def import LlamaSimpleMLP, LlamaSimpleMLPConfig

MODELS_DIR = Path(__file__).parent / "prev_paper" / "models"

PILE_4L = MODELS_DIR / "pile_4layer"
SIMPLE_2L = MODELS_DIR / "simplestories_2layer"


@dataclass
class Subcomponents:
    """Rank-one decomposition factors of one weight matrix: W ~= (V @ U)^T."""

    V: torch.Tensor  # (d_in, C)
    U: torch.Tensor  # (C, d_out)


class ParameterComponents:
    def __init__(self, decomposition_checkpoint: Path):
        state = torch.load(decomposition_checkpoint, map_location="cpu", weights_only=True)
        self.components: dict[str, Subcomponents] = {}
        self.ci_fn_state_dict: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            if key.startswith("_components."):
                # e.g. "_components.h-0-mlp-c_fc.U" -> module "h.0.mlp.c_fc", factor "U"
                _, module, factor = key.split(".")
                module = module.replace("-", ".")
                sub = self.components.setdefault(module, Subcomponents(V=None, U=None))
                setattr(sub, factor, tensor)
            elif key.startswith("ci_fn."):
                self.ci_fn_state_dict[key.removeprefix("ci_fn.")] = tensor
        assert all(s.U is not None and s.V is not None for s in self.components.values())

    def reconstruct(self, module: str) -> torch.Tensor:
        """The decomposition's approximation of the weight matrix of `module`,
        in nn.Linear orientation (d_out, d_in)."""
        sub = self.components[module]
        return (sub.V @ sub.U).T


def _load_target_model(target_dir: Path, checkpoint_name: str) -> LlamaSimpleMLP:
    with open(target_dir / "model_config.yaml") as f:
        config_dict = yaml.safe_load(f)
    config_dict.setdefault("model_type", "LlamaSimpleMLP")
    model = LlamaSimpleMLP(LlamaSimpleMLPConfig(**config_dict))

    checkpoint = target_dir / checkpoint_name
    if checkpoint.suffix == ".safetensors":
        state = load_file(checkpoint)
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    # safetensors deduplicates the tied wte/lm_head weight, so lm_head.weight may
    # be absent; the tie in __init__ makes loading wte.weight fill both.
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert set(missing) <= {"lm_head.weight"}, missing
    assert not unexpected, unexpected

    model.eval()
    model.requires_grad_(False)
    return model


def load_tokenizer(model_name: str):
    """Load the tokenizer for "pile_4l" (GPT-NeoX, vocab 50277) or "simple_2l"
    (SimpleStories GPT-2, vocab 4019) from the local tokenizer.json.

    Returns a transformers PreTrainedTokenizerFast: .encode(text) -> list[int],
    .decode(ids), .convert_ids_to_tokens(ids).
    """
    from transformers import PreTrainedTokenizerFast

    target_dir = {
        "pile_4l": PILE_4L / "target_model_t-9d2b8f02",
        "simple_2l": SIMPLE_2L / "target_model_gf6rbga0",
    }[model_name]
    return PreTrainedTokenizerFast(tokenizer_file=str(target_dir / "tokenizer.json"))


def load_pile_4l():
    """
    Load the Language model trained on the Pile dataset with 4 layers.

    Returns: (model, parameter_components, tokenizer)
    """
    model = _load_target_model(PILE_4L / "target_model_t-9d2b8f02", "model_step_99999.safetensors")
    parameter_components = ParameterComponents(
        PILE_4L / "vpd_decomposition_s-55ea3f9b" / "model_400000.pth"
    )
    return model, parameter_components, load_tokenizer("pile_4l")


class NoSpecialTokens:
    """Wrapper making a tokenizer's encode()/__call__ skip special tokens by
    default (the SimpleStories tokenizer otherwise appends [EOS] to every
    encoded text). Everything else is delegated to the wrapped tokenizer."""

    def __init__(self, tokenizer):
        self.wrapped = tokenizer

    def encode(self, *args, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return self.wrapped.encode(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return self.wrapped(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


def load_simple_2l():
    """
    Load the 2-layer language model trained on SimpleStories.

    Returns: (model, parameter_components, tokenizer)
    The tokenizer is wrapped so that it does NOT append [EOS] when encoding.
    """
    model = _load_target_model(SIMPLE_2L / "target_model_gf6rbga0", "model_step_99999.pt")
    parameter_components = ParameterComponents(
        SIMPLE_2L / "vpd_decomposition_s-eab2ace8" / "model_400000.pth"
    )
    return model, parameter_components, NoSpecialTokens(load_tokenizer("simple_2l"))


def load(model_name: str):
    """
    Load a model by name.

    Args:
        model_name (str): The name of the model to load. Supported values are:
            - "pile_4l": Load the 4-layer language model trained on the Pile dataset.
            - "simple_2l": Load the 2-layer language model trained on SimpleStories.

    Returns:
        tuple: (model, parameter_components, tokenizer)
    """
    if model_name == "pile_4l":
        return load_pile_4l()
    elif model_name == "simple_2l":
        return load_simple_2l()
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported names are 'pile_4l' and 'simple_2l'.")


# --- Training-data samples (via the public HuggingFace datasets-server REST API;
#     no HF login or `datasets` library needed) ---


def _rows(dataset: str, offset: int, length: int, split: str = "train") -> list[dict]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={urllib.parse.quote(dataset, safe='')}"
        f"&config=default&split={split}&offset={offset}&length={length}"
    )
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["rows"]


def pile_samples(n: int = 5, offset: int = 0) -> list[torch.Tensor]:
    """n rows of the 4-layer model's training data, as 1-D int64 token-id tensors.

    The dataset (danbraunai/pile-uncopyrighted-tok-shuffled) is pre-tokenized:
    513-token chunks of concatenated documents with <|endoftext|> boundaries
    mid-row, pre-shuffled — returned as-is, exactly what the model saw in
    training. Different `offset` values give different rows. Decode with
    load_tokenizer("pile_4l").decode(ids.tolist()).
    """
    rows = _rows("danbraunai/pile-uncopyrighted-tok-shuffled", offset, n)
    return [torch.tensor(row["row"]["input_ids"]) for row in rows]


def simple_samples(n: int = 5, offset: int = 0) -> list[torch.Tensor]:
    """n stories of the 2-layer model's training data, as 1-D int64 token-id tensors.

    The dataset (SimpleStories/SimpleStories) stores plain-text stories; each
    whole story is encoded with the model's tokenizer, without the [EOS] the
    tokenizer would otherwise append. Tensors have varying (story) lengths.
    """
    tok = load_tokenizer("simple_2l")
    rows = _rows("SimpleStories/SimpleStories", offset, n)
    return [
        torch.tensor(tok.encode(row["row"]["story"], add_special_tokens=False))
        for row in rows
    ]

# %%
