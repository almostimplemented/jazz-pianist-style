---
license: apache-2.0
tags:
  - music
  - symbolic-music
  - midi
  - music-generation
  - jazz
library_name: pytorch
---

# Jazz Pianist Style — Conditional Generator

Gated cross-attention conditional generator from the ISMIR 2026 paper
**Learning Jazz Pianist Style with Cross-Attention Conditioning**
(Edwards, Maezawa, Dixon).

Aria-medium (659M) with cross-attention over 12 learned pianist embeddings
inserted in the last 8 of 16 transformer layers, fine-tuned on PiJAMA-12.
Conditioning persists through long generations rather than decaying like a
prompt prefix: a sliding-window classifier attributes conditioned
continuations to the intended pianist about 33 points more often than
unconditioned baselines.

## Files

| File | Contents |
|---|---|
| `model.safetensors` | Generator weights (base + cross-attention adapter) |
| `artist_embeddings.safetensors` | 12 artist embeddings, 4 context vectors each |
| `config.json` | Architecture and training hyperparameters |

## Usage

```python
from safetensors.torch import load_file
from aria.config import load_model_config
from aria.model import ModelConfig
from llama_pijama.models import ArtistEmbedding, CrossAttentionTransformerLM

cfg = load_model_config("medium"); cfg["grad_checkpoint"] = False
model = CrossAttentionTransformerLM(
    ModelConfig(**cfg),
    {"layers": [8, 9, 10, 11, 12, 13, 14, 15], "dropout": 0.1, "gate_init": 0.1})
model.load_state_dict(load_file("model.safetensors"))

emb = ArtistEmbedding(num_artists=12, d_model=cfg["d_model"], context_length=4)
emb.load_state_dict(load_file("artist_embeddings.safetensors"))
```

Code, generation script, and evaluation:
<https://github.com/almostimplemented/jazz-pianist-style>

## Intended use and limitations

A research tool for studying pianist style, in the spirit of Dick Hyman's
pedagogical etudes — not a system for producing convincing forgeries. It
models 12 pianists selected for stylistic separability, so it says nothing
about artists outside that set. Generated music imitating a named performer
raises questions of consent and attribution; please consider the interests of
the artists whose styles are modeled.

## Credits

Built on [Aria](https://github.com/EleutherAI/aria) (Apache-2.0) and the
[PiJAMA](https://github.com/almostimplemented/PiJAMA) dataset.

```bibtex
@inproceedings{edwards2026jazzstyle,
  title     = {Learning Jazz Pianist Style with Cross-Attention Conditioning},
  author    = {Edwards, Drew and Maezawa, Akira and Dixon, Simon},
  booktitle = {Proc. Int. Society for Music Information Retrieval Conf. (ISMIR)},
  year      = {2026},
}
```
