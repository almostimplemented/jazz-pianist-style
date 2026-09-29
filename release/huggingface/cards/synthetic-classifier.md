---
license: apache-2.0
tags:
  - music
  - symbolic-music
  - midi
  - music-classification
  - jazz
library_name: pytorch
---

# Jazz Pianist Style — Synthetic-Only Classifier

Pianist classifier trained **exclusively on generated music**, from the ISMIR
2026 paper **Learning Jazz Pianist Style with Cross-Attention Conditioning**
(Edwards, Maezawa, Dixon).

Same architecture and recipe as the real-data classifier, but it never sees a
labeled real performance during training — only continuations sampled from the
conditional generator. Evaluated on real recordings it reaches **87.1% per
chunk and 95.0% per track**, evidence that the generated music carries
stylistic structure that transfers to actual performances.

## Files

| File | Contents |
|---|---|
| `best.pt` | Classifier weights (state dict) |
| `config.json` | Architecture summary |

## Usage

Identical to the real-data classifier:

```python
from llama_pijama.evaluation import load_model

model, _ = load_model("best.pt", model_name="medium", num_classes=12, device="cpu")
```

Code and evaluation scripts:
<https://github.com/almostimplemented/jazz-pianist-style>

## Intended use and limitations

Exists to measure what generated music preserves, not to be a better
classifier — the real-data model is stronger. Same 12-pianist limitation as
the rest of the family.

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
