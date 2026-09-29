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

# Jazz Pianist Style — Pianist Classifier

Pianist identification model from the ISMIR 2026 paper **Learning Jazz Pianist
Style with Cross-Attention Conditioning** (Edwards, Maezawa, Dixon).

Aria-medium contrastive backbone fine-tuned to classify 12 PiJAMA pianists
from 1024-token windows. On the held-out test split it reaches **95.8% per
chunk and 98.8% per track**; on the external Deep Pianist Identification
benchmark the same recipe reaches 96.9% track accuracy, above the published
ResNet-50 baseline of 94.4%. The paper uses this model as its measuring
instrument: it scores generated music and locates the most characteristic
moments within a performance.

## Files

| File | Contents |
|---|---|
| `best.pt` | Classifier weights (state dict) |
| `config.json` | Architecture summary |

## Usage

```python
from llama_pijama.evaluation import load_model, run_inference

model, _ = load_model("best.pt", model_name="medium", num_classes=12, device="cpu")
```

Track-level scores aggregate chunk predictions by majority vote, with exact
vote ties broken by mean logit.

Code and evaluation scripts:
<https://github.com/almostimplemented/jazz-pianist-style>

## Intended use and limitations

Trained on 12 pianists chosen for stylistic separability; it cannot recognize
anyone else, and confident predictions outside that set are meaningless.
Performer identification carries obvious privacy and attribution
implications — this is a research artifact, not an attribution service.

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
