# Learning Jazz Pianist Style with Cross-Attention Conditioning

Models, training, and evaluation code for the ISMIR 2026 paper.

**Demo and listening examples:** <https://almostimplemented.github.io/jazz-pianist-style>

Jazz pianists develop distinctive traits that experienced listeners can often
identify within seconds. This work studies that identity through a pretrained
symbolic music transformer: a classifier built on Aria representations
identifies pianists with high accuracy, a gated cross-attention adapter over
learned pianist embeddings conditions generation on a specific artist, and the
classifier is then repurposed to locate the moments in a performance that most
carry an artist's signature.

## Install

```bash
git clone https://github.com/almostimplemented/jazz-pianist-style
cd jazz-pianist-style
uv sync          # or: pip install -e .
```

Runs on CUDA, Apple Silicon (MPS), or CPU. Training the full generator needs a
40GB GPU; everything else runs on a laptop.

## Quickstart

Generate a continuation in the style of a pianist:

```bash
python scripts/generate.py \
    --checkpoint-dir checkpoints/generator_best \
    --artist-map data/eval/artist_to_id.json \
    --artist "Art Tatum" --num-samples 2 --out-dir samples/
```

Score a classifier on the real test split:

```bash
python scripts/evaluate_classifier.py \
    --checkpoint checkpoints/pijama12_classifier.pt \
    --test-jsonl data/pijama12_1024/test.jsonl \
    --artist-map data/pijama12_1024/artist_to_id.json \
    --out results/eval.json
```

See `data/README.md` for building the JSONL splits from PiJAMA.

## What is here

| Path | Contents |
|---|---|
| `llama_pijama/models/` | Gated cross-attention adapter, artist embeddings, KV-cached inference |
| `llama_pijama/tokenization/` | Aria tokenizer extended with artist tokens |
| `llama_pijama/evaluation/` | Inference, chunk metrics, track aggregation |
| `llama_pijama/analysis/` | Memorization and similarity checks |
| `llama_pijama/external/cheston_dpi/` | Vendored ResNet-50 baseline (MIT, see below) |
| `scripts/` | Training, generation, and evaluation entry points |
| `scripts/analysis/` | Memorization, diversity, and ResNet transfer experiments |

## Reproducing the paper

Every number below comes from the released checkpoints via the commands
shown. Times are for an M-series laptop (MPS).

| Paper result | Command | Expected | Runtime |
|---|---|---|---|
| Table 1, PiJAMA-12 classifier | `evaluate_classifier.py` with the classifier checkpoint | 95.8 chunk / 98.8 track | ~3 min |
| Table 3, synthetic-only classifier | `evaluate_classifier.py` with the synthetic classifier | 87.1 chunk / 95.0 track | ~3 min |
| Table 3, from-scratch ResNet-50 | `analysis/train_resnet_transfer.py --eval-checkpoint` | 70.2 clip / 90.0 track¹ | ~10 min |
| Figure 6, characteristic regions | `characteristic_regions.py --artist "Art Tatum" --track-contains Sophisticated` | peak z +1.9 / trough z −1.2 | ~1 min |
| Figure 3, agreement curves | `agreement_eval.py --prompt-length 256` | ~70% mean agreement | hours² |
| Table 2, perplexity | `train_cross_attention.py` validation pass | CA 6.82 / FT 6.96 | ~1 h² |

¹ Track accuracy by mean softmax; the paper's 91.3 is majority vote over the
same predictions.
² Generation-bound; practical on a CUDA GPU, slow on a laptop.

Track-level majority vote breaks exact ties by mean logit — the convention
stated in the paper's tables and implemented in
`llama_pijama/evaluation/track_aggregation.py`.

## Training

```bash
python scripts/train_cross_attention.py \
    --train-jsonl data/pijama12_4096/train.jsonl \
    --val-jsonl data/pijama12_4096/val.jsonl \
    --pretrained-checkpoint checkpoints/aria-medium-gen.safetensors \
    --out-dir checkpoints/run1
```

Defaults reproduce the paper's run: cross-attention on the last 8 of 16
layers, a 4-vector artist context, 15 epochs at effective batch size 32, peak
LR 5e-6 with half an epoch of warmup then cosine decay, weight decay 0.02,
fp16, seed 42. `--no-cross-attention` trains the unconditioned ablation.

## Checkpoints

Released on the Hugging Face Hub (see the demo page for links): the
conditioned generator, the PiJAMA-12 classifier used as the evaluation
instrument, and the classifier trained only on generated music.

## License

Apache-2.0, matching [Aria](https://github.com/EleutherAI/aria), which this
work builds on. The released checkpoints carry the same license.
`llama_pijama/external/cheston_dpi/` remains under its original MIT license.

## Credits

Built on [Aria](https://github.com/EleutherAI/aria) (Apache-2.0) and the
[PiJAMA](https://github.com/almostimplemented/PiJAMA) dataset.
`llama_pijama/external/cheston_dpi/` is vendored from
[deep-pianist-identification](https://github.com/HuwCheston/deep-pianist-identification)
(MIT) for the ResNet-50 comparison.

## Citation

```bibtex
@inproceedings{edwards2026jazzstyle,
  title     = {Learning Jazz Pianist Style with Cross-Attention Conditioning},
  author    = {Edwards, Drew and Maezawa, Akira and Dixon, Simon},
  booktitle = {Proc. Int. Society for Music Information Retrieval Conf. (ISMIR)},
  year      = {2026},
}
```
