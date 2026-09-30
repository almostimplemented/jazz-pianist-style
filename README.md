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

Runs on CUDA, Apple silicon (MPS), or CPU. Training needs a GPU; evaluation
of the released checkpoints runs on a laptop.

## Checkpoints

Three models are on the Hugging Face Hub:

| Repository | Contents |
|---|---|
| [`almostimplemented/jazz-pianist-style-generator`](https://huggingface.co/almostimplemented/jazz-pianist-style-generator) | The conditional generator: Aria-medium with gated cross-attention and 12 learned pianist embeddings |
| [`almostimplemented/jazz-pianist-style-classifier`](https://huggingface.co/almostimplemented/jazz-pianist-style-classifier) | The PiJAMA-12 pianist classifier, the paper's measuring instrument |
| [`almostimplemented/jazz-pianist-style-synthetic-classifier`](https://huggingface.co/almostimplemented/jazz-pianist-style-synthetic-classifier) | The classifier trained only on generated music |

```bash
hf download almostimplemented/jazz-pianist-style-generator --local-dir checkpoints/generator
hf download almostimplemented/jazz-pianist-style-classifier --local-dir checkpoints/classifier
hf download almostimplemented/jazz-pianist-style-synthetic-classifier --local-dir checkpoints/synthetic-classifier
```

## Data

The commands below read tokenized splits built from the public
[PiJAMA](https://github.com/almostimplemented/PiJAMA) dataset. Build them once:

```bash
python scripts/build_dataset.py --midi-root /path/to/PiJAMA \
    --metadata-csv data/pijama12.csv --out-dir data/pijama12_4096 --max-seq-len 4096
python scripts/build_dataset.py --midi-root /path/to/PiJAMA \
    --metadata-csv data/pijama12.csv --out-dir data/pijama12_1024 --max-seq-len 1024
```

`data/README.md` has the details: split membership, trimming, and the
expected sequence counts.

## Quickstart

Generate a continuation in the style of a pianist:

```bash
python scripts/generate.py \
    --checkpoint-dir checkpoints/generator \
    --artist-map data/pijama12_4096/artist_to_id.json \
    --artist "Art Tatum" --num-samples 2 --out-dir samples/
```

Score the classifier on the real test split:

```bash
python scripts/evaluate_classifier.py \
    --checkpoint checkpoints/classifier/best.pt \
    --test-jsonl data/pijama12_1024/test.jsonl \
    --artist-map data/pijama12_1024/artist_to_id.json \
    --out results/real_clf_eval.json
```

## What is here

| Path | Contents |
|---|---|
| `llama_pijama/models/` | Gated cross-attention adapter, artist embeddings, KV-cached inference |
| `llama_pijama/tokenization/` | Aria tokenizer extended with artist tokens |
| `llama_pijama/training/` | Datasets and the classifier training loop |
| `llama_pijama/evaluation/` | Inference, chunk metrics, track aggregation |
| `llama_pijama/analysis/` | Memorization and similarity checks |
| `llama_pijama/external/cheston_dpi/` | Vendored ResNet-50 baseline (MIT, see below) |
| `scripts/` | Data building, training, generation, and evaluation entry points |
| `scripts/analysis/` | Memorization, diversity, confidence intervals, ResNet transfer |
| `scripts/site/` | Builders for the companion page's data and audio |

## Reproducing the paper

Run from the repository root after downloading the checkpoints and building
the data. Times are for an Apple-silicon laptop unless noted.

| Paper result | Script | Expected | Time |
|---|---|---|---|
| Table 1, PiJAMA-12 classifier | `evaluate_classifier.py` with `checkpoints/classifier/best.pt` | 95.8 chunk / 98.8 track | ~3 min |
| Table 3, synthetic-only classifier | `evaluate_classifier.py` with `checkpoints/synthetic-classifier/best.pt` | 87.1 chunk / 95.0 track | ~3 min |
| Table 2, perplexity | `perplexity_eval.py --model-type {cross_attention,baseline,pretrained}` on `pijama12_4096/test.jsonl` | 6.82 / 6.96 / 11.41 | ~20 min each |
| Table 2 / Fig. 3, agreement | `agreement_eval.py --prompt-length 256` on `pijama12_4096/val.jsonl`; `--base-checkpoint` for the two baselines | 70 / 37 / 25% | GPU hours¹ |
| Fig. 6, per-pianist agreement | `per_artist_mean_agreement` in the same output | 96% (Hank Jones) … 29% (Cedar Walton) | — |
| Table 2 CIs, Hyman sink | `analysis/bootstrap_cis.py` over the agreement and classifier outputs | e.g. 70 [64, 75] | ~1 min |
| Fig. 5, characteristic regions | `characteristic_regions.py --save-excerpts` on `pijama12_1024/test.jsonl` | median peak z 0.90, max 1.99; Tatum "Sophisticated Lady" +1.9 / −1.2 | ~20 min |
| Section 6, memorization | `analysis/jaccard_top2.py --source-map` over the synthetic corpus | 0.20 generated vs 0.32 real | CPU-bound |
| Section 6, diversity | `analysis/measure_diversity.py` | same-artist 1.5–3.5× between-artist | minutes |
| Table 3, from-scratch ResNet-50 | `analysis/build_resnet_transfer_clips.py`, then `analysis/train_resnet_transfer.py` | 70.2 clip / 91.3 track (synthetic arm) | GPU hours; eval ~10 min |

¹ Generation-bound: 175 continuations of 4,096 tokens per mode and prompt
length. Practical on a CUDA GPU (`--batch-size 16`), slow on a laptop.

Track-level accuracy is a majority vote over chunks, with exact ties broken by
mean logit among the tied pianists, as stated in the paper's tables.

Not included: the DPI-20 benchmark (Table 1, top rows) uses Cheston et al.'s
data and splits; the PiJAMA-30 classifier (Table 1) uses the same training
script on the 30-artist data; the paper's figures were drawn from these
outputs with plotting code not released here.

## Training

The generator (about 18 hours on one 24 GB GPU):

```bash
python scripts/train_cross_attention.py \
    --train-jsonl data/pijama12_4096/train.jsonl \
    --val-jsonl data/pijama12_4096/val.jsonl \
    --pretrained-checkpoint /path/to/aria-medium-base/model.safetensors \
    --out-dir checkpoints/generator_run
```

Defaults are the paper's run: cross-attention on the last 8 of 16 layers, a
4-vector artist context, gate initialised at 0.1, embedding dropout 0.3,
15 epochs at effective batch size 32, peak learning rate 5e-6 with half an
epoch of warmup then cosine decay, weight decay 0.02, fp16.
`--no-cross-attention` trains the fine-tuned baseline.

The classifier (the same recipe trains the PiJAMA-12, PiJAMA-30 and
synthetic-only models; only the data changes):

```bash
python scripts/train_classifier.py \
    --train-jsonl data/pijama12_1024/train.jsonl \
    --val-jsonl data/pijama12_1024/val.jsonl \
    --artist-map data/pijama12_1024/artist_to_id.json \
    --out-dir checkpoints/classifier_run
```

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
