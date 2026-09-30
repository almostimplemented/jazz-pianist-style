#!/usr/bin/env python3
"""Bootstrap confidence intervals for the paper's headline numbers.

Method: stratified track-level percentile cluster bootstrap. Windows overlap
~8x (stride 128, window 1024) and several continuations can share a source
track, so the track is the largest exchangeable unit; window counts are not
binomial trials. Tracks are resampled with replacement within artist strata
and every statistic is recomputed. Song-level classification accuracy is a
genuine one-outcome-per-track binomial, so it also gets an exact
Clopper-Pearson interval (the bootstrap degenerates at a 100% boundary).

Inputs are the outputs of this repository's evaluation scripts:
  agreement_eval.py      one JSON per (mode, prompt length), passed as
                         --agreement conditioned:256=results/agreement_cond_P256.json
                         Modes compared for the gap: conditioned vs finetuned_baseline.
  evaluate_classifier.py real- and synthetic-classifier test results

Example:
    python scripts/analysis/bootstrap_cis.py \
        --agreement conditioned:256=results/agreement_cond_P256.json \
        --agreement finetuned_baseline:256=results/agreement_ft_P256.json \
        --real-clf-eval results/pijama12_real_clf_eval.json \
        --synth-clf-eval results/pijama12_synth_clf_eval.json \
        --out results/bootstrap_cis.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta


def load_run(path: Path):
    """Per-track (correct, present) counts on the window-position grid."""
    d = json.loads(Path(path).read_text())
    positions = sorted({w["position"] for s in d["per_sample"] for w in s["windows"]})
    pos_idx = {p: i for i, p in enumerate(positions)}
    npos = len(positions)
    tracks: dict[str, list[np.ndarray]] = {}
    artist_of: dict[str, str] = {}
    for s in d["per_sample"]:
        t = s["track_id"]
        artist_of[t] = s["artist"]
        if t not in tracks:
            tracks[t] = [np.zeros(npos), np.zeros(npos)]
        c, n = tracks[t]
        for w in s["windows"]:
            i = pos_idx[w["position"]]
            n[i] += 1
            c[i] += bool(w["correct_prompt"])
    return tracks, artist_of, 100 * d["mean_agreement"], d


def curve_mean(c: np.ndarray, n: np.ndarray) -> float:
    """The paper's headline statistic: mean over positions of pooled accuracy."""
    mask = n > 0
    return float(100 * np.mean(c[mask] / n[mask]))


def stat(track_ids, tracks) -> float:
    npos = len(next(iter(tracks.values()))[0])
    c = np.zeros(npos)
    n = np.zeros(npos)
    for t in track_ids:
        tc, tn = tracks[t]
        c += tc
        n += tn
    return curve_mean(c, n)


def pct_ci(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 1), round(float(hi), 1)]


def clopper_pearson(k: int, n: int) -> list[float]:
    lo = 0.0 if k == 0 else 100 * beta.ppf(0.025, k, n - k + 1)
    hi = 100.0 if k == n else 100 * beta.ppf(0.975, k + 1, n - k)
    return [round(float(lo), 1), round(float(hi), 1)]


def continuation_block(runs, rng: np.random.Generator, B: int, out: dict) -> None:
    for P in sorted({p for (_m, p) in runs}):
        modes = [m for (m, p) in runs if p == P]
        data = {m: load_run(runs[(m, P)]) for m in modes}
        anchor = "conditioned" if "conditioned" in data else modes[0]
        ref = set(data[anchor][0])
        for m, (trk, *_rest) in data.items():
            assert set(trk) == ref, f"different prompt tracks: {m} vs {anchor} at P={P}"
        artist_of = data[anchor][1]
        by_artist = {a: sorted(t for t, aa in artist_of.items() if aa == a) for a in sorted(set(artist_of.values()))}
        strata = [(a, ts) for a, ts in by_artist.items()]

        # point estimates must match what each evaluation reported
        points = {}
        for m in modes:
            pt = stat(sorted(ref), data[m][0])
            assert abs(pt - data[m][2]) < 0.01, f"{m} P={P}: {pt:.2f} != reported {data[m][2]:.2f}"
            points[m] = pt

        # per-artist frame: tracks with >=1 classified window (conditioned job)
        cond_tracks = data[anchor][0]
        pa_strata = {
            a: [t for t in ts if cond_tracks[t][1].sum() > 0] for a, ts in by_artist.items()
        }
        pa_points = {a: stat(ts, cond_tracks) for a, ts in pa_strata.items() if ts}

        boot = {m: np.empty(B) for m in modes}
        has_gap = {"conditioned", "finetuned_baseline"} <= set(modes)
        boot["gap"] = np.empty(B)
        pa_boot = {a: np.empty(B) for a in pa_points}
        for b in range(B):
            chosen = []
            for a, ts in strata:
                idx = rng.integers(0, len(ts), len(ts))
                chosen.extend(ts[i] for i in idx)
            for m in modes:
                boot[m][b] = stat(chosen, data[m][0])
            if has_gap:
                boot["gap"][b] = boot["conditioned"][b] - boot["finetuned_baseline"][b]
            for a in pa_points:
                ts = pa_strata[a]
                idx = rng.integers(0, len(ts), len(ts))
                pa_boot[a][b] = stat([ts[i] for i in idx], cond_tracks)

        out["mode_level"][f"P{P}"] = {
            m: {"point": round(points[m], 1), "ci95": pct_ci(boot[m])} for m in modes}
        if has_gap:
            out["mode_level"][f"P{P}"]["gap_cond_vs_ftbase"] = {
                "point": round(points["conditioned"] - points["finetuned_baseline"], 1),
                "ci95": pct_ci(boot["gap"]),
            }
        n_samples = Counter(s["artist"] for s in data[anchor][3]["per_sample"] if s["windows"])
        out["per_artist"][f"P{P}"] = {
            a: {
                "point": round(pa_points[a], 1),
                "ci95": pct_ci(pa_boot[a]),
                "n_tracks": len(pa_strata[a]),
                "n_samples": n_samples[a],
            }
            for a in sorted(pa_points, key=lambda a: -pa_points[a])
        }
        if P == 256:
            macro = float(np.mean(list(pa_points.values())))
            out["macro_micro"]["P256"] = {
                "micro": round(points[anchor], 1),
                "macro": round(macro, 1),
            }

        # Hyman error sink (conditioned job), bootstrap over tracks-with-errors
        if P == 256 and anchor == "conditioned":
            d = data[anchor][3]
            hyman = d["config"]["artists"].index("Dick Hyman")
            sink = {}
            for target in ("Art Tatum", "Erroll Garner"):
                per_track = defaultdict(lambda: [0, 0])  # to_hyman, wrong
                for s in d["per_sample"]:
                    if s["artist"] != target:
                        continue
                    for w in s["windows"]:
                        if not w["correct_prompt"]:
                            per_track[s["track_id"]][1] += 1
                            per_track[s["track_id"]][0] += w["pred"] == hyman
                ts = sorted(t for t, (_h, wr) in per_track.items() if wr)
                point = 100 * sum(per_track[t][0] for t in ts) / sum(per_track[t][1] for t in ts)
                bs = np.empty(B)
                for b in range(B):
                    idx = rng.integers(0, len(ts), len(ts))
                    h = sum(per_track[ts[i]][0] for i in idx)
                    w = sum(per_track[ts[i]][1] for i in idx)
                    bs[b] = 100 * h / w if w else np.nan
                sink[target] = {
                    "point": round(point, 1),
                    "to_hyman": sum(per_track[t][0] for t in ts),
                    "wrong": sum(per_track[t][1] for t in ts),
                    "ci95": pct_ci(bs[~np.isnan(bs)]),
                }
            out["hyman_sink"] = sink
            curve = d["agreement_curve"]
            out["survivorship"] = {"n_first": curve[0]["n_windows"], "n_last": curve[-1]["n_windows"]}


def classifier_block(evals, rng: np.random.Generator, B: int, out: dict) -> None:
    for name, path in evals.items():
        d = json.loads(Path(path).read_text())
        per_track = defaultdict(list)
        artist_of = {}
        for s in d["per_sample"]:
            per_track[s["track_id"]].append(s)
            artist_of[s["track_id"]] = s["true_artist"]
        tids = sorted(per_track)

        def chunk_acc(ts):
            c = sum(x["correct"] for t in ts for x in per_track[t])
            n = sum(len(per_track[t]) for t in ts)
            return 100 * c / n

        artists = sorted({x["true_artist"] for xs in per_track.values() for x in xs})
        has_logits = all("logits" in x for xs in per_track.values() for x in xs)

        def song_correct(t) -> bool:
            # majority vote; exact ties go to the tied artist with the highest
            # mean logit (the paper's convention)
            votes = Counter(x["predicted_artist"] for x in per_track[t])
            top = max(votes.values())
            tied = sorted(a for a, v in votes.items() if v == top)
            if len(tied) > 1 and has_logits:
                idx = {a: i for i, a in enumerate(d.get("artists") or artists)}
                mean = np.mean([x["logits"] for x in per_track[t]], axis=0)
                tied = [max(tied, key=lambda a: mean[idx[a]])]
            return tied[0] == artist_of[t]

        def song_acc(ts):
            return 100 * np.mean([song_correct(t) for t in ts])

        chunk_pt, song_pt = chunk_acc(tids), song_acc(tids)

        by_a = defaultdict(list)
        for t in tids:
            by_a[artist_of[t]].append(t)
        strata = sorted(by_a.items())
        chunk_b = np.empty(B)
        song_b = np.empty(B)
        for b in range(B):
            chosen = []
            for _a, ts in strata:
                idx = rng.integers(0, len(ts), len(ts))
                chosen.extend(ts[i] for i in idx)
            chunk_b[b] = chunk_acc(chosen)
            song_b[b] = song_acc(chosen)

        k = sum(song_correct(t) for t in tids)
        out["classifier"][name] = {
            "chunk": {"point": round(chunk_pt, 1), "ci95": pct_ci(chunk_b)},
            "song": {
                "point": round(song_pt, 1),
                "correct_tracks": f"{k}/{len(tids)}",
                "ci95_bootstrap": pct_ci(song_b),
                "ci95_clopper_pearson": clopper_pearson(k, len(tids)),
            },
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agreement", action="append", default=[], metavar="MODE:P=PATH",
                    help="an agreement_eval.py output, e.g. conditioned:256=results/a.json")
    ap.add_argument("--real-clf-eval", type=Path, default=None)
    ap.add_argument("--synth-clf-eval", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/bootstrap_cis.json"))
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260711)
    args = ap.parse_args()

    runs = {}
    for spec in args.agreement:
        key, path = spec.split("=", 1)
        mode, p = key.split(":")
        runs[(mode, int(p))] = Path(path)
    evals = {k: v for k, v in (("real", args.real_clf_eval), ("synth", args.synth_clf_eval)) if v}

    rng = np.random.default_rng(args.seed)
    out = {"seed": args.seed, "resamples": args.resamples,
           "inputs": {**{f"{m}:{p}": str(v) for (m, p), v in runs.items()},
                      **{k: str(v) for k, v in evals.items()}},
           "mode_level": {}, "per_artist": {}, "macro_micro": {}, "classifier": {}}
    if runs:
        continuation_block(runs, rng, args.resamples, out)
    if evals:
        classifier_block(evals, rng, args.resamples, out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}\n")
    for P, modes in out["mode_level"].items():
        print(f"--- {P} ---")
        for mode, v in modes.items():
            print(f"  {mode:22s} {v['point']:6.1f}  CI={v['ci95']}")
    for P, pa in out["per_artist"].items():
        print(f"\nper-artist {P}:")
        for a, v in pa.items():
            print(f"  {a:18s} {v['point']:6.1f}  CI={v['ci95']}  n_tracks={v['n_tracks']}")
    if "hyman_sink" in out:
        print(f"\nhyman_sink: {json.dumps(out['hyman_sink'])}")
    if out["classifier"]:
        print(f"\nclassifier: {json.dumps(out['classifier'], indent=1)}")


if __name__ == "__main__":
    main()
