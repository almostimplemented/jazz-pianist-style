#!/usr/bin/env python3
"""Bootstrap confidence intervals for every headline number in the paper.

Method (one method everywhere, decided 2026-07-11): stratified track-level
percentile cluster bootstrap. Windows overlap ~8x (stride 128, window 1024)
and multiple continuations share a source track, so the track is the largest
exchangeable unit; window counts are not binomial trials. Tracks are resampled
with replacement within artist strata and every statistic is recomputed.
The single carve-out: song-level classification accuracy is a genuine
one-binary-outcome-per-track binomial over 80 tracks, so it additionally gets
an exact Clopper-Pearson interval (the bootstrap degenerates at the real
classifier's 100% boundary).

Inputs are the cached eval artifacts only (no S3, no GPU):
  results/agreement_*.json                       (per_sample windows)
  results/{real,synth}_clf_eval.json

Output: results/bootstrap/bootstrap_cis.json + a printed report with
LaTeX-ready strings. Point estimates are asserted to reproduce the published
numbers before any interval is reported.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "results"

# Mode/P -> job map, mirrors scripts/paper/fig_agreement_curves.py JOBS
# (source of truth: docs/paper_manifest.yaml agreement_curves_grid).
JOBS = {
    "conditioned": {
        128: "cleaned-v2-final-cont-eval-v2-1774602262",
        256: "cleaned-v2-final-cont-eval-v2-1774602280",
        512: "cleaned-v2-final-cont-eval-v2-1774602299",
    },
    "ablation": {
        128: "cleaned-v2-final-cont-eval-v2-1774602271",
        256: "cleaned-v2-final-cont-eval-v2-1774602289",
        512: "cleaned-v2-final-cont-eval-v2-1774602308",
    },
    "pretrained_baseline": {
        128: "cleaned-v2-final-cont-eval-v2-1774602267",
        256: "cleaned-v2-final-cont-eval-v2-1774602285",
        512: "cleaned-v2-final-cont-eval-v2-1774602303",
    },
    "finetuned_baseline": {
        128: "cleaned-v2-finetuned-baseline-cont-eval-v2-1774602317",
        256: "cleaned-v2-finetuned-baseline-cont-eval-v2-1774602321",
        512: "cleaned-v2-finetuned-baseline-cont-eval-v2-1774602326",
    },
}
PROMPTS = (128, 256, 512)


def load_job(job: str):
    """Per-track (correct, present) counts on the window-start position grid."""
    d = json.loads((CACHE / "continuation_eval_v2" / f"{job}.json").read_text())
    positions = sorted({e["position"] for e in d["metrics"]["prompt_accuracy_curve"]})
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
        for w in s["window_results"]:
            i = pos_idx[w["start"]]
            n[i] += 1
            c[i] += bool(w["correct_prompt"])
    published = 100 * np.mean([e["accuracy"] for e in d["metrics"]["prompt_accuracy_curve"]])
    return tracks, artist_of, published, d


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


def continuation_block(rng: np.random.Generator, B: int, out: dict) -> None:
    for P in PROMPTS:
        data = {m: load_job(j) for m, j in ((m, JOBS[m][P]) for m in JOBS)}
        ref = set(data["conditioned"][0])
        for m, (trk, *_rest) in data.items():
            assert set(trk) == ref, f"track set mismatch: {m} P={P}"
        artist_of = data["conditioned"][1]
        by_artist = {a: sorted(t for t, aa in artist_of.items() if aa == a) for a in sorted(set(artist_of.values()))}
        strata = [(a, ts) for a, ts in by_artist.items()]

        # point estimates must reproduce the published curves before we go on
        points = {}
        for m in JOBS:
            pt = stat(sorted(ref), data[m][0])
            assert abs(pt - data[m][2]) < 0.01, f"{m} P={P}: {pt:.2f} != published {data[m][2]:.2f}"
            points[m] = pt

        # per-artist frame: tracks with >=1 classified window (conditioned job)
        cond_tracks = data["conditioned"][0]
        pa_strata = {
            a: [t for t in ts if cond_tracks[t][1].sum() > 0] for a, ts in by_artist.items()
        }
        pa_points = {a: stat(ts, cond_tracks) for a, ts in pa_strata.items() if ts}

        boot = {m: np.empty(B) for m in JOBS}
        boot["gap"] = np.empty(B)
        pa_boot = {a: np.empty(B) for a in pa_points}
        for b in range(B):
            chosen = []
            for a, ts in strata:
                idx = rng.integers(0, len(ts), len(ts))
                chosen.extend(ts[i] for i in idx)
            for m in JOBS:
                boot[m][b] = stat(chosen, data[m][0])
            boot["gap"][b] = boot["conditioned"][b] - boot["finetuned_baseline"][b]
            for a in pa_points:
                ts = pa_strata[a]
                idx = rng.integers(0, len(ts), len(ts))
                pa_boot[a][b] = stat([ts[i] for i in idx], cond_tracks)

        out["mode_level"][f"P{P}"] = {
            **{m: {"point": round(points[m], 1), "ci95": pct_ci(boot[m])} for m in JOBS},
            "gap_cond_vs_ftbase": {
                "point": round(points["conditioned"] - points["finetuned_baseline"], 1),
                "ci95": pct_ci(boot["gap"]),
            },
        }
        n_samples = Counter(
            s["artist"]
            for s in data["conditioned"][3]["per_sample"]
            if s["window_results"]
        )
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
                "micro": round(points["conditioned"], 1),
                "macro": round(macro, 1),
            }

        # Hyman error sink (conditioned job), bootstrap over tracks-with-errors
        if P == 256:
            d = data["conditioned"][3]
            hyman = d["artist_order"].index("Dick Hyman")
            sink = {}
            for target in ("Art Tatum", "Erroll Garner"):
                per_track = defaultdict(lambda: [0, 0])  # to_hyman, wrong
                for s in d["per_sample"]:
                    if s["artist"] != target:
                        continue
                    for w in s["window_results"]:
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
            curve = d["metrics"]["prompt_accuracy_curve"]
            out["survivorship"] = {"n_first": curve[0]["total"], "n_last": curve[-1]["total"]}


def classifier_block(rng: np.random.Generator, B: int, out: dict) -> None:
    for name, fname, expect in (
        ("real", "real_clf_eval.json", (95.8, 100.0)),
        ("synth", "synth_clf_eval.json", (87.1, 96.2)),
    ):
        d = json.loads((CACHE / "classifier_eval" / fname).read_text())
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

        def song_correct(t) -> bool:
            votes = Counter(x["predicted_artist"] for x in per_track[t])
            top = max(votes.values())
            winner = sorted(a for a, v in votes.items() if v == top)[0]
            return winner == artist_of[t]

        def song_acc(ts):
            return 100 * np.mean([song_correct(t) for t in ts])

        chunk_pt, song_pt = chunk_acc(tids), song_acc(tids)
        assert abs(chunk_pt - expect[0]) < 0.15, f"{name} chunk {chunk_pt:.2f} != {expect[0]}"
        assert abs(song_pt - expect[1]) < 0.15, f"{name} song {song_pt:.2f} != {expect[1]}"

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


def latex_strings(out: dict) -> dict:
    m = out["mode_level"]["P256"]
    mm = out["macro_micro"]["P256"]
    cl = out["classifier"]

    def b(d):
        return f"$[{d['ci95'][0]:.0f},{d['ci95'][1]:.0f}]$"

    return {
        "table2_dagger_cis": (
            f"rows top to bottom: {b(m['pretrained_baseline'])}, {b(m['finetuned_baseline'])}, "
            f"{b(m['conditioned'])}; conditioned-vs-fine-tuned gap "
            f"{m['gap_cond_vs_ftbase']['point']:.0f} points, CI {b(m['gap_cond_vs_ftbase'])}"
        ),
        "macro_sentence": f"macro {mm['macro']:.0f}\\% vs micro {mm['micro']:.0f}\\% at P=256",
        "table4_caption": (
            f"95\\% CIs: chunk ${cl['real']['chunk']['point']}\\,{b(cl['real']['chunk'])[1:-1]}$ vs.\\ "
            f"${cl['synth']['chunk']['point']}\\,{b(cl['synth']['chunk'])[1:-1]}$ (track-level cluster "
            f"bootstrap); song ${cl['real']['song']['point']:.0f}\\,[{cl['real']['song']['ci95_clopper_pearson'][0]},"
            f"{cl['real']['song']['ci95_clopper_pearson'][1]}]$ vs.\\ ${cl['synth']['song']['point']}\\,"
            f"[{cl['synth']['song']['ci95_clopper_pearson'][0]},{cl['synth']['song']['ci95_clopper_pearson'][1]}]$ "
            f"(exact binomial over 80 tracks)."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260711)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = {
        "seed": args.seed,
        "resamples": args.resamples,
        "jobs": JOBS,
        "mode_level": {},
        "per_artist": {},
        "macro_micro": {},
        "classifier": {},
    }
    continuation_block(rng, args.resamples, out)
    classifier_block(rng, args.resamples, out)
    out["latex_strings"] = latex_strings(out)

    dst = CACHE / "bootstrap" / "bootstrap_cis.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}\n")

    for P in PROMPTS:
        print(f"--- P={P} ---")
        for mode, v in out["mode_level"][f"P{P}"].items():
            print(f"  {mode:22s} {v['point']:6.1f}  CI={v['ci95']}")
    print(f"\nmacro/micro P256: {out['macro_micro']['P256']}")
    print("\nper-artist P256:")
    for a, v in out["per_artist"]["P256"].items():
        print(f"  {a:18s} {v['point']:6.1f}  CI={v['ci95']}  n_tracks={v['n_tracks']}")
    print(f"\nhyman_sink: {json.dumps(out['hyman_sink'])}")
    print(f"survivorship: {out['survivorship']}")
    print(f"\nclassifier: {json.dumps(out['classifier'], indent=1)}")
    print("\nLaTeX strings:")
    for k, v in out["latex_strings"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
