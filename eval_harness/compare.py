"""Paired comparison of two already-scored variants -- B against A.

    python -m eval_harness.compare --corpus voicebank16k --a dtln-f32 --b dtln-drq
    python -m eval_harness.compare --corpus dns --a wiener-mcra --b dtln-f32

enhance.py compares every variant with the noisy input. This reads the CSVs
it wrote (<root>/enhanced/<tag>/<corpus>.csv), pairs A and B by utterance id,
and runs stats.compare() per metric -- the test fixed by the metric, effect
size and CI beside the p-value, TOST against the locked margins -- with Holm
over the metrics of this one pair. It is the shape every algorithm-vs-
algorithm or compression-vs-baseline claim takes (Metric Stack Decision 6:
headline pooled per-corpus tests, one per (pair, metric), Holm over the
family). Rows are never pooled across corpora: one corpus per invocation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import manifest, stats

METRICS = ("pesq_mos_lqo", "pesq_raw_mos", "stoi", "estoi", "si_sdr")


def load_scores(root: Path, tag: str, corpus: str) -> pd.DataFrame:
    p = root / "enhanced" / tag / f"{corpus}.csv"
    if not p.exists():
        raise SystemExit(f"{p} missing -- run eval_harness.enhance for '{tag}' on {corpus} first")
    df = pd.read_csv(p)
    if (df.corpus != corpus).any():
        raise SystemExit(f"{p} carries rows from another corpus; never pooled")
    return df[df.ok].set_index("id")


# The one cross-corpus pairing that is legitimate: the same 824 utterances scored
# at 16 kHz wideband either from the 16 kHz mirror (classical, DTLN) or from the
# 48 kHz set through the locked downsample (RNNoise). The two filter-matched
# references agree to 0.0007 dB SI-SDR and 1/50th of the PESQ margin
# (refcheck.py, 2026-09-15), so a paired comparison across them is sound.
# Nothing else may be paired across corpora.
SAME_UTTERANCES = {frozenset({"voicebank16k", "voicebank48k"})}


def compare(root: Path, corpus: str, a: str, b: str, corpus_b: str | None = None) -> list[stats.Comparison]:
    corpus_b = corpus_b or corpus
    if corpus_b != corpus and frozenset({corpus, corpus_b}) not in SAME_UTTERANCES:
        raise SystemExit(f"{corpus} and {corpus_b} are different utterances; never paired")
    A, B = load_scores(root, a, corpus), load_scores(root, b, corpus_b)
    ids = A.index.intersection(B.index)
    if len(ids) < 20:
        raise SystemExit(f"only {len(ids)} paired utterances")
    cols = [c for c in METRICS if c in A.columns and c in B.columns]
    return [stats.compare(A.loc[ids, c].to_numpy(), B.loc[ids, c].to_numpy(), c) for c in cols]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--a", required=True, help="baseline variant tag, e.g. dtln-f32")
    ap.add_argument("--b", required=True, help="variant under test, e.g. dtln-drq")
    ap.add_argument("--corpus-b", default=None,
                    help="B's corpus if different; only voicebank16k<->voicebank48k is accepted")
    ap.add_argument("--root", type=Path, default=manifest.DEFAULT_ROOT)
    args = ap.parse_args(argv)
    cs = compare(args.root, args.corpus, args.a, args.b, args.corpus_b)
    cb = f" [{args.corpus_b}]" if args.corpus_b and args.corpus_b != args.corpus else ""
    print(f"=== {args.corpus}: {args.b}{cb} (B) vs {args.a} (A), paired by utterance, n = {cs[0].n}; "
          f"Holm over {len(cs)} metrics of this pair")
    print(stats.render(cs))
    adj = stats.holm([c.p_diff for c in cs])
    print("\nHolm-adjusted p(diff):")
    for c, p in zip(cs, adj):
        print(f"  {c.metric:<14} raw {c.p_diff:9.3g}  adj {p:9.3g}  {'significant' if p < stats.ALPHA else 'n.s.'}")


if __name__ == "__main__":
    main()
