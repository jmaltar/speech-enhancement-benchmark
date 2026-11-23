"""
eval_harness -- host-side evaluation and statistics pipeline (project task 6).

Deliberately a package rather than a script, because task 13 (the training
pipeline) must load audio through the SAME code path. Framing, resampling and
split handling baked into a trained model have to match what the evaluator
measures, and the cheapest way to guarantee that is to have one implementation.

The five metric decisions this implements are LOCKED and dated 2026-09-12 (see
CLAUDE.md, "Metric Stack Decisions"). They were fixed before any eval data
existed, which is the only thing that makes the equivalence claims falsifiable.
Do not revise them after seeing results.
"""
__all__ = ["manifest", "audio", "metrics", "pareto", "refcheck", "separation",
           "stats", "validate"]
