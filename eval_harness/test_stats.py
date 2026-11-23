"""
test_stats.py -- tests for the statistics layer (task 6).

WHY THIS FILE EXISTS, in the terms this project uses elsewhere: a statistics
layer is a measurement instrument, and every trap in CLAUDE.md's "Traps and
Rules" is some version of an instrument that reads correctly and does nothing.
`stats.py` is worse than most in that respect, because its failure mode is
SILENT AND PUBLISHABLE. A TOST with one of its two one-sided tests written
backwards declares equivalence for a difference of any size; nothing crashes,
nothing looks wrong, and the number goes into the paper.

So the organising rule here is trap 15 -- "a mock that cannot say no proves
nothing". Every test below is paired with a specific defect it would catch, and
the suite is mutation-tested: each mutation listed in MUTATIONS (the list lives in
eval_harness/mutate_stats.py; the record is in eval_harness/RESULTS.md)
must make at least one test fail. A test that passes against a broken stats.py
is decoration.

WHY unittest AND NOT pytest: requirements-eval.txt is the frozen 34-package
environment for the task-26 reproducibility artifact and pytest is not in it.
Adding a test-only dependency to the artifact's environment to test the
artifact is the wrong trade, and unittest is in the standard library.

Deterministic throughout: every sample comes from a fixed-seed PRNG, so a
number that moves is the code moving and not the data -- the same rule the
firmware bench harness applies to its operands.

Run:
    .venv/bin/python -m eval_harness.test_stats
    .venv/bin/python -m unittest discover -s eval_harness -t . -p 'test_*.py'
"""
from __future__ import annotations

import unittest
import warnings

import numpy as np

from . import stats as S

SEED = 20260913
_CHECKS = 0


class StatsTest(unittest.TestCase):
    """Counts assertions, so the suite reports a check count the way
    test_abf.c reports its 193."""

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name.startswith("assert") and callable(attr):
            def counting(*a, **kw):
                global _CHECKS
                _CHECKS += 1
                return attr(*a, **kw)
            return counting
        return attr

    def rng(self, offset: int = 0):
        return np.random.default_rng(SEED + offset)


# ---------------------------------------------------------------------------
# 1. The locked constants. Defect guarded: a margin "tuned" after seeing data,
#    which is the one thing CLAUDE.md says most emphatically not to do. A
#    margin is only meaningful because it was fixed in advance, so it belongs
#    in a test and not merely in a comment.
# ---------------------------------------------------------------------------
class TestLockedConstants(StatsTest):

    def test_margins_are_the_values_locked_on_2026_09_12(self):
        self.assertEqual(S.MARGIN["pesq_mos_lqo"], 0.10)
        self.assertEqual(S.MARGIN["pesq_raw_mos"], 0.10)
        self.assertEqual(S.MARGIN["stoi"], 0.01)
        self.assertEqual(S.MARGIN["estoi"], 0.01)
        # Just inside TinyLSTMs' own accepted 0.55 dB degradation.
        self.assertEqual(S.MARGIN["si_sdr"], 0.50)
        self.assertLess(S.MARGIN["si_sdr"], 0.55)

    def test_test_choice_is_fixed_per_metric(self):
        self.assertEqual(S.TEST["pesq_mos_lqo"], "wilcoxon")
        self.assertEqual(S.TEST["pesq_raw_mos"], "wilcoxon")
        self.assertEqual(S.TEST["stoi"], "wilcoxon")
        self.assertEqual(S.TEST["estoi"], "wilcoxon")
        self.assertEqual(S.TEST["si_sdr"], "t")

    def test_every_metric_has_both_a_test_and_a_margin(self):
        # Defect guarded: a metric added to one dict only. compare() looks up
        # TEST first and raises a helpful error, then indexes MARGIN and would
        # raise a bare KeyError halfway through a matrix run.
        self.assertEqual(set(S.TEST), set(S.MARGIN))

    def test_alpha(self):
        self.assertEqual(S.ALPHA, 0.05)


# ---------------------------------------------------------------------------
# 2. What compare() must REFUSE. Every one of these is a way a wrong number
#    reaches a table without anything looking wrong.
# ---------------------------------------------------------------------------
class TestRefusals(StatsTest):

    def test_unknown_metric_raises_rather_than_defaulting(self):
        r = self.rng()
        a, b = r.normal(0, 1, 40), r.normal(0, 1, 40)
        with self.assertRaises(ValueError):
            S.compare(a, b, "pesq")           # plausible but not a locked name
        with self.assertRaises(ValueError):
            S.compare(a, b, "sdr")

    def test_there_is_no_way_to_override_the_test(self):
        # The test is chosen by the metric, never by the caller. If a `test=`
        # argument ever appears, choosing it after seeing the p-value becomes
        # possible and this whole layer stops being trustworthy.
        import inspect
        params = set(inspect.signature(S.compare).parameters)
        self.assertEqual(params, {"a", "b", "metric", "margin", "alpha"})

    def test_below_twenty_pairs_is_refused(self):
        r = self.rng(1)
        a = r.normal(0, 1, 19)
        with self.assertRaises(ValueError):
            S.compare(a, a + 0.01, "si_sdr")
        # And exactly 20 is allowed, so the boundary is the documented one.
        a20 = r.normal(0, 1, 20)
        S.compare(a20, a20 + r.normal(0, 0.1, 20), "si_sdr")

    def test_n_is_counted_after_dropping_non_finite_values(self):
        # Defect guarded: 25 utterances of which 10 failed to score reads as
        # n = 25 and clears the N >= 20 rule on paper while carrying 15 pairs.
        r = self.rng(2)
        a = r.normal(2.0, 0.3, 25)
        b = a + r.normal(0.01, 0.05, 25)
        b[:10] = np.nan
        with self.assertRaises(ValueError) as cm:
            S.compare(a, b, "pesq_mos_lqo")
        self.assertIn("15", str(cm.exception))

    def test_unpaired_shapes_are_refused(self):
        # The message matters, not just the exception type: with the explicit
        # check removed, numpy's broadcast failure is ALSO a ValueError, so
        # assertRaises alone passes for the wrong reason and proves nothing.
        r = self.rng(3)
        with self.assertRaises(ValueError) as cm:
            S.compare(r.normal(0, 1, 40), r.normal(0, 1, 41), "si_sdr")
        self.assertIn("unpaired", str(cm.exception))

    def test_unknown_tost_kind_is_refused(self):
        with self.assertRaises(ValueError):
            S.tost(np.zeros(30) + 0.01, 0.1, "ttest")


# ---------------------------------------------------------------------------
# 3. THE CENTRAL TEST: can TOST say no? A TOST that always says "equivalent"
#    passes every smoke test anyone would naturally write.
# ---------------------------------------------------------------------------
class TestTostCanSayNo(StatsTest):

    def test_a_large_difference_is_not_equivalent(self):
        r = self.rng(10)
        a = r.normal(2.0, 0.4, 200)
        b = a + 0.5                       # 5x the PESQ margin
        c = S.compare(a, b, "pesq_mos_lqo")
        self.assertFalse(c.equivalent)
        self.assertTrue(c.significant)
        self.assertIn("DIFFERENT", c.verdict)

    def test_a_large_NEGATIVE_difference_is_not_equivalent_either(self):
        # Defect guarded: the classic TOST bug, one of the two one-sided tests
        # written with the wrong sign. It shows up on one side only, so a
        # suite that tests improvement but never degradation misses it -- and
        # degradation is the direction every compression claim runs in.
        r = self.rng(11)
        a = r.normal(2.0, 0.4, 200)
        c = S.compare(a, a - 0.5, "pesq_mos_lqo")
        self.assertFalse(c.equivalent)
        self.assertIn("DIFFERENT", c.verdict)

    def test_tost_is_symmetric_under_sign_flip(self):
        r = self.rng(12)
        d = r.normal(0.3, 0.1, 100)
        for kind in ("t", "wilcoxon"):
            self.assertAlmostEqual(S.tost(d, 0.1, kind),
                                   S.tost(-d, 0.1, kind), places=12)

    def test_a_small_difference_IS_equivalent(self):
        r = self.rng(13)
        a = r.normal(2.0, 0.4, 400)
        c = S.compare(a, a + r.normal(0.01, 0.03, 400), "pesq_mos_lqo")
        self.assertTrue(c.equivalent)

    def test_equivalence_tracks_the_margin_and_not_the_sample_size(self):
        # A difference of 0.30 must stay non-equivalent no matter how much data
        # is thrown at it -- that is exactly what distinguishes TOST from
        # reading a non-significant difference test as a null result.
        r = self.rng(14)
        for n in (50, 400, 2000):
            a = r.normal(2.0, 0.4, n)
            self.assertFalse(S.compare(a, a + 0.30, "pesq_mos_lqo").equivalent)

    def test_si_sdr_margin_boundary(self):
        # Straddle 0.5 dB: 0.2 dB equivalent, 0.9 dB not, on the same spread.
        r = self.rng(15)
        a = r.normal(8.0, 2.0, 300)
        self.assertTrue(S.compare(a, a + r.normal(0.2, 0.3, 300), "si_sdr").equivalent)
        self.assertFalse(S.compare(a, a + r.normal(0.9, 0.3, 300), "si_sdr").equivalent)


# ---------------------------------------------------------------------------
# 4. The definitional identity: equivalence is declared exactly when the
#    (1 - 2*alpha) interval lies inside +/- margin. If the CI printed in a
#    table and the verdict beside it disagree, one of them is wrong and a
#    reader has no way to tell which.
# ---------------------------------------------------------------------------
class TestTostCiIdentity(StatsTest):

    def test_ci_level_is_one_minus_two_alpha(self):
        r = self.rng(20)
        a = r.normal(8.0, 2.0, 60)
        c = S.compare(a, a + r.normal(0.1, 0.4, 60), "si_sdr")
        self.assertAlmostEqual(c.ci_level, 0.90, places=12)

    def test_t_branch_verdict_matches_its_own_interval(self):
        r = self.rng(21)
        crossings = 0
        prev = None
        for shift in np.linspace(0.0, 1.2, 41):
            a = r.normal(0, 1.0, 60)
            c = S.compare(a, a + r.normal(shift, 0.9, 60), "si_sdr")
            inside = (c.ci_lo > -c.margin) and (c.ci_hi < c.margin)
            self.assertEqual(inside, c.equivalent)
            if prev is not None and prev != c.equivalent:
                crossings += 1
            prev = c.equivalent
        # The sweep must actually cross the boundary, or it proves nothing.
        self.assertGreaterEqual(crossings, 1)

    def test_wilcoxon_branch_verdict_matches_its_own_interval(self):
        r = self.rng(22)
        seen = set()
        for shift in np.linspace(0.0, 0.25, 41):
            a = r.normal(2.0, 0.4, 100)
            c = S.compare(a, a + r.normal(shift, 0.15, 100), "pesq_mos_lqo")
            inside = (c.ci_lo > -c.margin) and (c.ci_hi < c.margin)
            self.assertEqual(inside, c.equivalent)
            seen.add(c.equivalent)
        self.assertEqual(seen, {True, False})


# ---------------------------------------------------------------------------
# 5. The four-way verdict. The row that matters is the third: not significant
#    AND not equivalent is UNDERPOWERED, not a no-difference result, and it is
#    the one routinely misreported in the literature.
# ---------------------------------------------------------------------------
class TestVerdicts(StatsTest):

    def test_mapping(self):
        self.assertIn("equivalent", S.verdict(True, True))
        self.assertIn("DIFFERENT", S.verdict(True, False))
        self.assertIn("EQUIVALENT", S.verdict(False, True))
        self.assertIn("INCONCLUSIVE", S.verdict(False, False))

    def test_inconclusive_explicitly_disclaims_a_null_result(self):
        v = S.verdict(False, False)
        self.assertIn("NOT", v)
        self.assertIn("underpowered", v.lower())

    def test_underpowered_data_really_produces_inconclusive(self):
        # n = 20 with a spread far wider than the margin: neither test can
        # reject. A suite that only checks verdict() with hand-passed booleans
        # never proves compare() can reach this state.
        r = self.rng(30)
        a = r.normal(2.0, 0.8, 20)
        c = S.compare(a, a + r.normal(0.0, 0.5, 20), "pesq_mos_lqo")
        self.assertFalse(c.significant)
        self.assertFalse(c.equivalent)
        self.assertIn("INCONCLUSIVE", c.verdict)

    def test_real_but_unimportant_difference_is_reachable(self):
        # Significant AND equivalent: the row that shows why a p-value alone is
        # not a finding. n large, shift far below the margin.
        r = self.rng(31)
        a = r.normal(2.0, 0.4, 800)
        c = S.compare(a, a + r.normal(0.02, 0.05, 800), "pesq_mos_lqo")
        self.assertTrue(c.significant)
        self.assertTrue(c.equivalent)


# ---------------------------------------------------------------------------
# 6. Point estimates and effect sizes.
# ---------------------------------------------------------------------------
class TestPointEstimates(StatsTest):

    def test_hodges_lehmann_against_brute_force(self):
        r = self.rng(40)
        d = r.normal(0.1, 0.5, 37)
        walsh = [(d[i] + d[j]) / 2.0
                 for i in range(len(d)) for j in range(i, len(d))]
        self.assertAlmostEqual(S.hodges_lehmann(d), float(np.median(walsh)),
                               places=12)

    def test_hodges_lehmann_includes_the_self_pairs(self):
        # The diagonal (i == j) is part of the Walsh set. Dropping it -- an easy
        # off-by-one in triu_indices(k=0 vs k=1) -- shifts the estimate.
        d = np.array([0.0, 0.0, 4.0])
        # Walsh averages with diagonal: 0,0,2,0,2,4 -> median 1.0
        self.assertAlmostEqual(S.hodges_lehmann(d), 1.0, places=12)

    def test_hl_is_the_estimator_that_goes_with_the_signed_rank_test(self):
        # On a skewed sample the reported point estimate must be HL, not the
        # mean -- reporting a mean beside a Wilcoxon p-value mixes estimands.
        r = self.rng(41)
        a = r.normal(2.0, 0.3, 200)
        d = r.exponential(0.02, 200)          # strongly right-skewed
        c = S.compare(a, a + d, "pesq_mos_lqo")
        self.assertAlmostEqual(c.diff, S.hodges_lehmann(d), places=9)
        self.assertNotAlmostEqual(c.diff, float(d.mean()), places=4)

    def test_hl_ci_brackets_the_point_estimate(self):
        r = self.rng(42)
        for k in range(8):
            d = r.normal(0.05, 0.2, 40 + 7 * k)
            lo, hi = S._hl_ci(d, 0.90)
            self.assertLessEqual(lo, S.hodges_lehmann(d))
            self.assertLessEqual(S.hodges_lehmann(d), hi)

    def test_hl_ci_covers_the_truth_at_about_its_nominal_rate(self):
        # Defect guarded: a wrong critical value or an off-by-one in the order
        # statistic. Both keep the interval looking sane while its coverage is
        # wrong, which no single-sample assertion detects.
        r = self.rng(43)
        for n in (20, 40):
            hits = sum(int(lo <= 0.05 <= hi)
                       for lo, hi in (S._hl_ci(r.normal(0.05, 0.2, n), 0.90)
                                      for _ in range(600)))
            self.assertAlmostEqual(hits / 600, 0.90, delta=0.05)

    def test_hl_ci_endpoints_are_mirrored_order_statistics(self):
        # The construction takes the (k+1)-th smallest Walsh average and the
        # (k+1)-th largest. An off-by-one on one side only leaves an interval
        # that still looks sane and still covers at roughly the right rate --
        # coverage simulation is far too blunt to see it, so assert the
        # symmetry in RANK directly.
        r = self.rng(46)
        for n in (25, 41, 60):
            d = r.normal(0.05, 0.2, n)
            lo, hi = S._hl_ci(d, 0.90)
            i, j = np.triu_indices(n, k=0)
            w = np.sort((d[i] + d[j]) / 2.0)
            k_lo = int(np.searchsorted(w, lo))
            k_hi = int(np.searchsorted(w, hi))
            self.assertEqual(w[k_lo], lo)
            self.assertEqual(w[k_hi], hi)
            self.assertEqual(k_lo, len(w) - 1 - k_hi)

    def test_rank_biserial_saturates_and_centres(self):
        self.assertAlmostEqual(S.rank_biserial(np.array([0.1, 0.2, 0.3])), 1.0)
        self.assertAlmostEqual(S.rank_biserial(np.array([-0.1, -0.2, -0.3])), -1.0)
        self.assertAlmostEqual(S.rank_biserial(np.array([-0.2, -0.1, 0.1, 0.2])), 0.0)
        self.assertEqual(S.rank_biserial(np.zeros(10)), 0.0)

    def test_rank_biserial_does_not_grow_with_n(self):
        # The property that makes it worth reporting beside a p-value: at
        # N = 824 a p-value is driven by N, an effect size is not.
        r = self.rng(44)
        eff = [abs(S.rank_biserial(r.normal(0.05, 0.4, n))) for n in (50, 400, 3000)]
        self.assertLess(max(eff) - min(eff), 0.20)

    def test_cohen_dz(self):
        r = self.rng(45)
        d = r.normal(0.3, 0.5, 100)
        self.assertAlmostEqual(S.cohen_dz(d),
                               float(d.mean() / d.std(ddof=1)), places=12)

    def test_cohen_dz_on_zero_variance_is_zero_not_infinity(self):
        self.assertEqual(S.cohen_dz(np.full(30, 0.7)), 0.0)


# ---------------------------------------------------------------------------
# 7. The task-15 case: a bit-exact port scored against the float oracle gives
#    differences that are all exactly zero. That must read as EQUIVALENT and
#    must not emit a NaN p-value into the results table or into holm().
# ---------------------------------------------------------------------------
class TestIdenticalSamples(StatsTest):

    def test_identical_samples_are_equivalent_on_both_branches(self):
        r = self.rng(50)
        a = r.normal(8.0, 2.0, 60)
        for metric in ("si_sdr", "pesq_mos_lqo"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = S.compare(a, a.copy(), metric)
            self.assertTrue(c.equivalent, metric)
            self.assertFalse(c.significant, metric)
            self.assertIn("EQUIVALENT", c.verdict)
            self.assertAlmostEqual(c.diff, 0.0, places=12)

    def test_identical_samples_do_not_produce_a_nan_p_value(self):
        # A NaN p_diff compares False against every threshold, so it reads as
        # "not significant" and looks right -- until it reaches holm(), where
        # it sorts unpredictably and silently corrupts the adjustment of every
        # other comparison in the family.
        r = self.rng(51)
        a = r.normal(8.0, 2.0, 60)
        for metric in ("si_sdr", "pesq_mos_lqo"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = S.compare(a, a.copy(), metric)
            self.assertTrue(np.isfinite(c.p_diff), metric)
            self.assertAlmostEqual(c.p_diff, 1.0, places=12)

    def test_a_negligible_difference_is_equivalent(self):
        r = self.rng(52)
        a = r.normal(8.0, 2.0, 60)
        c = S.compare(a, a + r.normal(0, 1e-12, 60), "si_sdr")
        self.assertTrue(c.equivalent)


# ---------------------------------------------------------------------------
# 8. Holm-Bonferroni. Not applied automatically, and the decision of whether
#    headline claims are per-cell or pooled is NOT yet recorded in CLAUDE.md --
#    but the arithmetic still has to be right when it is used.
# ---------------------------------------------------------------------------
class TestHolm(StatsTest):

    def test_hand_computed_values(self):
        # m = 3, sorted 0.01, 0.03, 0.04:
        #   3*0.01 = 0.03 ; 2*0.03 = 0.06 ; 1*0.04 = 0.04 -> held at 0.06
        got = S.holm([0.01, 0.04, 0.03])
        for g, want in zip(got, [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(g, want, places=12)

    def test_equal_p_values_all_take_the_largest_adjustment(self):
        got = S.holm([0.02] * 4)
        for g in got:
            self.assertAlmostEqual(g, 0.08, places=12)

    def test_single_comparison_is_unchanged(self):
        self.assertAlmostEqual(S.holm([0.031])[0], 0.031, places=12)

    def test_monotone_bounded_and_never_smaller_than_the_raw_p(self):
        r = self.rng(60)
        p = list(r.uniform(0, 1, 25))
        adj = S.holm(p)
        for raw, a in zip(p, adj):
            self.assertGreaterEqual(a + 1e-12, raw)
            self.assertLessEqual(a, 1.0)
        order = np.argsort(p)
        run = [adj[i] for i in order]
        self.assertTrue(all(run[i] <= run[i + 1] + 1e-12
                            for i in range(len(run) - 1)))

    def test_holm_is_less_conservative_than_bonferroni_except_at_the_smallest(self):
        p = [0.001, 0.01, 0.02, 0.04]
        adj = S.holm(p)
        self.assertAlmostEqual(adj[0], 4 * 0.001, places=12)
        self.assertLess(adj[-1], 4 * 0.04)


# ---------------------------------------------------------------------------
# 9. describe() and check_normality().
# ---------------------------------------------------------------------------
class TestDescribe(StatsTest):

    def test_ci_matches_the_t_formula(self):
        from scipy import stats as sps
        r = self.rng(70)
        x = r.normal(2.0, 0.5, 120)
        d = S.describe(x, 0.95)
        half = sps.t.ppf(0.975, 119) * x.std(ddof=1) / np.sqrt(120)
        self.assertAlmostEqual(d["ci_lo"], x.mean() - half, places=12)
        self.assertAlmostEqual(d["ci_hi"], x.mean() + half, places=12)
        self.assertEqual(d["n"], 120)

    def test_non_finite_values_are_dropped_and_n_reflects_it(self):
        x = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
        self.assertEqual(S.describe(x)["n"], 3)

    def test_a_wider_level_gives_a_wider_interval(self):
        r = self.rng(71)
        x = r.normal(0, 1, 100)
        a, b = S.describe(x, 0.90), S.describe(x, 0.99)
        self.assertLess(a["ci_hi"] - a["ci_lo"], b["ci_hi"] - b["ci_lo"])


class TestBootstrap(StatsTest):

    def test_the_interval_brackets_the_statistic(self):
        r = self.rng(75)
        x = r.normal(2.0, 0.5, 300)
        lo, hi = S.bootstrap_ci(x, 0.95)
        self.assertLess(lo, x.mean())
        self.assertLess(x.mean(), hi)

    def test_it_agrees_with_the_t_interval_on_normal_data(self):
        # Defect guarded: a bootstrap that is simply wrong. On data where the
        # t-interval IS correct the two must nearly coincide, so this pins the
        # scale; the next test pins what it adds over the t-interval.
        r = self.rng(76)
        x = r.normal(2.0, 0.5, 600)
        blo, bhi = S.bootstrap_ci(x, 0.95)
        d = S.describe(x, 0.95)
        self.assertAlmostEqual(blo, d["ci_lo"], delta=0.01)
        self.assertAlmostEqual(bhi, d["ci_hi"], delta=0.01)

    def test_it_is_asymmetric_on_skewed_data_where_the_t_interval_cannot_be(self):
        # The whole reason it is here. PESQ and STOI are bounded and skewed --
        # which is why their tests are rank-based -- and a t-interval on such a
        # quantity is symmetric by construction whether or not the sampling
        # distribution is.
        r = self.rng(77)
        x = 1.0 + 3.5 * r.beta(2, 5, 824)          # PESQ-shaped
        m = x.mean()
        lo, hi = S.bootstrap_ci(x, 0.95)
        self.assertNotAlmostEqual(m - lo, hi - m, places=4)
        d = S.describe(x, 0.95)
        self.assertAlmostEqual(m - d["ci_lo"], d["ci_hi"] - m, places=12)

    def test_it_is_deterministic(self):
        # A confidence interval that moves between runs of the same data is the
        # resampler talking, not the data -- the operand rule the firmware
        # bench harness applies, in the statistics layer.
        r = self.rng(78)
        x = r.normal(0, 1, 200)
        self.assertEqual(S.bootstrap_ci(x), S.bootstrap_ci(x))

    def test_a_wider_level_gives_a_wider_interval(self):
        r = self.rng(79)
        x = r.normal(0, 1, 300)
        n_lo, n_hi = S.bootstrap_ci(x, 0.90)
        w_lo, w_hi = S.bootstrap_ci(x, 0.99)
        self.assertLess(w_lo, n_lo)
        self.assertLess(n_hi, w_hi)

    def test_describe_adds_the_bootstrap_without_disturbing_the_t_interval(self):
        r = self.rng(80)
        x = r.normal(2.0, 0.5, 250)
        plain, both = S.describe(x), S.describe(x, bootstrap=True)
        for k in plain:
            self.assertEqual(plain[k], both[k])
        self.assertIn("boot_lo", both)
        self.assertEqual(both["boot_resamples"], S.BOOTSTRAP_RESAMPLES)
        self.assertNotIn("boot_lo", plain)


class TestNormality(StatsTest):

    def test_reports_shape_not_just_a_p_value(self):
        r = self.rng(80)
        d = S.check_normality(r.exponential(1.0, 500), "skewed")
        self.assertGreater(d["skew"], 1.0)
        self.assertIn("excess_kurtosis", d)
        self.assertEqual(d["name"], "skewed")

    def test_symmetric_data_reports_near_zero_skew(self):
        r = self.rng(81)
        self.assertAlmostEqual(S.check_normality(r.normal(0, 1, 2000))["skew"],
                               0.0, delta=0.2)

    def test_survives_more_than_the_shapiro_limit(self):
        # scipy only WARNS above 5000, so the truncation is silent policy and
        # has to be visible in the output. The full matrix has runs longer than
        # this.
        r = self.rng(82)
        d = S.check_normality(r.normal(0, 1, 6000))
        self.assertEqual(d["n"], 6000)
        self.assertEqual(d["shapiro_n"], S.SHAPIRO_MAX)
        self.assertTrue(np.isfinite(d["shapiro_p"]))

    def test_the_large_sample_subsample_is_random_and_not_the_head(self):
        # CLAUDE.md trap 14: results arrive in manifest order, so the head is
        # one speaker at the easy end of the SNR range. A head-truncated
        # normality check would be run on the easiest tenth of the matrix and
        # would not say so. Ordered data makes the difference visible.
        r = self.rng(83)
        x = np.sort(r.normal(0, 1, 8000))          # strictly ordered, as a run is
        got = S.check_normality(x)
        head = S.check_normality(x[:S.SHAPIRO_MAX])
        self.assertNotAlmostEqual(got["shapiro_p"], head["shapiro_p"], places=6)
        # Full-sample statistics are unaffected by the subsampling.
        self.assertAlmostEqual(got["skew"], float(__import__("scipy.stats",
                               fromlist=["skew"]).skew(x)), places=12)

    def test_the_subsample_is_deterministic(self):
        # Fixed seed: a number that moves between runs would be the sampler,
        # not the data.
        r = self.rng(84)
        x = r.normal(0, 1, 7000)
        self.assertEqual(S.check_normality(x)["shapiro_p"],
                         S.check_normality(x)["shapiro_p"])


# ---------------------------------------------------------------------------
# 10. Presentation. Cheap, but a table that silently drops the margin or the
#     verdict is how an equivalence claim gets read as a difference claim.
# ---------------------------------------------------------------------------
class TestPresentation(StatsTest):

    def _two(self):
        r = self.rng(90)
        a = r.normal(2.0, 0.4, 100)
        b = r.normal(8.0, 2.0, 100)
        return [S.compare(a, a + r.normal(0.01, 0.05, 100), "pesq_mos_lqo"),
                S.compare(b, b + r.normal(0.2, 0.4, 100), "si_sdr")]

    def test_table_carries_every_field(self):
        df = S.table(self._two())
        self.assertEqual(len(df), 2)
        for col in ("metric", "n", "test", "diff", "ci_lo", "ci_hi",
                    "margin", "p_diff", "p_tost", "effect", "effect_name",
                    "equivalent", "significant", "verdict"):
            self.assertIn(col, df.columns)

    def test_render_names_the_test_and_the_verdict(self):
        out = S.render(self._two())
        self.assertIn("wilcoxon", out)
        self.assertIn("paired t", out)
        self.assertIn("margin", out)
        self.assertIn("EQUIVALENT", out.upper())

    def test_comparison_round_trips_through_as_dict(self):
        c = self._two()[0]
        self.assertEqual(c.as_dict()["metric"], "pesq_mos_lqo")
        self.assertEqual(c.as_dict()["margin"], 0.10)


if __name__ == "__main__":
    import sys
    res = unittest.main(exit=False, verbosity=2).result
    print(f"\n{_CHECKS} checks in {res.testsRun} tests "
          f"({len(res.failures)} failed, {len(res.errors)} errors)")
    sys.exit(0 if res.wasSuccessful() else 1)
