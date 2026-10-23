"""Evaluate the matcher against your labeled pairs and regression cases.

Usage:  python eval.py dataset.json --repeats 3 --verbose
        python eval.py regression_cases.json --repeats 1 --verbose

Each case may contain:
  human_score   your label (omit it for cases that only check behavior; they are left out of the metrics)
  profile       {"country": "...", "location": "...", "needs_sponsorship": "yes|no|unknown"} sent like the app does
  group         cases with the same group (same resume + JD, different profile) must all get the SAME score
  expect        optional checks run on EVERY repeat:
                  eligibility   "ok" | "check" | "blocked"
                  verdict       "Apply" | "Stretch" | "Skip"
                  min_score     number
                  soft_contains list of substrings that must appear among the unscored soft requirements
                  no_constraint_contains  substrings that must NOT appear in any returned constraint label
Needs OPENAI_API_KEY (reads .env like the API does).
"""
import argparse
import itertools
import json
import statistics
import time

from main import MODEL, CandidateProfile, run_analysis


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def pairwise_accuracy(human, app):
    total = correct = 0.0
    for i, j in itertools.combinations(range(len(human)), 2):
        if human[i] == human[j]:
            continue
        total += 1
        if app[i] == app[j]:
            correct += 0.5
        elif (human[i] - human[j]) * (app[i] - app[j]) > 0:
            correct += 1
    return correct / total if total else float("nan")


def run_checks(expect, runs):
    """Return a list of (description, passed). Every repeat must satisfy each check."""
    checks = []
    if "eligibility" in expect:
        got = sorted({r.eligibility for r in runs})
        checks.append((f"eligibility == {expect['eligibility']} (got {got})", got == [expect["eligibility"]]))
    if "verdict" in expect:
        got = sorted({r.verdict for r in runs})
        checks.append((f"verdict == {expect['verdict']} (got {got})", got == [expect["verdict"]]))
    if "min_score" in expect:
        low = min(r.match_score for r in runs)
        checks.append((f"score >= {expect['min_score']} (lowest {low})", low >= expect["min_score"]))
    for needle in expect.get("soft_contains", []):
        ok = all(any(needle.lower() in s.requirement.lower() for s in r.soft_requirements) for r in runs)
        checks.append((f"'{needle}' listed as unscored soft requirement", ok))
    for needle in expect.get("no_constraint_contains", []):
        bad = any(needle.lower() in c.constraint.lower() for r in runs for c in r.hard_constraints)
        checks.append((f"no constraint mentions '{needle}'", not bad))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--verbose", action="store_true", help="print every requirement per pair")
    args = ap.parse_args()

    pairs = json.load(open(args.pairs))
    rows = []
    for p in pairs:
        profile = CandidateProfile(**p["profile"]) if p.get("profile") else None
        runs = [run_analysis(p["resume"], p["job_description"], profile) for _ in range(args.repeats)]
        scores = [r.match_score for r in runs]
        last = runs[-1]
        rows.append(
            {
                "id": p["id"],
                "human": p.get("human_score"),
                "mean": statistics.fmean(scores),
                "noise_sd": statistics.pstdev(scores),
                "scores": scores,
                "group": p.get("group"),
                "verdict": last.verdict,
                "eligibility": last.eligibility,
                "missing": last.missing_skills,
                "partial": last.partial_skills,
                "soft": [s.requirement for s in last.soft_requirements],
                "constraints": [c.model_dump() for c in last.hard_constraints],
                "feedback": last.feedback,
                "breakdown": last.score_breakdown,
                "checks": run_checks(p.get("expect", {}), runs),
            }
        )

    print(f"\nmodel={MODEL} repeats={args.repeats}\n")
    print(f"{'pair':<28}{'human':>7}{'app':>7}{'diff':>7}{'noise_sd':>10}  {'verdict':<8} eligibility")
    for r in rows:
        human = "-" if r["human"] is None else r["human"]
        diff = "" if r["human"] is None else f"{r['mean'] - r['human']:.1f}"
        print(
            f"{r['id']:<28}{human:>7}{r['mean']:>7.1f}{diff:>7}{r['noise_sd']:>10.1f}  "
            f"{r['verdict']:<8} {r['eligibility']}"
        )

    if args.verbose:
        for r in rows:
            b = r["breakdown"]
            print(f"\n--- {r['id']} (last run) raw={b['raw_score']} caps={b['caps_applied']} profile_used={b['profile_used']}")
            for q in b.get("requirements", []):
                core = "*" if q["core"] else " "
                print(f"  {core}[{q['tier']:<4}] {q['status']:<8} {q['requirement']}  <- {q['evidence'][:70]!r}")
            for s in r["soft"]:
                print(f"   [soft] not scored: {s}")
            for u in b.get("unverified_evidence", []):
                print(f"  UNVERIFIED ({u['claimed']}): {u['requirement']} <- {u['evidence'][:70]!r}")
            for c in r["constraints"]:
                print(f"  CONSTRAINT {c['status']}: {c['constraint']}  ({c['reason']})")
            print(f"  dropped (not in JD): {b['dropped_not_in_jd']}  out of scope: {b['dropped_out_of_scope']}  constraints: {b['dropped_constraints']}")
            print(f"  jd_cached={b['jd_cached']} scoped={b['jd_scoped']} seniority_note={b['seniority_note']!r}")
            print(f"  feedback: {r['feedback']}")

    all_checks = [(r["id"], d, ok) for r in rows for d, ok in r["checks"]]
    groups = {}
    for r in rows:
        if r["group"]:
            groups.setdefault(r["group"], []).append(r)
    for g, members in groups.items():
        scores = sorted({s for m in members for s in m["scores"]})
        reqs = {tuple(q["requirement"] for q in m["breakdown"]["requirements"]) for m in members}
        all_checks.append((g, f"same score across profile variants (got {scores})", len(scores) == 1))
        all_checks.append((g, "same requirement list across profile variants", len(reqs) == 1))
    if all_checks:
        print("\nChecks")
        for rid, desc, ok in all_checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {rid}: {desc}")
        failed = sum(1 for _, _, ok in all_checks if not ok)
        print(f"  {len(all_checks) - failed}/{len(all_checks)} passed")

    labeled = [r for r in rows if r["human"] is not None]
    if len(labeled) >= 2:
        human = [r["human"] for r in labeled]
        app = [r["mean"] for r in labeled]
        n = len(labeled)
        print("\nMetrics")
        print(f"  pairs                  : {n}" + ("  (too few to trust; aim for 20+)" if n < 20 else ""))
        print(f"  MAE                    : {statistics.fmean(abs(a - h) for a, h in zip(app, human)):.1f}")
        print(f"  Spearman               : {pearson(ranks(human), ranks(app)):.2f}")
        print(f"  score SD  app / human  : {statistics.pstdev(app):.1f} / {statistics.pstdev(human):.1f}")
        print(f"  mean run-to-run SD     : {statistics.fmean(r['noise_sd'] for r in labeled):.1f}")
        print(f"  pairwise ranking acc.  : {pairwise_accuracy(human, app):.2f}")

    fname = f"eval_results_{int(time.time())}.json"
    json.dump({"model": MODEL, "repeats": args.repeats, "rows": rows}, open(fname, "w"), indent=2)
    print(f"\nSaved details to {fname}")


if __name__ == "__main__":
    main()
