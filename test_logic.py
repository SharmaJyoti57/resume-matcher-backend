"""Offline tests for the deterministic parts of main.py (no OpenAI calls).

Run:  python test_logic.py
The model is replaced by canned JSON, so these check what CODE does with model output:
grounding, section scoping, JD caching, constraint verification and judging, structured profile,
tip validation, soft requirements, feedback and scoring.
"""
import json
import os
import types

os.environ.setdefault("OPENAI_API_KEY", "dummy")

import main  # noqa: E402
from main import CandidateProfile  # noqa: E402

REG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression_cases.json")))
AI_JD = REG[0]["job_description"]          # the real AI-trainer JD used in the app tests
RESUME = (
    "Jyoti Sharma\nMalden MA | Linkedin\n"
    "Mobile App Developer with Java, Kotlin and Swift experience.\n"
    "B.E. in Information Technology Engineering\n"
    "Fitchburg State University Master's in Computer Science\n"
    "Built Flutter apps with Riverpod.\nSoft Skills: Written & Verbal Communication"
)
LOC_QUOTE = "This job is only available to those in the US, Canada, UK, Ireland, Australia, and New Zealand."
PLACES = ["US", "Canada", "UK", "Ireland", "Australia", "New Zealand"]

PAY_JD = (
    "Senior Backend Engineer\nWe build payments software.\n\n"
    "Responsibilities:\nOwn the billing service.\n\n"
    "Requirements:\n5+ years of backend experience.\nExperience with Go.\nA bachelor's degree.\n\n"
    "Preferred:\nKubernetes experience.\n\n"
    "We are unable to sponsor visas for this role."
)
PAY_RESUME = "Backend dev.\nB.Sc. in Computer Science\nBuilt services in Python for 3 years."


def ereq(label, quote, tier="must", kind="skill", core=False):
    return {"requirement": label, "jd_quote": quote, "tier": tier, "kind": kind, "is_core": core}


def econ(label, quote, type_, places=()):
    return {"constraint": label, "jd_quote": quote, "type": type_, "places": list(places)}


def ai_extract(cons=None, extra_reqs=()):
    return {
        "requirements": [
            ereq("Programming language proficiency", "Proficiency in at least one of the following programming languages or frameworks", core=True),
            ereq("Bachelor's degree", "A bachelor's degree (completed or in progress)."),
            ereq("Software developer experience", "Previous experience as a Software Developer, Coder, Software Engineer, or Programmer is preferred.", tier="must"),
            ereq("Fluency in English", "Fluency in English (native or bilingual level).", kind="soft"),
            # The model forgot to flag this as soft; the code-side pattern must catch it.
            ereq("Excellent writing and grammar skills", "Excellent writing and grammar skills."),
            *extra_reqs,
        ],
        "hard_constraints": cons if cons is not None else [econ("Location: US, Canada, UK, Ireland, AU, NZ", LOC_QUOTE, "location_country", PLACES)],
        "seniority_note": "",
    }


AI_EVIDENCE = {
    "Programming language proficiency": ("met", "Java, Kotlin and Swift experience"),
    "Bachelor's degree": ("met", "B.E. in Information Technology Engineering"),
    "Software developer experience": ("met", "Built Flutter apps with Riverpod"),
}


class FakeModel:
    """Routes by system prompt. `judge` maps requirement label -> (status, evidence)."""

    def __init__(self, extract, judge=None, tips=None, seniority="aligned"):
        self.extract, self.judge, self.tips, self.seniority = extract, judge or {}, tips or [], seniority
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        system, user = kw["messages"][0]["content"], kw["messages"][1]["content"]
        if system.startswith("You read a job description"):
            data = self.extract
        else:
            listing = user.split("<requirements>\n")[1].split("\n</requirements>")[0]
            js = []
            for line in listing.split("\n"):
                r = json.loads(line)
                status, ev = self.judge.get(r["requirement"], ("missing", ""))
                js.append({"id": r["id"], "status": status, "resume_evidence": ev})
            data = {"judgments": js, "seniority_fit": self.seniority, "tips": self.tips}
        msg = types.SimpleNamespace(content=json.dumps(data))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    def n(self, kind):
        return sum(1 for c in self.calls if c["messages"][0]["content"].startswith("You read a job description") == (kind == "extract"))


def use(model):
    main.clear_cache()
    main.client.chat.completions.create = model.create
    return model


US = CandidateProfile(country="United States", location="Malden, MA")
NEPAL = CandidateProfile(country="Nepal")


# ---- eligibility / constraints ----
def test_soft_requirements_are_not_scored():
    use(FakeModel(ai_extract(), AI_EVIDENCE))
    out = main.run_analysis(RESUME, AI_JD, US)
    assert out.match_score == 100, out.match_score
    assert out.missing_skills == []
    assert sorted(s.requirement for s in out.soft_requirements) == ["Excellent writing and grammar skills", "Fluency in English"]


def test_no_profile_country_rule_is_check():
    use(FakeModel(ai_extract(), AI_EVIDENCE))
    out = main.run_analysis(RESUME, AI_JD)
    assert out.eligibility == "check" and not out.profile_used
    assert "Profile" in out.hard_constraints[0].reason


def test_profile_country_in_list_is_ok():
    use(FakeModel(ai_extract(), AI_EVIDENCE))
    out = main.run_analysis(RESUME, AI_JD, US)
    assert out.eligibility == "ok" and out.hard_constraints[0].status == "ok"


def test_country_derived_from_location_text_when_country_blank():
    use(FakeModel(ai_extract(), AI_EVIDENCE))
    out = main.run_analysis(RESUME, AI_JD, CandidateProfile(location="Malden, MA, USA"))
    assert out.eligibility == "ok"


def test_profile_country_outside_list_blocks():
    use(FakeModel(ai_extract(), AI_EVIDENCE))
    out = main.run_analysis(RESUME, AI_JD, NEPAL)
    assert out.eligibility == "blocked" and "Nepal" in out.hard_constraints[0].reason
    assert "Eligibility conflict" in out.feedback


def test_needs_sponsorship_yes_does_not_invent_a_visa_blocker():
    # The model labels the location sentence as a visa constraint (the bug seen in the app).
    bogus = [econ("Visa sponsorship", LOC_QUOTE, "work_authorization")]
    for needs in ("yes", "no", "unknown"):
        use(FakeModel(ai_extract(bogus), AI_EVIDENCE))
        prof = CandidateProfile(country="United States", needs_sponsorship=needs)
        out = main.run_analysis(RESUME, AI_JD, prof)
        assert out.hard_constraints == [], (needs, out.hard_constraints)
        assert out.eligibility == "ok"
    assert "Visa sponsorship" in main._CACHE[next(iter(main._CACHE))].dropped_constraints


def test_three_profile_variants_same_score_and_requirements():
    m = use(FakeModel(ai_extract(), AI_EVIDENCE))
    outs = [
        main.run_analysis(RESUME, AI_JD),
        main.run_analysis(RESUME, AI_JD, CandidateProfile(country="United States", needs_sponsorship="yes")),
        main.run_analysis(RESUME, AI_JD, CandidateProfile(country="United States", needs_sponsorship="no")),
    ]
    assert len({o.match_score for o in outs}) == 1
    assert len({tuple(o.matched_skills) for o in outs}) == 1
    assert [o.eligibility for o in outs] == ["check", "ok", "ok"]
    assert m.n("extract") == 1   # JD analysed once, then cached


def test_real_sponsorship_rule_is_judged_from_profile():
    quote = "We are unable to sponsor visas for this role."
    ex = {"requirements": [ereq("Go", "Experience with Go."), ereq("Backend", "5+ years of backend experience."),
                           ereq("Degree", "A bachelor's degree.")],
          "hard_constraints": [econ("No visa sponsorship", quote, "work_authorization")], "seniority_note": ""}
    got = {}
    for needs in ("yes", "no", "unknown"):
        use(FakeModel(ex))
        got[needs] = main.run_analysis(PAY_RESUME, PAY_JD, CandidateProfile(needs_sponsorship=needs)).eligibility
    assert got == {"yes": "blocked", "no": "ok", "unknown": "check"}, got


def test_constraint_quote_must_support_label():
    cons = [
        econ("Security clearance", LOC_QUOTE, "clearance"),                  # quote has no clearance word
        econ("Location", "Fully remote: work from anywhere in the US", "location_country", ["Mars"]),  # place not in quote
        econ("Made up rule", "Must hold an active security clearance.", "clearance"),   # not in JD
    ]
    use(FakeModel(ai_extract(cons), AI_EVIDENCE))
    out = main.run_analysis(RESUME, AI_JD, NEPAL)
    assert out.hard_constraints == [] and out.eligibility == "ok"


def test_constraint_labels_are_written_by_code():
    use(FakeModel(ai_extract([econ("location_country", LOC_QUOTE, "location_country", PLACES)]), AI_EVIDENCE))
    out = main.run_analysis(RESUME, AI_JD, NEPAL)
    assert out.hard_constraints[0].constraint == "Open only to US, Canada, UK, Ireland, Australia, New Zealand"
    assert "location_country" not in out.feedback
    assert out.hard_constraints[0].reason.startswith("Open only to United States, Canada, United Kingdom, Ireland")


def test_country_aliases_and_unknown_regions():
    assert main.canon_country("U.S.A.") == main.canon_country("United States") == "united states"
    assert main.canon_country("UK") == main.canon_country("Great Britain") == "united kingdom"
    assert main.find_country_in("Malden, MA, USA") == "united states"
    # A JD that lists a region we can't match must never produce a hard conflict.
    c = main.ExtractedCon(constraint="EU only", jd_quote="Open to candidates in the EU.", type="location_country", places=["EU"])
    assert main._judge_one(c, CandidateProfile(country="Nepal"))[0] == "unverifiable"
    c2 = main.ExtractedCon(constraint="x", jd_quote="Open to candidates in Canada excluding Quebec.", type="location_country", places=["Canada"])
    assert main._judge_one(c2, CandidateProfile(country="Nepal"))[0] == "unverifiable"
    assert main._judge_one(c2, CandidateProfile(country="Zimbabwe"))[0] == "unverifiable"


def test_profile_never_reaches_the_model():
    m = use(FakeModel(ai_extract(), AI_EVIDENCE))
    main.run_analysis(RESUME, AI_JD, CandidateProfile(country="Nepal", location="Kathmandu", needs_sponsorship="yes"))
    blob = json.dumps([c["messages"] for c in m.calls]).lower()
    assert "nepal" not in blob and "kathmandu" not in blob
    assert "candidate_profile" not in blob and "needs_sponsorship" not in blob


# ---- section scoping / caching ----
def test_intro_text_creates_no_requirement():
    extra = [ereq("Experience with AI systems", "work closely with state-of-the-art AI models")]
    m = use(FakeModel(ai_extract(extra_reqs=extra), {**AI_EVIDENCE}))
    out = main.run_analysis(RESUME, AI_JD, US)
    names = out.matched_skills + out.missing_skills + out.partial_skills
    assert "Experience with AI systems" not in names
    ex = main._CACHE[next(iter(main._CACHE))]
    assert ex.dropped_out_of_scope == ["Experience with AI systems"] and ex.scoped


def test_tier_comes_from_section_and_wording():
    extra = [ereq("Evaluate AI code", "Evaluate AI-generated code for accuracy, performance, and clarity.", tier="must")]
    use(FakeModel(ai_extract(extra_reqs=extra), AI_EVIDENCE))
    out = main.run_analysis(RESUME, AI_JD, US)
    tiers = {q["requirement"]: q["tier"] for q in out.score_breakdown["requirements"]}
    assert tiers["Evaluate AI code"] == "nice"                 # responsibilities section
    assert tiers["Software developer experience"] == "nice"    # "is preferred" wording
    assert tiers["Bachelor's degree"] == "must"


def test_scoping_safety_net_when_it_would_leave_too_little():
    jd = "About us:\nWe use Rust and Go every day.\nWe also need Python.\nAnd Kubernetes.\n\nQualifications:\nA degree.\n"
    ex = {"requirements": [ereq("Rust", "We use Rust and Go every day."), ereq("Python", "We also need Python."),
                           ereq("Kubernetes", "And Kubernetes."), ereq("Degree", "A degree.")],
          "hard_constraints": [], "seniority_note": ""}
    use(FakeModel(ex, {}))
    out = main.run_analysis("resume", jd)
    assert len(out.score_breakdown["requirements"]) == 4 and out.score_breakdown["jd_scoped"] is False


def test_extraction_cached_per_jd_but_resume_still_judged():
    m = use(FakeModel(ai_extract(), AI_EVIDENCE))
    a = main.run_analysis(RESUME, AI_JD, US)
    m.judge = {**AI_EVIDENCE, "Bachelor's degree": ("missing", "")}
    b = main.run_analysis(RESUME, AI_JD, US)
    assert m.n("extract") == 1 and m.n("judge") == 2
    assert b.score_breakdown["jd_cached"] and not a.score_breakdown["jd_cached"]
    assert b.match_score < a.match_score and "Bachelor's degree" in b.missing_skills


# ---- tips ----
def test_tips_are_validated_against_their_gap():
    ex = {"requirements": [ereq("5+ years backend", "5+ years of backend experience.", core=True),
                           ereq("Go programming", "Experience with Go."),
                           ereq("Bachelor's degree", "A bachelor's degree."),
                           ereq("Kubernetes", "Kubernetes experience.", tier="nice")],
          "hard_constraints": [], "seniority_note": "5+ years"}
    judge = {"5+ years backend": ("partial", "Built services in Python for 3 years"),
             "Go programming": ("missing", ""),
             "Bachelor's degree": ("partial", "B.Sc. in Computer Science"),
             "Kubernetes": ("missing", "")}
    tips = [
        {"id": 1, "tip": "Gain two more years of backend experience."},          # years requirement: no tips
        {"id": 2, "tip": "Build a small Go service and list it under Projects."},  # valid
        {"id": 3, "tip": "Consider completing your master's degree."},           # degree: dropped
        {"id": 4, "tip": "Take up a hobby in woodworking."},                      # unrelated: dropped
        {"id": 9, "tip": "Out of range."},
    ]
    use(FakeModel(ex, judge, tips))
    out = main.run_analysis(PAY_RESUME, PAY_JD)
    assert "Go service" in out.feedback
    assert "master" not in out.feedback and "woodworking" not in out.feedback and "two more years" not in out.feedback


def test_unverified_evidence_gets_no_tip_and_explains_itself():
    ex = {"requirements": [ereq("Go programming", "Experience with Go."), ereq("Backend", "5+ years of backend experience."),
                           ereq("Degree", "A bachelor's degree.")], "hard_constraints": [], "seniority_note": ""}
    judge = {"Go programming": ("met", "Wrote Go microservices at Acme"),   # not in the resume
             "Backend": ("met", "Built services in Python for 3 years"), "Degree": ("met", "B.Sc. in Computer Science")}
    use(FakeModel(ex, judge, [{"id": 1, "tip": "Add a Go project to your portfolio."}]))
    out = main.run_analysis(PAY_RESUME, PAY_JD)
    assert out.partial_skills == ["Go programming"]
    assert "Couldn't confirm Go programming" in out.feedback and "portfolio" not in out.feedback


def test_skipped_requirement_counts_missing_not_crash():
    m = use(FakeModel(ai_extract(), AI_EVIDENCE))
    orig = m.create

    def create(**kw):
        r = orig(**kw)
        content = json.loads(r.choices[0].message.content)
        if "judgments" in content:
            content["judgments"] = content["judgments"][:-1]
            r.choices[0].message.content = json.dumps(content)
        return r
    main.client.chat.completions.create = create
    out = main.run_analysis(RESUME, AI_JD, US)
    assert out.score_breakdown["unjudged"]


def test_coercion_of_sloppy_output():
    e = main._coerce_extract({"requirements": [{"requirement": "a", "jd_quote": "b", "tier": "Preferred", "kind": "Soft skill"}],
                              "hard_constraints": [{"constraint": "c", "jd_quote": "d", "type": "weird", "places": "US"}]})
    assert e["requirements"][0]["tier"] == "nice" and e["requirements"][0]["kind"] == "soft"
    assert e["hard_constraints"][0]["type"] == "other" and e["hard_constraints"][0]["places"] == []
    j = main._coerce_judge({"judgments": [{"id": "1", "status": "major_gap"}, {"id": "1", "status": "met"}, "junk"],
                            "tips": [{"id": "x", "tip": "t"}, {"id": "2", "tip": "ok"}]})
    assert j["judgments"] == [{"id": 1, "status": "missing", "resume_evidence": ""}]
    assert j["tips"] == [{"id": 2, "tip": "ok"}] and j["seniority_fit"] == "stretch"


def test_only_soft_requirements_refuses_to_score():
    ex = ai_extract()
    ex["requirements"] = [r for r in ex["requirements"] if r["kind"] == "soft" or "writing" in r["requirement"].lower()]
    use(FakeModel(ex))
    try:
        main.run_analysis(RESUME, AI_JD, US)
    except ValueError:
        assert len(main._CACHE) == 0   # failed extractions are not cached
        return
    raise AssertionError("expected ValueError")


def test_legacy_profile_payload_is_accepted():
    p = CandidateProfile.model_validate({"location": "Malden, MA, USA", "work_authorization": "needs sponsorship"})
    assert p.needs_sponsorship == "unknown" and not p.is_empty()
    assert CandidateProfile.model_validate({"needs_sponsorship": True}).needs_sponsorship == "yes"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
