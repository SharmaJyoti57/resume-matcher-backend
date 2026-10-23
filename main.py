"""Resume Matcher API v4.

Pipeline (the model never sees your profile, and the JD is analysed once per JD):
  1. EXTRACT  (model, JD only, cached per JD): requirements + typed hard constraints + seniority note.
     Code then verifies quotes against the JD, scopes requirements to the JD's own
     Qualifications / Requirements sections (intro, company and perk text creates no requirements),
     sets the tier from the section / wording, and checks every constraint's quote really supports
     its label (a "visa" constraint needs a visa / sponsorship word in its quote, and so on).
  2. JUDGE    (model, resume + numbered requirements): status + verbatim evidence per requirement,
     seniority fit, and optional tips keyed by requirement number.
  3. CODE     verifies evidence, scores, validates tips against the gap they belong to, builds feedback,
     and judges eligibility by comparing the JD's typed constraints with the structured profile
     (country, needs-sponsorship yes / no / unknown).

Consequences: a profile change can never move the score (the model never sees it), and the same JD always
produces the same requirement list (cached), so score differences come only from the resume.
"""
import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()
log = logging.getLogger("resume_matcher")

app = FastAPI(title="Resume Matcher API")

# Lock this down to your actual Flutter web origin / app in production --
# "*" is fine for local development only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = os.environ.get("MATCHER_MODEL", "gpt-4o-mini")
PROMPT_VERSION = "v4.0"   # bump when either prompt changes: it invalidates the JD cache

# ---- Scoring config: tune these against your labeled pairs -------------------
WEIGHTS = {"must": 0.70, "nice": 0.20, "level": 0.10}
CREDIT = {"met": 1.0, "partial": 0.5, "missing": 0.0}
LEVEL_CREDIT = {"aligned": 1.0, "stretch": 0.5, "major_gap": 0.0}
CORE_WEIGHT = 3.0          # a core requirement counts this many times inside its tier
CAP_CORE_MISSING = 60      # a defining requirement has no evidence
CAP_CORE_PARTIAL = 80      # a defining requirement is only adjacent
CAP_MAJOR_SENIORITY = 55   # large years/level gap
APPLY_AT = 75
STRETCH_AT = 55
MIN_GROUNDED_REQUIREMENTS = 3
JD_MATCH = 0.85            # share of quote words that must appear in the JD
RESUME_MATCH = 0.75        # share of evidence words that must appear in the resume
CACHE_SIZE = 256

# Backup for when the model forgets to flag a soft requirement. Matches the LABEL only.
SOFT_PATTERN = re.compile(
    r"\b(fluen\w*|native[- ]level|bilingual|communication skills?|written (?:and|&) verbal"
    r"|verbal (?:and|&) written|(?:excellent|strong|good|clear) (?:written|writing|verbal|interpersonal)"
    r"|grammar|interpersonal|team player|self[- ]motivated|detail[- ]oriented|work ethic|passion\w*"
    r"|willing(?:ness)? to|eager to learn|fast learner|time management|ability to explain)",
    re.I,
)

EXTRACT_PROMPT = """You read a job description (JD) and list what it requires. You never see any candidate. Return JSON only.

1. requirements (6-15): concrete things a candidate must show. Take them from the JD's qualifications / requirements / preferred lists, and from specific technologies or experience named in the responsibilities. Do NOT take requirements from the company description, the intro paragraph, perks, pay or application logistics. Merge near-duplicates. For each:
   - requirement: short label, e.g. "React Native production experience"
   - jd_quote: a VERBATIM excerpt (max 20 words) copied from ONE line of the JD that states it
   - tier: "must" for named technologies, stated years, degrees and anything called required / minimum / must; "nice" for anything called preferred / a plus / ideal / bonus, and for generic task bullets under responsibilities
   - kind: "soft" for traits a resume cannot list as evidence (language fluency, quality of writing or communication, work style, attitude, willingness, passion); "skill" for everything else (technologies, tools, degrees, years, domain experience)
   - is_core: true only for the at most 3 skill requirements that define the role (primary stack, core domain, core seniority); all others false
2. hard_constraints: eligibility conditions that are not skills. Each has:
   - constraint: short label
   - jd_quote: VERBATIM excerpt (max 25 words) of the sentence that states the rule itself
   - type: one of
       "work_authorization"  visa, sponsorship, right to work, citizenship
       "location_country"    the job is open only to people in certain countries
       "location_onsite"     must work from a specific city / office / hybrid days
       "clearance"           security clearance
       "license"             required license or professional certification
   - places: for the two location types, the country or city names exactly as written in the quote; otherwise []
   List a constraint only if its quote states the rule. A work_authorization constraint's quote must mention visas, sponsorship or work authorization; a location constraint's quote must name where the candidate has to be. If the JD states no such rule, return an empty list. Never infer a rule.
3. seniority_note: the years of experience or level the JD asks for, in a few words ("5+ years mobile development", "senior"), or "" if it states none.

Rules: judge only what is written. Quotes must be copied character for character; if you cannot quote it, do not list it.

Return ONLY valid JSON:
{
  "requirements": [{"requirement": "...", "jd_quote": "...", "tier": "must|nice", "kind": "skill|soft", "is_core": false}],
  "hard_constraints": [{"constraint": "...", "jd_quote": "...", "type": "work_authorization|location_country|location_onsite|clearance|license", "places": []}],
  "seniority_note": ""
}
"""

JUDGE_PROMPT = """You check a resume against a numbered list of job requirements. You do NOT produce an overall score. Return JSON only.

For every requirement id give:
- status:
    "met": the resume shows it explicitly. Evidence may be anywhere: a work bullet, a project, or a role's "Tech Stack:" line (a technology named in a role's Tech Stack line counts as used at work). A shipped / released / published statement counts as evidence of publishing.
    "partial": adjacent but not the thing itself (e.g. Flutter for a native Android requirement, 2 years where 5 are required, a personal project where production use is required), or a skill that appears only in a Skills list with no role or project using it
    "missing": no evidence anywhere in the resume
- resume_evidence: a VERBATIM excerpt (max 25 words) copied from ONE line of the resume that justifies met or partial; "" if missing. Short is fine. Never merge lines, never use "..." and never paraphrase.
Degrees: B.E., B.Tech, B.Sc, B.A., BS, BEng and "Bachelor of ..." are bachelor's degrees. A master's or doctorate also proves a bachelor's. "Completed or in progress" accepts either.

seniority_fit: compare <seniority_asked> with the candidate's relevant professional experience, computed from the resume's dates.
  "aligned": candidate meets it, nothing is asked, or the role is more junior than the candidate (internship / entry-level is "aligned").
  "stretch": short by up to about half of the required years, or one level.
  "major_gap": short by more than that.

tips: at most 2 items {"id": <id of a requirement you marked missing or partial>, "tip": ONE short sentence with a concrete way to close THAT gap (a project to build, a resume rewording, a certification)}. No tips for degrees or years of experience.

Rules: search the WHOLE resume before marking something missing. Judge only what is written; do not assume skills from job titles. Never mention a numeric score. status is exactly "met", "partial" or "missing".

Return ONLY valid JSON:
{
  "judgments": [{"id": 1, "status": "met|partial|missing", "resume_evidence": "..."}],
  "seniority_fit": "aligned|stretch|major_gap",
  "tips": [{"id": 1, "tip": "..."}]
}
"""


# ---- Schemas -----------------------------------------------------------------
class CandidateProfile(BaseModel):
    """Facts only the candidate knows. Compared with the JD by CODE; never sent to the model."""
    country: str = Field(default="", max_length=80)
    location: str = Field(default="", max_length=200)   # optional city / region text
    needs_sponsorship: Literal["yes", "no", "unknown"] = "unknown"

    @field_validator("needs_sponsorship", mode="before")
    @classmethod
    def _tolerant_sponsorship(cls, v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        s = str(v or "unknown").strip().lower()
        return s if s in ("yes", "no", "unknown") else "unknown"

    def is_empty(self) -> bool:
        return not (self.country.strip() or self.location.strip()) and self.needs_sponsorship == "unknown"


class ExtractedReq(BaseModel):
    requirement: str
    jd_quote: str
    tier: Literal["must", "nice"] = "must"
    kind: Literal["skill", "soft"] = "skill"
    is_core: bool = False


class ExtractedCon(BaseModel):
    constraint: str
    jd_quote: str
    type: Literal["work_authorization", "location_country", "location_onsite", "clearance", "license", "other"]
    places: list[str] = []


class Extraction(BaseModel):
    requirements: list[ExtractedReq]
    hard_constraints: list[ExtractedCon] = []
    seniority_note: str = ""


class Judgment(BaseModel):
    id: int
    status: Literal["met", "partial", "missing"]
    resume_evidence: str = ""


class Tip(BaseModel):
    id: int
    tip: str


class JudgeOutput(BaseModel):
    judgments: list[Judgment]
    seniority_fit: Literal["aligned", "stretch", "major_gap"]
    tips: list[Tip] = []


class Requirement(BaseModel):
    """A scoped extraction plus the resume judgment."""
    requirement: str
    jd_quote: str
    tier: Literal["must", "nice"]
    kind: Literal["skill", "soft"] = "skill"
    is_core: bool = False
    section: str = ""
    status: Literal["met", "partial", "missing"] = "missing"
    resume_evidence: str = ""


class JDExtraction(BaseModel):
    """The cached, fully verified result of step 1 for one JD."""
    reqs: list[Requirement]
    constraints: list[ExtractedCon]
    seniority_note: str = ""
    dropped_not_in_jd: int = 0
    dropped_out_of_scope: list[str] = []
    dropped_constraints: list[str] = []
    scoped: bool = False


class Constraint(BaseModel):
    constraint: str
    jd_quote: str
    status: Literal["ok", "conflict", "unverifiable"]
    reason: str = ""


class SoftNote(BaseModel):
    requirement: str
    jd_quote: str


class AnalyzeRequest(BaseModel):
    resume: str
    job_description: str
    profile: Optional[CandidateProfile] = None


class AnalyzeResponse(BaseModel):
    # Original fields, unchanged, so the Flutter app keeps working.
    match_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    feedback: str
    # Added in v2.
    verdict: Literal["Apply", "Stretch", "Skip"]
    partial_skills: list[str]
    hard_constraints: list[Constraint]
    score_breakdown: dict
    # Added in v3.
    eligibility: Literal["ok", "check", "blocked"]
    soft_requirements: list[SoftNote]
    profile_used: bool


# ---- Input cleaning ----------------------------------------------------------
_BOILERPLATE = re.compile(
    r"^\s*(posted\b|source:|visit job page|copy link|save job|find contacts|report job|apply now)",
    re.I,
)


def clean_text(text: str, strip_boilerplate: bool = False) -> str:
    text = text.replace("\\n", "\n").replace("\r", "")
    lines = [ln.rstrip() for ln in text.split("\n")]
    if strip_boilerplate:
        lines = [ln for ln in lines if not _BOILERPLATE.match(ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ---- Grounding: model quotes must really exist in the inputs -----------------
def _norm(s: str) -> str:
    # Lowercase; every run of characters other than letters, digits, + and # becomes one space.
    return re.sub(r"[^a-z0-9+#]+", " ", s.lower()).strip()


def grounded(quote: str, source_norm: str, threshold: float) -> bool:
    """True if every fragment of the quote (split on '...') appears in the source as a whole-word
    phrase, or (for fragments of 3+ words) at least `threshold` of its words appear in the source."""
    padded = f" {source_norm} "
    fragments = [f for f in re.split(r"\.\.\.|…", quote) if _norm(f)]
    if not fragments:
        return False
    src_tokens = set(source_norm.split())
    for frag in fragments:
        q = _norm(frag)
        if f" {q} " in padded:
            continue
        tokens = q.split()
        if len(tokens) >= 3 and sum(t in src_tokens for t in tokens) / len(tokens) >= threshold:
            continue
        return False
    return True


def is_soft(r: Requirement) -> bool:
    return r.kind == "soft" or bool(SOFT_PATTERN.search(r.requirement))


# ---- JD structure: which section does a line belong to? ----------------------
_BULLET = re.compile(r"^\s*(?:[•·●▪‣◦\-–—*]|\d+[.)])\s+")
_PREF_H = re.compile(
    r"(?:preferred|desired|desirable|bonus|nice to haves?|plus(?:es)?|ideal|extra)"
    r"(?: (?:qualifications?|skills|experience|requirements?|points))?"
    r"|great to have|bonus points|it's a plus|what sets you apart"
)
_REQ_H = re.compile(
    r"(?:(?:required|minimum|basic|key|core|essential|must have) )?"
    r"(?:qualifications?|requirements?|skills(?: (?:and|&) (?:experience|qualifications|requirements))?|experience)"
    r"(?: (?:and|&) (?:qualifications?|requirements?|skills))?"
    r"|(?:responsibilit(?:y|ies)|duties) (?:and|&) (?:qualifications?|requirements?)"
    r"|what you(?:'ll)? (?:bring|need|have)|what we(?:'re| are) looking for|what's required|who you are"
    r"|about you|you have|you bring|(?:the )?ideal candidate|must haves?|your background|looking for"
)
_DUTY_H = re.compile(
    r"(?:(?:key|main|core|job|primary) )?(?:responsibilit(?:y|ies)|duties)"
    r"|what you(?:'ll| will) (?:do|work on|be doing|be working on|own)"
    r"|day to day|in this role|your (?:role|impact|mission)|the work|you will"
)
_OTHER_H = re.compile(
    r"about (?:us|the (?:company|team|role|job|position|opportunity)|[a-z0-9&. ]{1,30})"
    r"|(?:full |complete )?job description|(?:company |role |position |job )?(?:overview|summary|description)"
    r"|who we are|our (?:mission|values|culture|story|team)|benefits|perks|(?:compensation|salary|pay)(?: (?:and|&) benefits)?"
    r"|what we offer|why (?:join|work)(?: us| with us| at .*)?|equal opportunity.*|how to apply|application process"
    r"|the opportunity|(?:work )?location"
)
_PREF_CUE = re.compile(r"\b(preferred|a plus|nice to have|bonus|desirable|ideal(?:ly)?)\b", re.I)


def heading_kind(line: str) -> Optional[str]:
    """requirements | preferred | duties | other | unknown, or None if the line is not a heading."""
    s = line.strip()
    if not s or _BULLET.match(line):
        return None
    clean = re.sub(r"^[#*_\s]+|[#*_\s:]+$", "", s)
    if not clean or len(clean.split()) > 8 or clean.endswith((".", ",", ";")):
        return None
    h = re.sub(r"[^a-z0-9&' ]+", " ", clean.lower().replace("’", "'"))
    h = re.sub(r"\s+", " ", h).strip()
    for kind, pat in (("preferred", _PREF_H), ("requirements", _REQ_H), ("duties", _DUTY_H), ("other", _OTHER_H)):
        if pat.fullmatch(h):
            return kind
    return "unknown" if s.rstrip("*_ ").endswith(":") else None


def jd_lines(jd: str):
    """[(raw_line, normalized_line, section)] for every non-heading, non-empty line."""
    section, out = "intro", []
    for raw in jd.split("\n"):
        kind = heading_kind(raw)
        if kind:
            section = kind
            continue
        n = _norm(raw)
        if n:
            out.append((raw, n, section))
    return out


def locate(quote: str, lines) -> Optional[tuple]:
    """The JD line a quote came from: (raw, norm, section), or None if it can't be placed."""
    frags = [f for f in re.split(r"\.\.\.|…", quote) if _norm(f)]
    if not frags:
        return None
    q = _norm(frags[0])
    for ln in lines:
        if f" {q} " in f" {ln[1]} ":
            return ln
    toks = q.split()
    best, best_ratio = None, 0.0
    for ln in lines:
        src = set(ln[1].split())
        ratio = sum(t in src for t in toks) / len(toks)
        if ratio > best_ratio:
            best, best_ratio = ln, ratio
    return best if best_ratio >= 0.7 else None


def scope_requirements(reqs: list[Requirement], jd: str):
    """Keep only requirements that sit in the JD's own requirement-bearing text; set tier from the
    section / wording. Returns (kept, dropped_names, scoped)."""
    lines = jd_lines(jd)
    has_req_section = any(s in ("requirements", "preferred") for _, _, s in lines)
    drop_sections = {"intro", "other"} if has_req_section else {"other"}
    kept, dropped = [], []
    for r in reqs:
        ln = locate(r.jd_quote, lines)
        section = ln[2] if ln else "unknown"
        tier = r.tier
        if ln:
            if section == "preferred" or _PREF_CUE.search(ln[0]) or section == "duties":
                tier = "nice"
        r = r.model_copy(update={"section": section, "tier": tier})
        if section in drop_sections:
            dropped.append(r.requirement)
        else:
            kept.append(r)
    if len(kept) < MIN_GROUNDED_REQUIREMENTS <= len(reqs):
        return _unscoped(reqs, lines), [], False
    return kept, dropped, has_req_section


def _unscoped(reqs, lines):
    # Safety net: scoping would leave too little to score, so keep everything (tiers still adjusted).
    out = []
    for r in reqs:
        ln = locate(r.jd_quote, lines)
        tier = "nice" if ln and (_PREF_CUE.search(ln[0]) or ln[2] in ("preferred", "duties")) else r.tier
        out.append(r.model_copy(update={"section": ln[2] if ln else "unknown", "tier": tier}))
    return out


# ---- Constraints: the quote must support the label ---------------------------
SUPPORT = {
    "work_authorization": re.compile(
        r"sponsor|visa|authori[sz]|work permit|right to work|eligible to work|eligibility to work|citizen"
        r"|green card|permanent resident|h-?1b|\bopt\b|work in the", re.I),
    "clearance": re.compile(r"clearance|\bsecret\b|ts/?sci|polygraph|security (?:check|vetting)", re.I),
    "license": re.compile(r"licen[sc]e|licensed|certif|registered|credential|\bboard\b", re.I),
    "location_onsite": re.compile(
        r"on-?site|in[- ]office|in[- ]person|hybrid|relocat|\boffice\b|commut|days? (?:a|per|each) week"
        r"|located in|based in", re.I),
    "location_country": re.compile(
        r"only|available|located|based|resid|remote|work from|must be|eligible|open to|candidates|within"
        r"|anywhere in|authori[sz]|countr|region|apply from|citizen", re.I),
}
NO_SPONSOR = re.compile(
    r"\b(?:no|not|without|unable to|cannot|can't|can not|won't|will not|don't|do not|doesn't|does not|never|unavailable)\b"
    r"[^.;]{0,50}\bsponsor\w*|\bsponsor\w*[^.;]{0,30}\b(?:not|unavailable|n't)\b",
    re.I,
)
CITIZENSHIP = re.compile(r"citizen|permanent resident|green card", re.I)
EXCLUSION = re.compile(r"\b(excluding|except|not available in|unavailable in|other than)\b", re.I)

_COUNTRY_ALIASES = {
    "united states": ["united states", "united states of america", "usa", "us", "u s", "u s a", "america"],
    "canada": ["canada"],
    "united kingdom": ["united kingdom", "uk", "u k", "great britain", "britain", "england", "scotland", "wales",
                       "northern ireland"],
    "ireland": ["ireland", "republic of ireland"],
    "australia": ["australia"],
    "new zealand": ["new zealand", "nz"],
    "india": ["india"], "nepal": ["nepal"], "pakistan": ["pakistan"], "bangladesh": ["bangladesh"],
    "sri lanka": ["sri lanka"], "germany": ["germany"], "france": ["france"], "spain": ["spain"],
    "italy": ["italy"], "netherlands": ["netherlands", "the netherlands", "holland"], "belgium": ["belgium"],
    "portugal": ["portugal"], "switzerland": ["switzerland"], "austria": ["austria"], "poland": ["poland"],
    "sweden": ["sweden"], "norway": ["norway"], "denmark": ["denmark"], "finland": ["finland"],
    "romania": ["romania"], "ukraine": ["ukraine"], "turkey": ["turkey", "turkiye"], "israel": ["israel"],
    "brazil": ["brazil"], "mexico": ["mexico"], "argentina": ["argentina"], "colombia": ["colombia"],
    "chile": ["chile"], "philippines": ["philippines"], "indonesia": ["indonesia"], "vietnam": ["vietnam"],
    "malaysia": ["malaysia"], "thailand": ["thailand"], "singapore": ["singapore"], "japan": ["japan"],
    "south korea": ["south korea", "korea", "republic of korea"], "china": ["china"],
    "united arab emirates": ["united arab emirates", "uae"], "saudi arabia": ["saudi arabia"],
    "egypt": ["egypt"], "nigeria": ["nigeria"], "kenya": ["kenya"], "ghana": ["ghana"],
    "south africa": ["south africa"], "russia": ["russia"],
}
_ALIAS_TO_CANON = {_norm(a): canon for canon, al in _COUNTRY_ALIASES.items() for a in al}
_ALIASES_LONGEST_FIRST = sorted(_ALIAS_TO_CANON, key=len, reverse=True)


def canon_country(text: str) -> Optional[str]:
    return _ALIAS_TO_CANON.get(_norm(text))


def find_country_in(text: str) -> Optional[str]:
    padded = f" {_norm(text)} "
    for alias in _ALIASES_LONGEST_FIRST:
        if f" {alias} " in padded:
            return _ALIAS_TO_CANON[alias]
    return None


def _title(canon: str) -> str:
    return canon.title()


def verify_constraints(cons: list[ExtractedCon], jd: str):
    """Keep a constraint only if its quote is in the JD AND supports its label.
    Returns (kept, dropped_labels)."""
    jd_n = _norm(jd)
    kept, dropped = [], []
    for c in cons:
        if c.type == "other" or not grounded(c.jd_quote, jd_n, JD_MATCH):
            dropped.append(c.constraint)
            continue
        quote_n = f" {_norm(c.jd_quote)} "
        if not SUPPORT[c.type].search(c.jd_quote):
            dropped.append(c.constraint)       # label not supported by its own quote
            continue
        places = [p for p in c.places if _norm(p) and f" {_norm(p)} " in quote_n]
        if c.type == "location_country" and not places:
            dropped.append(c.constraint)       # claims a country rule but names no place in the quote
            continue
        kept.append(c.model_copy(update={"places": places if c.type.startswith("location") else []}))
    return kept, dropped


def _judge_one(c: ExtractedCon, profile: CandidateProfile):
    """(status, reason) for one verified constraint, from structured facts only."""
    if c.type == "work_authorization":
        needs = profile.needs_sponsorship
        if CITIZENSHIP.search(c.jd_quote):
            return "unverifiable", "The post has a citizenship / residency rule; check it against your status."
        if NO_SPONSOR.search(c.jd_quote):
            if needs == "yes":
                return "conflict", "The post doesn't offer sponsorship and your profile says you need it."
            if needs == "no":
                return "ok", "No sponsorship offered; your profile says you don't need it."
            return "unverifiable", "The post doesn't offer sponsorship. Set 'Need sponsorship?' in Profile to check."
        if needs == "no":
            return "ok", "Your profile says you don't need sponsorship."
        return "unverifiable", "The post requires work authorization but doesn't say whether it sponsors."

    if c.type == "location_country":
        if EXCLUSION.search(c.jd_quote):
            return "unverifiable", "The location rule has exclusions; read it yourself."
        allowed = {canon_country(p) for p in c.places}
        unknown_places = None in allowed
        allowed.discard(None)
        entered = profile.country.strip()
        mine = canon_country(entered) if entered else find_country_in(profile.location)
        if entered and not mine:
            return "unverifiable", f"Couldn't recognize the country '{entered}' in your profile."
        if not mine:
            return "unverifiable", "Add your country in Profile to check this automatically."
        if mine in allowed:
            return "ok", f"Your country ({_title(mine)}) is in the list."
        if unknown_places:
            return "unverifiable", "The post names regions I can't match to your country."
        ordered = list(dict.fromkeys(canon_country(p) for p in c.places if canon_country(p)))
        return "conflict", f"Open only to {', '.join(_title(a) for a in ordered)}; your profile says {_title(mine)}."

    if c.type == "location_onsite":
        mine_text = _norm(f"{profile.location} {profile.country}")
        if mine_text and any(_norm(p) and f" {_norm(p)} " in f" {mine_text} " for p in c.places):
            return "ok", "Your profile location matches."
        return "unverifiable", "On-site / hybrid rule; check it against where you can work."

    return "unverifiable", "Check this requirement yourself."   # clearance, license


def constraint_label(c: ExtractedCon) -> str:
    """Written by code from the verified type and quote, so a sloppy model label can't leak into the UI."""
    if c.type == "work_authorization":
        if CITIZENSHIP.search(c.jd_quote):
            return "Citizenship / residency requirement"
        return "No visa sponsorship offered" if NO_SPONSOR.search(c.jd_quote) else "Work authorization required"
    if c.type == "location_country":
        return "Open only to " + ", ".join(c.places)
    if c.type == "location_onsite":
        return "On-site / hybrid: " + ", ".join(c.places) if c.places else "On-site or hybrid work"
    if c.type == "clearance":
        return "Security clearance required"
    return "License / certification required"


def judge_constraints(cons: list[ExtractedCon], profile: CandidateProfile) -> list[Constraint]:
    out = []
    for c in cons:
        status, reason = _judge_one(c, profile)
        out.append(Constraint(constraint=constraint_label(c), jd_quote=c.jd_quote, status=status, reason=reason))
    return out


def eligibility_for(cons: list[Constraint]) -> str:
    if any(c.status == "conflict" for c in cons):
        return "blocked"
    if any(c.status == "unverifiable" for c in cons):
        return "check"
    return "ok"


# ---- Evidence verification ----------------------------------------------------
def verify_evidence(reqs: list[Requirement], resume: str):
    resume_n = _norm(resume)
    out, unverified = [], []
    for r in reqs:
        if r.status != "missing" and not grounded(r.resume_evidence, resume_n, RESUME_MATCH):
            # The model believes there is evidence but we can't find the quote: half credit, flagged.
            unverified.append({"requirement": r.requirement, "claimed": r.status, "evidence": r.resume_evidence})
            r = r.model_copy(update={"status": "partial"})
        out.append(r)
    return out, unverified


# ---- Scoring -----------------------------------------------------------------
def _ratio(items: list[Requirement]):
    if not items:
        return None
    weights = [CORE_WEIGHT if i.is_core else 1.0 for i in items]
    return sum(w * CREDIT[i.status] for w, i in zip(weights, items)) / sum(weights)


def compute_score(reqs: list[Requirement], seniority: str):
    parts = {
        "must": _ratio([r for r in reqs if r.tier == "must"]),
        "nice": _ratio([r for r in reqs if r.tier == "nice"]),
        "level": LEVEL_CREDIT[seniority],
    }
    used = {k: v for k, v in parts.items() if v is not None}
    total_w = sum(WEIGHTS[k] for k in used)
    raw = 100 * sum(WEIGHTS[k] * v for k, v in used.items()) / total_w

    caps = []
    if any(r.is_core and r.status == "missing" for r in reqs):
        caps.append(("core requirement missing", CAP_CORE_MISSING))
    elif any(r.is_core and r.status == "partial" for r in reqs):
        caps.append(("core requirement only partial", CAP_CORE_PARTIAL))
    if seniority == "major_gap":
        caps.append(("major seniority gap", CAP_MAJOR_SENIORITY))

    score = raw
    for _, cap in caps:
        score = min(score, cap)
    breakdown = {
        "must_ratio": None if parts["must"] is None else round(parts["must"], 3),
        "nice_ratio": None if parts["nice"] is None else round(parts["nice"], 3),
        "seniority_fit": seniority,
        "raw_score": round(raw, 1),
        "caps_applied": [name for name, _ in caps],
    }
    return int(round(score)), breakdown


def verdict_for(score: int) -> str:
    if score >= APPLY_AT:
        return "Apply"
    if score >= STRETCH_AT:
        return "Stretch"
    return "Skip"


# ---- Tips: only for a verified gap, and only if they fit it -------------------
_NO_TIP_KIND = re.compile(r"\b(degrees?|bachelor\w*|master\w*|ph\.?d|doctorate|diploma|years?|yrs?)\b", re.I)
_TIP_STOP = {
    "your", "with", "that", "this", "from", "have", "into", "build", "using", "add", "show", "work", "make",
    "project", "projects", "small", "simple", "resume", "experience", "skills", "skill", "knowledge", "learn",
    "write", "create", "gain", "take", "consider", "also", "like", "such", "more", "about", "their", "will",
}


_SHORT_STOP = {
    "the", "and", "for", "you", "are", "can", "get", "how", "new", "one", "two", "any", "all", "not", "has",
    "was", "who", "its", "out", "job", "our", "via", "per", "use", "add", "set", "in", "to", "of", "an", "is",
    "it", "at", "on", "as", "by", "or", "be", "we", "do", "if", "up", "so", "no", "my", "me", "us", "a",
}


def _content_words(text: str) -> set:
    # Short tokens count too ("go", "c#", "r"), because tech names are often 1-3 characters.
    return {
        w for w in _norm(text).split()
        if (len(w) >= 4 and w not in _TIP_STOP) or (2 <= len(w) <= 3 and w not in _SHORT_STOP) or "+" in w or "#" in w
    }


def valid_tips(tips: list[Tip], scored: list[Requirement], unverified: set) -> dict:
    """{requirement label: tip} for tips that attach to a verified missing/partial skill gap and
    share a content word with that requirement. Anything else is dropped."""
    out = {}
    for t in tips:
        if not 1 <= t.id <= len(scored) or not t.tip.strip():
            continue
        r = scored[t.id - 1]
        if r.status == "met" or r.requirement in unverified:
            continue
        basis = f"{r.requirement} {r.jd_quote}"
        if _NO_TIP_KIND.search(basis):
            continue
        if not (_content_words(t.tip) & _content_words(basis)):
            continue
        out.setdefault(r.requirement, t.tip.strip())
    return out


# ---- Feedback: built from verified data only ---------------------------------
def _gap_key(r: Requirement):
    return (0 if r.is_core else 1, 0 if r.tier == "must" else 1, 0 if r.status == "missing" else 1)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def build_feedback(scored, cons, tip_by_req: dict, unverified: set) -> str:
    met = [r for r in scored if r.status == "met"]
    partial = [r for r in scored if r.status == "partial"]
    if partial:
        parts = [f"Your resume fully covers {len(met)} of {len(scored)} scored requirements and partly covers {len(partial)}."]
    else:
        parts = [f"Your resume fully covers {len(met)} of {len(scored)} scored requirements."]

    gaps = sorted((r for r in scored if r.status != "met"), key=_gap_key)
    if gaps:
        top = gaps[0]
        if top.status == "missing":
            parts.append(f"Biggest gap: {top.requirement}.")
        elif top.requirement in unverified:
            parts.append(
                f"Couldn't confirm {top.requirement} from your resume's wording, so it counts as partial; tap it to review."
            )
        else:
            parts.append(
                f"Closest to a gap: {top.requirement} (your resume shows something adjacent, not the requirement itself)."
            )
        for g in gaps[:2]:
            tip = tip_by_req.get(g.requirement)
            if tip:
                parts.append(_sentence(tip))
    else:
        parts.append("No gaps found against the scored requirements.")

    conflicts = [c for c in cons if c.status == "conflict"]
    checks = [c for c in cons if c.status == "unverifiable"]
    if conflicts:
        parts.append("Eligibility conflict: " + "; ".join(c.constraint for c in conflicts[:2]) + ".")
    elif checks:
        parts.append("Verify before applying: " + "; ".join(c.constraint for c in checks[:2]) + ".")
    return " ".join(parts)


# ---- Tolerant parsing of the model's JSON ------------------------------------
_STATUS_FIX = {
    "aligned": "met", "stretch": "partial", "major_gap": "missing",
    "yes": "met", "no": "missing", "none": "missing", "unmet": "missing", "not_met": "missing",
}
_CON_TYPES = {"work_authorization", "location_country", "location_onsite", "clearance", "license"}


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_extract(data):
    if not isinstance(data, dict):
        return data
    reqs = []
    for r in data.get("requirements") or []:
        if not isinstance(r, dict):
            continue
        t = str(r.get("tier", "")).strip().lower()
        r["tier"] = "nice" if t.startswith(("nice", "pref", "bonus")) else "must"
        k = str(r.get("kind", "skill")).strip().lower()
        r["kind"] = "soft" if k.startswith("soft") else "skill"
        if not isinstance(r.get("is_core"), bool):
            r["is_core"] = str(r.get("is_core")).strip().lower() == "true"
        reqs.append(r)
    data["requirements"] = reqs
    cons = []
    for c in data.get("hard_constraints") or []:
        if not isinstance(c, dict):
            continue
        ty = str(c.get("type", "")).strip().lower().replace(" ", "_")
        c["type"] = ty if ty in _CON_TYPES else "other"
        pl = c.get("places")
        c["places"] = [str(p) for p in pl] if isinstance(pl, list) else []
        cons.append(c)
    data["hard_constraints"] = cons
    if not isinstance(data.get("seniority_note"), str):
        data["seniority_note"] = ""
    return data


def _coerce_judge(data):
    if not isinstance(data, dict):
        return data
    seen, js = set(), []
    for j in data.get("judgments") or []:
        if not isinstance(j, dict):
            continue
        i = _as_int(j.get("id"))
        if i is None or i in seen:
            continue
        seen.add(i)
        j["id"] = i
        s = str(j.get("status", "")).strip().lower().replace(" ", "_")
        j["status"] = s if s in CREDIT else _STATUS_FIX.get(s, "missing")
        if not isinstance(j.get("resume_evidence"), str):
            j["resume_evidence"] = ""
        js.append(j)
    data["judgments"] = js
    tips = []
    for t in data.get("tips") or []:
        if isinstance(t, dict) and _as_int(t.get("id")) is not None and isinstance(t.get("tip"), str):
            tips.append({"id": _as_int(t["id"]), "tip": t["tip"]})
    data["tips"] = tips
    s = str(data.get("seniority_fit", "")).strip().lower().replace(" ", "_")
    if s not in LEVEL_CREDIT:
        log.warning("Unexpected seniority_fit %r; using 'stretch'", s)
        s = "stretch"
    data["seniority_fit"] = s
    return data


def _call_json(system: str, user: str, coerce, schema):
    last_error = None
    for attempt in range(2):
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            seed=7 + attempt,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        try:
            return schema.model_validate(coerce(json.loads(completion.choices[0].message.content)))
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            log.warning("Model output invalid (attempt %d): %s", attempt + 1, e)
    raise last_error


# ---- Step 1: JD extraction (cached) ------------------------------------------
_CACHE: "OrderedDict[str, JDExtraction]" = OrderedDict()


def clear_cache():
    _CACHE.clear()


def extract_jd(jd: str):
    """(JDExtraction, from_cache). Depends on the JD text only."""
    key = hashlib.sha256(f"{MODEL}|{PROMPT_VERSION}|{_norm(jd)}".encode()).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key], True

    raw = _call_json(EXTRACT_PROMPT, f"<job_description>\n{jd}\n</job_description>", _coerce_extract, Extraction)

    jd_n = _norm(jd)
    grounded_reqs, not_in_jd = [], 0
    for r in raw.requirements:
        if grounded(r.jd_quote, jd_n, JD_MATCH):
            grounded_reqs.append(Requirement(**r.model_dump()))
        else:
            not_in_jd += 1
    kept, out_of_scope, scoped = scope_requirements(grounded_reqs, jd)
    cons, dropped_cons = verify_constraints(raw.hard_constraints, jd)

    ex = JDExtraction(
        reqs=kept, constraints=cons, seniority_note=raw.seniority_note.strip(),
        dropped_not_in_jd=not_in_jd, dropped_out_of_scope=out_of_scope,
        dropped_constraints=dropped_cons, scoped=scoped,
    )
    scoreable = [r for r in kept if not is_soft(r)]
    if len(kept) < MIN_GROUNDED_REQUIREMENTS or not scoreable:
        # Not cached: a bad extraction must not stick to this JD.
        raise ValueError(f"Only {len(scoreable)} scoreable grounded requirements; refusing to score.")
    _CACHE[key] = ex
    while len(_CACHE) > CACHE_SIZE:
        _CACHE.popitem(last=False)
    return ex, False


# ---- Step 2: judge the resume against the fixed requirement list -------------
def judge_resume(resume: str, scored: list[Requirement], seniority_note: str) -> JudgeOutput:
    listing = "\n".join(
        json.dumps({"id": i, "requirement": r.requirement, "jd_quote": r.jd_quote}, ensure_ascii=False)
        for i, r in enumerate(scored, 1)
    )
    user = (
        f"<resume>\n{resume}\n</resume>\n\n<requirements>\n{listing}\n</requirements>\n\n"
        f"<seniority_asked>\n{seniority_note or 'nothing stated'}\n</seniority_asked>"
    )
    return _call_json(JUDGE_PROMPT, user, _coerce_judge, JudgeOutput)


# ---- Pipeline ----------------------------------------------------------------
def run_analysis(
    resume: str, job_description: str, profile: Optional[CandidateProfile] = None
) -> AnalyzeResponse:
    resume = clean_text(resume)
    jd = clean_text(job_description, strip_boilerplate=True)
    profile = profile or CandidateProfile()

    ex, cached = extract_jd(jd)
    scored_base = [r for r in ex.reqs if not is_soft(r)]
    soft = [r for r in ex.reqs if is_soft(r)]

    judged = judge_resume(resume, scored_base, ex.seniority_note)
    by_id = {j.id: j for j in judged.judgments}
    unjudged = [r.requirement for i, r in enumerate(scored_base, 1) if i not in by_id]
    if unjudged:
        log.warning("Model skipped requirements %s; counting them as missing", unjudged)
    merged = []
    for i, r in enumerate(scored_base, 1):
        j = by_id.get(i)
        merged.append(r.model_copy(update={
            "status": j.status if j else "missing",
            "resume_evidence": j.resume_evidence if j else "",
        }))
    scored, unverified = verify_evidence(merged, resume)
    unverified_names = {u["requirement"] for u in unverified}

    seniority = judged.seniority_fit if ex.seniority_note else "aligned"
    score, breakdown = compute_score(scored, seniority)
    tip_by_req = valid_tips(judged.tips, scored, unverified_names)
    constraints = judge_constraints(ex.constraints, profile)

    breakdown.update(
        n_requirements=len(scored),
        n_unscored_soft=len(soft),
        dropped_not_in_jd=ex.dropped_not_in_jd,
        dropped_out_of_scope=ex.dropped_out_of_scope,
        dropped_constraints=ex.dropped_constraints,
        jd_scoped=ex.scoped,
        jd_cached=cached,
        unjudged=unjudged,
        seniority_note=ex.seniority_note,
        unverified_evidence=unverified,
        profile_used=not profile.is_empty(),
        requirements=[
            {
                "requirement": r.requirement,
                "tier": r.tier,
                "core": r.is_core,
                "status": r.status,
                "evidence": r.resume_evidence,
                "section": r.section,
            }
            for r in scored
        ],
    )
    return AnalyzeResponse(
        match_score=score,
        matched_skills=[r.requirement for r in scored if r.status == "met"],
        missing_skills=[r.requirement for r in scored if r.status == "missing"],
        feedback=build_feedback(scored, constraints, tip_by_req, unverified_names),
        verdict=verdict_for(score),
        partial_skills=[r.requirement for r in scored if r.status == "partial"],
        hard_constraints=constraints,
        score_breakdown=breakdown,
        eligibility=eligibility_for(constraints),
        soft_requirements=[SoftNote(requirement=r.requirement, jd_quote=r.jd_quote) for r in soft],
        profile_used=not profile.is_empty(),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not req.resume.strip() or not req.job_description.strip():
        raise HTTPException(400, "Both resume and job_description are required.")
    try:
        return run_analysis(req.resume, req.job_description, req.profile)
    except Exception:
        log.exception("Analysis failed")
        raise HTTPException(502, "Analysis failed. Please try again.")


@app.get("/health")
def health():
    return {"status": "ok"}
