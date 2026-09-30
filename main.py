import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

app = FastAPI(title="Resume Matcher API")

# Lock this down to your actual Flutter web origin / app in production —
# "*" is fine for local development only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# 1. Load the variables from your hidden .env file into memory
load_dotenv() 

# 2. Initialize the client safely
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a strict, realistic technical recruiter. Compare this resume against this job description critically — do not be generous. Most resumes have some gaps; find them honestly.

Rules:
- match_score should rarely exceed 85 unless the resume is a near-perfect literal match to every single requirement listed.
- missing_skills must include anything mentioned in the job description that is not clearly evidenced in the resume, even minor ones (e.g. specific years of experience, specific tools, soft skills explicitly requested).
- Do not pad matched_skills with generic resume content that isn't specifically relevant to this job description's actual requirements.
- feedback must include at least one honest, specific area for improvement — never say the resume has no weaknesses.

Return ONLY valid JSON, no text before or after, in this exact format:
{
  "match_score": <0-100>,
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "feedback": "2-3 specific sentences including at least one real gap or improvement"
}
"""


class AnalyzeRequest(BaseModel):
    resume: str
    job_description: str


class AnalyzeResponse(BaseModel):
    match_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    feedback: str


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not req.resume.strip() or not req.job_description.strip():
        raise HTTPException(400, "Both resume and job_description are required.")

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Resume:\n{req.resume}\n\nJob Description:\n{req.job_description}",
                },
            ],
        )
        parsed = json.loads(completion.choices[0].message.content)
        return AnalyzeResponse(**parsed)
    except Exception as e:
        # In a real deployment, log the exception with real logging (not print)
        # and return a generic message — never leak internals to the client.
        raise HTTPException(502, f"Analysis failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}
