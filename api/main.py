"""Beat the Detector API (SPEC §7.9). Run locally:

    uvicorn api.main:app --port 8000          # docs at http://localhost:8000/docs

Stateless apart from in-memory game sessions (30-minute idle expiry, at most 25 at once). Nothing is persisted and no
personal data is collected.
"""

import os
import subprocess
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from suspoker.config import ROOT
from suspoker.game.session import MAX_HANDS, SessionStore
from suspoker.predict import Models

ORIGINS = ["https://coder-r2.github.io", "http://localhost:5173", "http://localhost:4173"]

app = FastAPI(title="SusPoker: Beat the Detector", version="1.0",
              description="Play six-max Hold'em with a secret bot partner while a collusion detector watches. "
                          "Every hand is simulated; the detector was trained on the Kaggle competition "
                          "'Detect Suspicious Value Transfers in Poker' (Slash).")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])

MODELS = Models.load()
STORE = SessionStore(MODELS, cap=int(os.environ.get("SUSPOKER_MAX_SESSIONS", 25)))


def _git_sha() -> str:
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


GIT_SHA = _git_sha()


class NewSession(BaseModel):
    strategy: Literal["soft_play", "directed_transfer", "coordinated_isolation", "honest"]
    direction: Literal["partner_to_you", "you_to_partner"] = "partner_to_you"
    intensity: float = Field(0.5, ge=0, le=1)
    partner_seat: int | None = Field(None, ge=1, le=5)


class Action(BaseModel):
    action: Literal["fold", "check", "call", "bet", "raise", "all_in"]
    amount: int | None = Field(None, ge=0, description="bet/raise: your total commitment on this street")


class Autoplay(BaseModel):
    hands: int = Field(..., ge=1, le=50)
    user_mode: Literal["colluding", "honest"] = "colluding"
    animate: bool = False  # return step-by-step events (meant for one hand at a time)


def _session(sid: str):
    try:
        return STORE.get(sid)
    except KeyError:
        raise HTTPException(404, "Session not found or expired") from None


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODELS.meta.get("model_version", "kaggle-04"), "git_sha": GIT_SHA,
            "sessions": len(STORE.sessions), "max_hands": MAX_HANDS}


@app.post("/api/sessions")
def create_session(req: NewSession) -> dict:
    try:
        s = STORE.create(strategy=req.strategy, intensity=req.intensity, direction=req.direction,
                         partner_seat=req.partner_seat)
    except OverflowError as e:
        raise HTTPException(503, str(e)) from None
    with s.lock:
        return {"session_id": s.id, "state": s.state(), "events": s.take_events()}


@app.get("/api/sessions/{sid}")
def get_state(sid: str) -> dict:
    s = _session(sid)
    with s.lock:
        return {"state": s.state()}


@app.post("/api/sessions/{sid}/action")
def act(sid: str, req: Action) -> dict:
    s = _session(sid)
    with s.lock:
        try:
            done = s.act(req.action, req.amount)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return {"state": s.state(), "hand_complete": done, "detector": s.detector_json() if done else None,
                "events": s.take_events()}


@app.post("/api/sessions/{sid}/autoplay")
def autoplay(sid: str, req: Autoplay) -> dict:
    s = _session(sid)
    with s.lock:
        if s.finished:
            raise HTTPException(409, f"The session is over ({MAX_HANDS} hands)")
        s.autoplay(min(req.hands, MAX_HANDS - s.hands_played), req.user_mode, req.animate)
        return {"state": s.state(), "detector": s.detector_json(), "events": s.take_events()}


@app.get("/api/sessions/{sid}/detector")
def detector(sid: str) -> dict:
    s = _session(sid)
    with s.lock:
        return s.detector_json()


@app.get("/api/sessions/{sid}/report")
def report(sid: str) -> dict:
    s = _session(sid)
    with s.lock:
        return s.report()


@app.get("/api/sessions/{sid}/hands/{hand_id}")
def hand(sid: str, hand_id: str) -> dict:
    s = _session(sid)
    with s.lock:
        try:
            return s.replay(hand_id)
        except KeyError:
            raise HTTPException(404, "Hand not found in this session") from None
