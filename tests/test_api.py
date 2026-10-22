"""The game API end to end (SPEC §7.9): needs the game extras and the committed models."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pokerkit")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import STORE, app  # noqa: E402

client = TestClient(app)


def _play_until_hands(sid: str, n: int) -> dict:
    out = {}
    while True:
        st = client.get(f"/api/sessions/{sid}").json()["state"]
        if st["hands_played"] >= n:
            return out
        legal = st["legal"]
        out = client.post(f"/api/sessions/{sid}/action",
                          json={"action": "check" if legal["check"] else "call"}).json()


def test_health():
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["git_sha"]


def test_session_lifecycle_play_autoplay_detector_report_and_replay():
    r = client.post("/api/sessions", json={"strategy": "directed_transfer", "intensity": 0.8, "partner_seat": 3})
    assert r.status_code == 200
    sid, st = r.json()["session_id"], r.json()["state"]
    assert st["to_act"] == 0 and st["legal"] and st["seats"][0]["cards"] and st["seats"][3]["is_partner"]
    assert all(s["cards"] is None for s in st["seats"][1:])  # bots' cards stay hidden during play
    assert client.get(f"/api/sessions/{sid}/detector").json()["status"] == "insufficient"

    _play_until_hands(sid, 3)
    r = client.post(f"/api/sessions/{sid}/autoplay", json={"hands": 40, "user_mode": "colluding"}).json()
    det = r["detector"]
    assert det["status"] == "scored" and len(det["pairs"]) == 15 and det["you_rank"] >= 1
    assert all(p["drivers"] for p in det["pairs"][:3])
    hand_id = r["state"]["last_hand"]["hand_id"]
    rep = client.get(f"/api/sessions/{sid}/hands/{hand_id}").json()
    assert len(rep["seats"]) == 6 and rep["actions"]

    report = client.get(f"/api/sessions/{sid}/report").json()
    assert report["verdict"] in ("caught", "suspected", "clear") and len(report["evidence"]) >= 1
    assert {b["style"] for b in report["bots"]}  # the report reveals the bots
    rep = client.get(f"/api/sessions/{sid}/hands/{report['evidence'][0]['hand_id']}").json()
    assert all("??" not in s["cards"] for s in rep["seats"])  # ...and every card


def test_errors():
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions", json={"strategy": "cheating"}).status_code == 422
    sid = client.post("/api/sessions", json={"strategy": "honest"}).json()["session_id"]
    assert client.post(f"/api/sessions/{sid}/action", json={"action": "raise"}).status_code == 409  # no amount
    assert client.post(f"/api/sessions/{sid}/autoplay", json={"hands": 51}).status_code == 422


def test_session_cap_returns_503():
    cap, STORE.cap = STORE.cap, len(STORE.sessions)
    try:
        assert client.post("/api/sessions", json={"strategy": "honest"}).status_code == 503
    finally:
        STORE.cap = cap


def test_action_returns_step_by_step_events_for_animation():
    sid = client.post("/api/sessions", json={"strategy": "soft_play", "partner_seat": 2}).json()["session_id"]
    for _ in range(30):
        legal = client.get(f"/api/sessions/{sid}").json()["state"]["legal"]
        r = client.post(f"/api/sessions/{sid}/action", json={"action": "fold" if legal["fold"] else "check"}).json()
        ev = r["events"]
        assert ev and ev[0]["type"] == "action" and ev[0]["seat"] == 0  # your own move comes first
        for e in ev:
            t = e["table"]
            assert len(t["seats"]) == 6 and t["seats"][0]["cards"]  # you always see your cards
            hidden = [s for s in t["seats"][1:] if s["cards"] is not None]
            assert e["type"] == "hand_end" or not hidden  # bots' cards only show at showdown
        if any(e["type"] == "hand_end" for e in ev):
            assert any(e["type"] == "new_hand" for e in ev)
            return
    raise AssertionError("no hand finished in 30 actions")


def test_opening_deal_and_animated_autoplay_return_events():
    r = client.post("/api/sessions", json={"strategy": "directed_transfer", "partner_seat": 4}).json()
    assert r["events"][0]["type"] == "new_hand"  # the first deal is animated up to your turn
    sid = r["session_id"]
    r = client.post(f"/api/sessions/{sid}/autoplay", json={"hands": 1, "animate": True}).json()
    types = [e["type"] for e in r["events"]]
    # one hand, then the next is dealt up to your turn (rarely a walk: all fold to your big blind, one more hand ends)
    assert types.count("hand_end") >= 1 and types[-1] != "hand_end"
    assert any(e["type"] == "action" and e["seat"] == 0 for e in r["events"])  # your seat's auto moves are shown
    played = r["state"]["hands_played"]
    assert played == types.count("hand_end") and r["state"]["legal"]
    r = client.post(f"/api/sessions/{sid}/autoplay", json={"hands": 2}).json()
    assert r["events"] == [] and r["state"]["hands_played"] >= played + 2  # not animated unless asked


def test_top3_rank_alone_is_not_suspicious():
    # someone is always in the top 3 of 15 pairs: "suspected" also needs the risk above the warning line
    sid = client.post("/api/sessions", json={"strategy": "honest"}).json()["session_id"]
    client.post(f"/api/sessions/{sid}/autoplay", json={"hands": 40, "user_mode": "honest"})
    mine = next(p for p in STORE.get(sid).detector.pairs if p["you_and_partner"])
    mine |= {"rank": 2, "risk": 0.0005, "flagged": False, "watched": False}
    report = client.get(f"/api/sessions/{sid}/report").json()
    assert report["verdict"] == "clear" and report["headline"].startswith("Cleared")
    mine |= {"risk": 0.02, "watched": True}
    assert client.get(f"/api/sessions/{sid}/report").json()["verdict"] == "suspected"
