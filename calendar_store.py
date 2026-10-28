"""アプリ内カレンダーの予定を JSON ファイルに保存する

保存先：data/calendar_events.json
- 自分の PC で動かす場合：アプリを閉じても予定は残る．
- Streamlit Cloud の場合：アプリが再起動すると消える．また，使っている人全員で同じカレンダーを共有する．
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path

from extractor import Event

STORE_PATH = Path(__file__).resolve().parent / "data" / "calendar_events.json"


def load() -> list[dict]:
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save(records: list[dict]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STORE_PATH)


def _hm(t: time | None) -> str | None:
    return t.strftime("%H:%M") if t else None


def to_record(ev: Event) -> dict:
    return {
        "id": uuid.uuid4().hex,
        "title": ev.title or "予定",
        "date": ev.date.isoformat(),
        "start": _hm(ev.start),
        "end": _hm(ev.end),
        "source": ev.source,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def _same(a: dict, b: dict) -> bool:
    return (a["date"], a["start"], a["end"], a["title"]) == (b["date"], b["start"], b["end"], b["title"])


def add(events: list[Event]) -> tuple[list[tuple[str, Event]], int]:
    """予定を追加する．同じ予定（日付・時刻・予定名が同じ）が既にあれば追加しない．
    戻り値：([(追加した予定の id, 予定), ...], 重複でスキップした件数)"""
    records = load()
    added, skipped = [], 0
    for ev in events:
        if not ev.register or ev.date is None:
            continue
        rec = to_record(ev)
        if any(_same(rec, r) for r in records):
            skipped += 1
            continue
        records.append(rec)
        added.append((rec["id"], ev))
    save(records)
    return added, skipped


def delete(event_id: str) -> dict | None:
    records = load()
    hit = next((r for r in records if r["id"] == event_id), None)
    if hit:
        save([r for r in records if r["id"] != event_id])
    return hit


def clear() -> None:
    save([])


def to_events(records: list[dict]) -> list[Event]:
    """保存した予定を Event に戻す（.ics 書き出し用）．"""
    out = []
    for r in records:
        out.append(Event(
            date=date.fromisoformat(r["date"]),
            start=time.fromisoformat(r["start"]) if r.get("start") else None,
            end=time.fromisoformat(r["end"]) if r.get("end") else None,
            title=r.get("title", ""), source=r.get("source", ""),
        ))
    return out


def to_fullcalendar(records: list[dict], highlight_ids: set[str] = frozenset()) -> list[dict]:
    """FullCalendar（streamlit-calendar）用の形式に変換する．今回追加した予定は色を変える．"""
    out = []
    for r in records:
        color = "#E8833A" if r["id"] in highlight_ids else "#3B6FB6"
        item = {"id": r["id"], "title": r["title"], "backgroundColor": color, "borderColor": color,
                "extendedProps": {"source": r.get("source", "")}}
        if r.get("start"):
            start = datetime.fromisoformat(f'{r["date"]}T{r["start"]}')
            end = datetime.fromisoformat(f'{r["date"]}T{r["end"]}') if r.get("end") else start + timedelta(hours=1)
            if end <= start:
                end = start + timedelta(hours=1)
            item.update(start=start.isoformat(), end=end.isoformat(), allDay=False)
        else:
            item.update(start=r["date"], allDay=True)
        out.append(item)
    return out
