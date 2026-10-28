"""予定を iCalendar 形式（.ics）で書き出す

.ics ファイルは Google カレンダー・iPhone のカレンダー・Outlook などに取り込める．
Google カレンダー：設定 → インポート/エクスポート → ファイルを選択 → インポート
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from icalendar import Calendar, Event as ICalEvent

from extractor import Event

TZ = ZoneInfo("Asia/Tokyo")
DEFAULT_DURATION = timedelta(hours=1)   # 終了時刻がない場合の長さ


def build_ics(events: list[Event]) -> bytes:
    cal = Calendar()
    cal.add("prodid", "-//Handwritten Schedule to Calendar//JA")
    cal.add("version", "2.0")
    cal.add("x-wr-timezone", "Asia/Tokyo")

    for ev in events:
        if not ev.register or ev.date is None:
            continue
        ve = ICalEvent()
        ve.add("uid", f"{uuid.uuid4()}@handwritten-calendar")
        ve.add("dtstamp", datetime.now(timezone.utc))
        ve.add("summary", ev.title or "予定")
        if ev.start:
            start = datetime.combine(ev.date, ev.start, tzinfo=TZ)
            end = datetime.combine(ev.date, ev.end, tzinfo=TZ) if ev.end else start + DEFAULT_DURATION
            if end <= start:
                end = start + DEFAULT_DURATION
            ve.add("dtstart", start)
            ve.add("dtend", end)
        else:                                   # 時刻がなければ終日の予定
            ve.add("dtstart", ev.date)
            ve.add("dtend", ev.date + timedelta(days=1))
        if ev.source:
            ve.add("description", f"手書きメモから登録：{ev.source}")
        cal.add_component(ve)
    return cal.to_ical()
