"""アプリ内カレンダーの保存処理のテスト"""
import sys
from datetime import date, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calendar_store                          # noqa: E402
from extractor import extract_events           # noqa: E402

REF = date(2026, 10, 28)


def test_add_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(calendar_store, "STORE_PATH", tmp_path / "events.json")
    evs = extract_events(["15:00 日付なし", "10/30 15:00 ゼミ", "11/3 祝日"], REF)
    added, skipped = calendar_store.add(evs)
    assert len(added) == 2 and skipped == 0            # 日付なしは登録しない
    added2, skipped2 = calendar_store.add(evs)          # 同じ予定は二重に登録しない
    assert added2 == [] and skipped2 == 2

    recs = calendar_store.load()
    fc = calendar_store.to_fullcalendar(recs, {added[0][0]})
    assert fc[0]["start"] == "2026-10-30T15:00:00" and fc[0]["end"] == "2026-10-30T16:00:00"
    assert fc[0]["backgroundColor"] != fc[1]["backgroundColor"]   # 今回の予定は色が違う
    assert fc[1]["allDay"] is True and fc[1]["start"] == "2026-11-03"

    back = calendar_store.to_events(recs)
    assert (back[0].date, back[0].start, back[0].title) == (date(2026, 10, 30), time(15, 0), "ゼミ")

    calendar_store.delete(added[0][0])
    assert len(calendar_store.load()) == 1
    calendar_store.clear()
    assert calendar_store.load() == []
