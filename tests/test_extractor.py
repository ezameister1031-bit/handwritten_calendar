"""日時抽出と修正指示のテスト（Gemini API なしで動く）

実行方法：  python -m pytest tests   または   python tests/test_extractor.py
"""
import sys
from datetime import date, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calendar_export import build_ics          # noqa: E402
from corrections import apply_correction        # noqa: E402
from extractor import extract_events, parse_line  # noqa: E402

REF = date(2026, 10, 28)   # 水曜日


def P(text):
    p = parse_line(text, REF)
    return p.date, p.start, p.end, p.title


def test_table1_examples():
    # 説明書の表1の例
    assert P("10/30 15:00 ゼミ") == (date(2026, 10, 30), time(15, 0), None, "ゼミ")
    assert P("11月5日 13時～14時半 MTG") == (date(2026, 11, 5), time(13, 0), time(14, 30), "MTG")
    assert P("来週火 3時 歯医者") == (date(2026, 11, 3), time(15, 0), None, "歯医者")
    assert P("明日 バイト 18-22") == (date(2026, 10, 29), time(18, 0), time(22, 0), "バイト")


def test_dates():
    assert P("2027/1/5 帰省")[0] == date(2027, 1, 5)
    assert P("1/10 成人式")[0] == date(2027, 1, 10)          # 過ぎた月日は翌年
    assert P("10/28 今日の予定")[0] == date(2026, 10, 28)
    assert P("11/5(木) 会議")[:1] + P("11/5(木) 会議")[3:] == (date(2026, 11, 5), "会議")
    assert P("１１／６　面談")[0] == date(2026, 11, 6)         # 全角
    assert P("明後日 買い物")[0] == date(2026, 10, 30)
    assert P("3日後 締切")[0] == date(2026, 10, 31)
    assert P("来月5日 支払い")[0] == date(2026, 11, 5)
    assert P("今週金曜 飲み会")[0] == date(2026, 10, 30)
    assert P("金 飲み会")[0] == date(2026, 10, 30)
    assert P("(土) 掃除")[0] == date(2026, 10, 31)
    assert P("次の水曜 定例")[0] == date(2026, 11, 4)
    assert P("再来週月曜 発表")[0] == date(2026, 11, 9)


def test_not_weekday():
    # 予定名に含まれる漢字を曜日と誤認しない
    assert P("月例会 10時")[0] is None and P("月例会 10時")[3] == "月例会"
    assert P("火災訓練")[0] is None


def test_times():
    assert P("10/30 午前10時 健診")[1] == time(10, 0)
    assert P("10/30 午後3時 面談")[1] == time(15, 0)
    assert P("10/30 3pm call")[1] == time(15, 0)
    assert P("10/30 9:30 授業")[1] == time(9, 30)
    assert P("10/30 15時30分 歯医者")[1] == time(15, 30)
    assert P("10/30 13:00-14:30 ゼミ")[1:3] == (time(13, 0), time(14, 30))
    assert P("10/30 1~3 ゼミ")[1:3] == (time(13, 0), time(15, 0))
    assert P("10/30 11時~1時 昼会")[1:3] == (time(11, 0), time(13, 0))
    assert P("10/30 15時~ 会議")[1:] == (time(15, 0), None, "会議")
    assert P("10/30 8時 朝練")[1] == time(8, 0)                # 8時は午前のまま


def test_spoken_style():
    # 音声入力（話し言葉）の書き起こし
    assert P("明日の15時からゼミ") == (date(2026, 10, 29), time(15, 0), None, "ゼミ")
    assert P("11月5日の1時から2時半までMTG") == (date(2026, 11, 5), time(13, 0), time(14, 30), "MTG")
    assert P("来週の火曜の3時に歯医者") == (date(2026, 11, 3), time(15, 0), None, "歯医者")
    assert P("金曜日の夜9時から飲み会") == (date(2026, 10, 30), time(21, 0), None, "飲み会")
    assert P("あさっての朝7時からバイト") == (date(2026, 10, 30), time(7, 0), None, "バイト")
    assert P("10/30 午後3時からゼミの予定")[3] == "ゼミ"
    assert P("11/2 歯医者に行く")[3] == "歯医者に行く"


def test_multiline_header():
    lines = ["11/2(月)", "10:00 授業", "15:00", "ゼミ発表", "夕方 買い物", "11/3 祝日"]
    evs = extract_events(lines, REF)
    got = [(e.date, e.start, e.title) for e in evs]
    assert got == [
        (date(2026, 11, 2), time(10, 0), "授業"),
        (date(2026, 11, 2), time(15, 0), "ゼミ発表"),
        (date(2026, 11, 2), None, "夕方 買い物"),
        (date(2026, 11, 3), None, "祝日"),
    ]


def test_text_only_after_event():
    evs = extract_events(["明日 バイト 18-22", "らくがき"], REF)
    assert evs[1].register is False and evs[1].title == "らくがき"


def test_no_date():
    evs = extract_events(["15:00 打ち合わせ"], REF)
    assert evs[0].date is None and evs[0].register is False


def _ev(text):
    return extract_events([text], REF)[0]


def test_corrections():
    ev = _ev("10/30 15:00 ゼミ")
    new, msg = apply_correction(ev, "3時じゃなくて4時", REF)
    assert new.start == time(16, 0)
    new, _ = apply_correction(ev, "1時半に変更", REF)
    assert new.start == time(13, 30)
    new, _ = apply_correction(ev, "11月6日に変更してください", REF)
    assert new.date == date(2026, 11, 6) and new.start == time(15, 0)
    new, _ = apply_correction(ev, "タイトルを研究会に変更", REF)
    assert new.title == "研究会"
    new, _ = apply_correction(ev, "ゼミじゃなくて研究会", REF)
    assert new.title == "研究会"
    new, _ = apply_correction(ev, "削除して", REF)
    assert new.register is False

    ev2 = _ev("11/5 13:00-14:30 MTG")
    new, _ = apply_correction(ev2, "2時に変更", REF)              # 長さ（1時間半）を保つ
    assert (new.start, new.end) == (time(14, 0), time(15, 30))
    new, _ = apply_correction(ev2, "終わりを5時に", REF)
    assert (new.start, new.end) == (time(13, 0), time(17, 0))
    new, _ = apply_correction(ev2, "2時半じゃなくて3時", REF)     # 終了時刻の訂正
    assert (new.start, new.end) == (time(13, 0), time(15, 0))


def test_ics():
    evs = extract_events(["15:00 日付なし", "10/30 15:00 ゼミ", "11/3 祝日"], REF)
    ics = build_ics(evs).decode()
    assert ics.count("BEGIN:VEVENT") == 2
    assert "DTSTART;TZID=Asia/Tokyo:20261030T150000" in ics
    assert "DTEND;TZID=Asia/Tokyo:20261030T160000" in ics
    assert "DTSTART;VALUE=DATE:20261103" in ics


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
