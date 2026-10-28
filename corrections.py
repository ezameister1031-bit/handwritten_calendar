"""音声（または文字）による修正指示を解釈して予定を書き換える

対応する言い方の例
- 「3時じゃなくて4時」          → 開始時刻を変更（終了時刻があれば同じ長さだけずらす）
- 「1時半に変更」              → 開始時刻を変更
- 「終わりを5時に」            → 終了時刻を変更
- 「11月6日に変更」「明日にして」 → 日付を変更
- 「タイトルを会議に変更」       → 予定名を変更
- 「ゼミじゃなくて研究会」       → 予定名の一部を置き換え
- 「削除」「消して」            → この予定を登録しない
"""
from __future__ import annotations

import copy
import re
from datetime import date, datetime, time, timedelta

from extractor import DEFAULT_PM_RANGE, Event, describe_event, normalize, parse_line, speak_time

_DELETE = re.compile(r"削除|消して|消す|取り消|キャンセル|いらない|要らない")
_NOT = re.compile(r"(?P<x>.+?)\s*(?:じゃなくて|ではなくて|じゃなく|ではなく|でなくて|でなく)\s*(?P<y>.+)")
_FIELD = re.compile(
    r"^(?P<f>タイトル|予定名|名前|件名|内容|日付|日にち|開始時刻|開始時間|開始|始まり|終了時刻|終了時間|終了|終わり)\s*(?:の時間|の時刻)?\s*(?:を|は|が)?\s*"
)
_TAIL = re.compile(r"\s*(?:に|へ)?\s*(?:変更|変えて|変える|直して|直す|修正|して|する|です|だよ|でお願い)\S*$")


def _strip_tail(s: str) -> str:
    s = s.strip(" 。.!！、")
    prev = None
    while prev != s:
        prev = s
        s = _TAIL.sub("", s).strip(" 。.!！、")
    return s


def _shift(t: time, delta: timedelta) -> time:
    dt = datetime.combine(date(2000, 1, 1), t) + delta
    if dt.date() != date(2000, 1, 1):
        return time(23, 59)
    return dt.time()


def apply_correction(ev: Event, command: str, ref: date,
                     pm_range: tuple[int, int] = DEFAULT_PM_RANGE) -> tuple[Event, str]:
    """修正指示を予定に適用する．戻り値：(新しい予定, 読み上げ用のメッセージ)"""
    new = copy.deepcopy(ev)
    cmd = normalize(command)
    if not cmd:
        return new, "修正の指示を聞き取れませんでした。"

    if _DELETE.search(cmd):
        new.register = False
        return new, f"{new.title or 'この予定'}は登録しないようにしました。"

    x = None
    m = _NOT.match(cmd)
    if m:
        x, rest = m["x"].strip(), m["y"]
    else:
        rest = cmd

    field = None
    fm = _FIELD.match(rest) or (_FIELD.match(x) if x else None)
    if fm:
        field = fm["f"]
        if _FIELD.match(rest):
            rest = rest[fm.end():]
    y = _strip_tail(rest)
    if not y:
        return new, "修正の内容を聞き取れませんでした。もう一度お願いします。"

    p = parse_line(y, ref, pm_range)
    changes = []

    if field in ("タイトル", "予定名", "名前", "件名", "内容"):
        new.title = y
        changes.append(f"予定名を{y}")
    elif field in ("日付", "日にち"):
        if p.date:
            new.date = p.date
            changes.append(f"日付を{p.date.month}月{p.date.day}日")
    elif field in ("終了", "終了時刻", "終了時間", "終わり"):
        if p.start:
            new.end = p.start
            changes.append(f"終了時刻を{speak_time(p.start)}")
    else:
        # 時刻
        if p.start:
            x_time = parse_line(x, ref, pm_range).start if x else None
            if field is None and x_time and new.end and x_time == new.end and not p.end:
                new.end = p.start
                changes.append(f"終了時刻を{speak_time(p.start)}")
            else:
                if p.end:
                    new.start, new.end = p.start, p.end
                elif new.start and new.end:
                    dur = datetime.combine(date(2000, 1, 1), new.end) - datetime.combine(date(2000, 1, 1), new.start)
                    new.start, new.end = p.start, _shift(p.start, dur)
                else:
                    new.start = p.start
                changes.append(f"時刻を{speak_time(new.start)}" + (f"から{speak_time(new.end)}" if new.end else ""))
        # 日付
        if p.date:
            new.date = p.date
            changes.append(f"日付を{p.date.month}月{p.date.day}日")
        # 日付も時刻もなければ予定名の変更とみなす
        if not p.start and not p.date:
            if x and x in new.title:
                new.title = new.title.replace(x, y)
            else:
                new.title = y
            changes.append(f"予定名を{new.title}")

    if not changes:
        return new, "どこを直せばよいか分かりませんでした。画面で直接修正してください。"
    new.register = True if new.date else new.register
    if new.date and "日付が見つかりません" in new.note:
        new.note = new.note.replace("日付が見つかりません", "").strip("／")
    return new, "、".join(changes) + "に変更しました。" + describe_event(new, for_speech=True) + "です。"
