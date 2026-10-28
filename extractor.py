"""認識した文字列から日付・時刻・予定名を取り出す（自前のルール）

- 日付  ：「10/30」「11月5日」「2026/11/5」「11/5(水)」
- 相対日付：「今日」「明日」「明後日」「3日後」「来月5日」「来週火」「今週金曜」「次の月曜」「(土)」
- 時刻  ：「15:00」「15時」「13時半」「15時30分」「午後3時」「3pm」
- 時間帯：「13:00~14:30」「13時〜14時半」「18-22」
ルール
- 年が書かれていなければ，基準日（撮影日）以降で最も近い日付にする．
- 午前・午後が書かれていない「1〜7時」は午後とみなす（範囲は変更可）．
- 日付だけの行は「見出し」とみなし，次の行以降の予定の日付として使う．
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

WEEKDAYS = "月火水木金土日"  # date.weekday() の 0〜6 に対応
DEFAULT_PM_RANGE = (1, 7)


@dataclass
class Event:
    date: date | None
    start: time | None
    end: time | None
    title: str
    source: str = ""            # 元になった行の文字列
    line_index: int | None = None
    register: bool = True       # カレンダーに登録するか
    note: str = ""              # 要確認の理由など


@dataclass
class LineParse:
    date: date | None = None
    start: time | None = None
    end: time | None = None
    title: str = ""
    raw: str = ""
    spans: list[tuple[str, str]] = field(default_factory=list)  # (種類, 一致した文字列) ※説明用


# ------------------------------------------------------------------ 正規化
def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)          # 全角数字・記号を半角に
    t = t.replace("〜", "~").replace("～", "~")
    t = re.sub(r"(?<=\d)\s*[ー−–—―]\s*(?=\d)", "-", t)  # 数字の間の長音記号などをハイフンに
    t = re.sub(r"\s+", " ", t)
    # 「夜9時」「朝9時」のような話し言葉を午前・午後に置き換える
    t = re.sub(r"(?:夜|晩|夕方)(?=\s*\d{1,2}\s*[時:])", "午後", t)
    t = re.sub(r"(?:朝)(?=\s*\d{1,2}\s*[時:])", "午前", t)
    return t.strip()


_MASK = "\x00"   # 日付・時刻として使った部分を塗りつぶす文字


def _mask(text: str, start: int, end: int) -> str:
    return text[:start] + _MASK * (end - start) + text[end:]


# ------------------------------------------------------------------ 日付
_DATE_ABS = re.compile(
    r"(?<!\d)(?:(?P<y>\d{4})\s*[/年.]\s*)?(?P<mo>\d{1,2})\s*(?:/|月)\s*(?P<d>\d{1,2})(?!\d)\s*日?"
    r"(?:\s*[(\[]\s*[月火水木金土日](?:曜日|曜)?\s*[)\]]"          # 11/5(水)
    r"|[月火水木金土日](?:曜日|曜)?(?![぀-ヿ一-鿿]))?"  # 11/5水
)
_DATE_REL = re.compile(r"今日|きょう|本日|明後日|あさって|明日|あした|あす")
_DATE_NDAYS = re.compile(r"(?<!\d)(?P<n>\d{1,2})\s*日後")
_DATE_NEXTMONTH = re.compile(r"来月\s*(?P<d>\d{1,2})\s*日")
_DATE_WEEKDAY = re.compile(
    r"(?P<prefix>再来週|来週|今週|次の|次)?\s*(?:の\s*)?(?P<open>[(\[])?\s*(?P<wd>[月火水木金土日])(?P<suf>曜日|曜)?\s*(?P<close>[)\]])?"
)


def infer_year(month: int, day: int, ref: date) -> date | None:
    """年が書かれていない日付を，基準日以降で最も近い日付にする．"""
    for year in (ref.year, ref.year + 1):
        try:
            d = date(year, month, day)
        except ValueError:
            continue
        if d >= ref:
            return d
    return None


def resolve_weekday(wd: str, prefix: str | None, ref: date) -> date:
    idx = WEEKDAYS.index(wd)
    monday = ref - timedelta(days=ref.weekday())
    if prefix == "今週":
        return monday + timedelta(days=idx)
    if prefix == "来週":
        return monday + timedelta(days=7 + idx)
    if prefix == "再来週":
        return monday + timedelta(days=14 + idx)
    delta = (idx - ref.weekday()) % 7
    if prefix in ("次", "次の") and delta == 0:
        delta = 7
    return ref + timedelta(days=delta)


def _find_date(text: str, ref: date) -> tuple[date | None, str, list[tuple[str, str]]]:
    """最初に見つかった日付を返し，その部分を空白で塗りつぶした文字列も返す．"""
    spans = []
    m = _DATE_ABS.search(text)
    if m:
        mo, d = int(m["mo"]), int(m["d"])
        if 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                found = date(int(m["y"]), mo, d) if m["y"] else infer_year(mo, d, ref)
            except ValueError:
                found = None
            if found:
                spans.append(("日付", m.group().strip()))
                return found, _mask(text, m.start(), m.end()), spans

    m = _DATE_NEXTMONTH.search(text)
    if m:
        y, mo = (ref.year + 1, 1) if ref.month == 12 else (ref.year, ref.month + 1)
        try:
            found = date(y, mo, int(m["d"]))
            spans.append(("日付", m.group()))
            return found, _mask(text, m.start(), m.end()), spans
        except ValueError:
            pass

    m = _DATE_REL.search(text)
    if m:
        offset = {"今日": 0, "きょう": 0, "本日": 0, "明日": 1, "あした": 1, "あす": 1,
                  "明後日": 2, "あさって": 2}[m.group()]
        spans.append(("日付", m.group()))
        return ref + timedelta(days=offset), _mask(text, m.start(), m.end()), spans

    m = _DATE_NDAYS.search(text)
    if m:
        spans.append(("日付", m.group()))
        return ref + timedelta(days=int(m["n"])), _mask(text, m.start(), m.end()), spans

    for m in _DATE_WEEKDAY.finditer(text):
        prefix, suf = m["prefix"], m["suf"]
        paren = m["open"] and m["close"]
        if not (prefix or suf or paren):
            # 「火」1文字だけの場合は，前後が空白・行頭行末・数字のときだけ曜日とみなす
            # （「月例会」「火災訓練」などを誤って曜日にしないため）
            s, e = m.start("wd"), m.end("wd")
            before_ok = s == 0 or text[s - 1] in " " + _MASK
            after_ok = e == len(text) or text[e] in " ~0123456789午" + _MASK
            if not (before_ok and after_ok):
                continue
        spans.append(("日付", m.group().strip()))
        return resolve_weekday(m["wd"], prefix, ref), _mask(text, m.start(), m.end()), spans

    return None, text, spans


# ------------------------------------------------------------------ 時刻
_AMPM = r"午前|午後|AM|PM|am|pm|a\.m\.|p\.m\."


def _time_pattern(p: str) -> str:
    return (rf"(?P<{p}ampm>{_AMPM})?\s*(?P<{p}h>\d{{1,2}})"
            rf"(?:\s*:\s*(?P<{p}m>\d{{2}})|\s*時\s*(?:(?P<{p}m2>\d{{1,2}})\s*分|(?P<{p}half>半))?|(?=\s*(?:am|pm|AM|PM)))"
            rf"(?:\s*(?P<{p}ampm2>am|pm|AM|PM))?")


_RANGE_SEP = r"\s*(?:~|-|から)\s*"
_TIME_RANGE = re.compile(
    _time_pattern("s") + _RANGE_SEP +
    rf"(?:{_time_pattern('e')}|(?P<ebare>\d{{1,2}})(?![\d/:]))"
)
_TIME_SINGLE = re.compile(_time_pattern("s") + r"(?:\s*(?:~|から))?")   # 「15時〜」のように開始だけの場合も
_TIME_BARE_RANGE = re.compile(r"(?<![\d/:.])(?P<bs>\d{1,2})\s*[-~]\s*(?P<be>\d{1,2})(?![\d/:時.])")


def _ampm_kind(*values: str | None) -> str | None:
    for v in values:
        if not v:
            continue
        v = v.lower().replace(".", "")
        if v in ("午後", "pm"):
            return "pm"
        if v in ("午前", "am"):
            return "am"
    return None


def _to_time(h: int, m: int, ampm: str | None, pm_range: tuple[int, int]) -> time | None:
    if not (0 <= h <= 24 and 0 <= m <= 59):
        return None
    if ampm == "pm" and h < 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0
    elif ampm is None and pm_range[0] <= h <= pm_range[1]:
        h += 12
    if h >= 24:
        return time(23, 59)
    return time(h, m)


def _minutes(g: dict, p: str) -> int:
    if g.get(f"{p}m"):
        return int(g[f"{p}m"])
    if g.get(f"{p}m2"):
        return int(g[f"{p}m2"])
    if g.get(f"{p}half"):
        return 30
    return 0


def _fix_end(start: time, end: time | None) -> time | None:
    """終了が開始より前になったら 12 時間後ろにずらす（例：13時〜2時 → 13時〜14時）．"""
    if end is None or end > start:
        return end
    if end.hour < 12 and end.hour + 12 > start.hour:
        return time(end.hour + 12, end.minute)
    return end


def _find_time(text: str, pm_range) -> tuple[time | None, time | None, str, list]:
    spans = []
    m = _TIME_RANGE.search(text)
    if m:
        g = m.groupdict()
        s_ampm = _ampm_kind(g["sampm"], g["sampm2"])
        start = _to_time(int(g["sh"]), _minutes(g, "s"), s_ampm, pm_range)
        if g["ebare"]:
            e_h, e_m, e_ampm = int(g["ebare"]), 0, None
        else:
            e_h, e_m, e_ampm = int(g["eh"]), _minutes(g, "e"), _ampm_kind(g["eampm"], g["eampm2"])
        if e_ampm is None and s_ampm == "pm":
            e_ampm = "pm"
        end = _to_time(e_h, e_m, e_ampm, pm_range)
        if start:
            spans.append(("時刻", m.group().strip()))
            return start, _fix_end(start, end), _mask(text, m.start(), m.end()), spans

    m = _TIME_SINGLE.search(text)
    if m:
        g = m.groupdict()
        start = _to_time(int(g["sh"]), _minutes(g, "s"), _ampm_kind(g["sampm"], g["sampm2"]), pm_range)
        if start:
            spans.append(("時刻", m.group().strip()))
            return start, None, _mask(text, m.start(), m.end()), spans

    m = _TIME_BARE_RANGE.search(text)
    if m:
        bs, be = int(m["bs"]), int(m["be"])
        if bs <= 24 and be <= 24:
            start = _to_time(bs, 0, None, pm_range)
            end = _to_time(be, 0, None, pm_range)
            if start:
                spans.append(("時刻", m.group().strip()))
                return start, _fix_end(start, end), _mask(text, m.start(), m.end()), spans

    return None, None, text, spans


# ------------------------------------------------------------------ 予定名
_BULLETS = "・･-*+□■○●◯◎☆★→>"


# 話し言葉（音声入力）で日付・時刻の前後に残る助詞など．例：「明日の15時からゼミ」→「の」「ゼミ」
_LEAD_WORDS = re.compile(r"^(?:(?:の|に|は|で|から|まで|より|ごろ|頃|くらい|ぐらい)(?=\s|$)\s*)+")
_TRAIL_WORDS = re.compile(r"(?:\s*(?:がある|があります|あり|です|の予定|予定|を入れて|入れて|入れといて|をお願い))+$")


def _clean_title(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    t = t.strip(" :;,、。.~-()[]「」『』" + _BULLETS)
    t = re.sub(r"\(\s*\)|\[\s*\]", "", t)   # 中身が消えた括弧
    t = re.sub(r"\s+", " ", t).strip(" :;,、。")
    prev = None
    while prev != t:                              # 「の まで MTG」→「MTG」
        prev = t
        t = _LEAD_WORDS.sub("", t).strip(" :;,、。")
        stripped = _TRAIL_WORDS.sub("", t).strip(" :;,、。")
        t = stripped or t                         # 予定名が空になる場合は残す
    return t


# ------------------------------------------------------------------ 1行の解析
def parse_line(text: str, ref: date, pm_range: tuple[int, int] = DEFAULT_PM_RANGE) -> LineParse:
    norm = normalize(text)
    d, rest, s1 = _find_date(norm, ref)
    st, en, rest, s2 = _find_time(rest, pm_range)
    # 話し言葉で日付・時刻の直後に付く助詞や，時刻の直前の「夜」などを取り除く
    # 例：「明日の夜7時から飲み会」→ 日付・時刻を除いた残りが「飲み会」になる
    rest = re.sub(r"(?:夕方|朝|昼|夜|晩)(?=\x00)", "", rest)
    rest = re.sub(r"\x00+(?:\s*(?:までに|から|まで|より|ごろ|くらい|ぐらい|頃|の|に|は|で)(?![ぁ-ゖ]))*", " ", rest)
    rest = rest.replace(_MASK, " ")
    return LineParse(date=d, start=st, end=en, title=_clean_title(rest), raw=text, spans=s1 + s2)


# ------------------------------------------------------------------ 複数行から予定を作る
def extract_events(lines: list[str], ref: date, pm_range: tuple[int, int] = DEFAULT_PM_RANGE) -> list[Event]:
    events: list[Event] = []
    current_date: date | None = None   # 直前に書かれた日付
    from_header = False                # current_date が「日付だけの行（見出し）」から来たか

    for i, line in enumerate(lines):
        p = parse_line(line, ref, pm_range)
        if p.date:
            current_date = p.date
            from_header = not p.title and p.start is None

        # 日付だけの行（見出し）や空行
        if not p.title and p.start is None:
            continue

        # 文字だけの行
        if p.date is None and p.start is None:
            prev = events[-1] if events else None
            if prev and not prev.title and prev.line_index == i - 1:
                prev.title = p.title                     # 前の行（時刻だけ）の続き
                prev.source += " / " + line
                continue
            ev = Event(current_date, None, None, p.title, line, i)   # 見出しの日の終日予定とみなす
            if current_date is None:
                ev.register, ev.note = False, "日付が見つかりません"
            elif not from_header:
                ev.register, ev.note = False, "日付・時刻が書かれていない行です（メモの可能性）"
            events.append(ev)
            continue

        ev = Event(p.date or current_date, p.start, p.end, p.title, line, i)
        if ev.date is None:
            ev.register, ev.note = False, "日付が見つかりません"
        events.append(ev)

    for ev in events:
        if not ev.title:
            ev.note = (ev.note + "／" if ev.note else "") + "予定名がありません"
    return events


# ------------------------------------------------------------------ 表示・読み上げ用
def format_date_ja(d: date | None) -> str:
    if d is None:
        return "日付不明"
    return f"{d.month}月{d.day}日（{WEEKDAYS[d.weekday()]}）"


def speak_time(t: time) -> str:
    return f"{t.hour}時" + (f"{t.minute}分" if t.minute else "")


def describe_event(ev: Event, for_speech: bool = False) -> str:
    d = ev.date
    if d is None:
        when = "日付不明"
    else:
        when = f"{d.month}月{d.day}日" if for_speech else format_date_ja(d)
    if ev.start:
        when += " " + speak_time(ev.start)
        if ev.end:
            when += "から" + speak_time(ev.end)
    return f"{when}、{ev.title or '予定名なし'}"


def to_datetime(d: date, t: time) -> datetime:
    return datetime.combine(d, t)
