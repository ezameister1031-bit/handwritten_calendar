"""手書きスケジュール → カレンダー登録アプリ（Streamlit）

起動：  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from datetime import date

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import calendar_store
from calendar_export import build_ics
from corrections import apply_correction
from extractor import Event, describe_event, extract_events, parse_line
from ocr_gemini import DEFAULT_MODEL, GeminiError, make_client, ocr_lines, transcribe_audio, transcribe_schedule
from preprocess import decode_image, encode_png, preprocess

try:
    from streamlit_calendar import calendar as st_calendar
except ImportError:          # ライブラリが入っていない場合
    st_calendar = None

st.set_page_config(page_title="手書きスケジュール → カレンダー", layout="wide")

# ------------------------------------------------------------------ 状態
ss = st.session_state
ss.setdefault("base_events", [])     # 表の元データ（読み取り・音声修正のときだけ更新）
ss.setdefault("editor_ver", 0)       # 表を作り直すときに増やす
ss.setdefault("lines", [])
ss.setdefault("final_img", None)
ss.setdefault("steps", {})
ss.setdefault("speech", None)        # 読み上げる文
ss.setdefault("speech_id", 0)        # 読み上げる文の番号（新しい文だけ自動で読み上げる）
ss.setdefault("spoken_id", 0)
ss.setdefault("last_audio", None)
ss.setdefault("cal_ver", 0)          # カレンダーを作り直すときに増やす
ss.setdefault("highlight_ids", set())  # 直前に登録した予定（色を変えて表示）
ss.setdefault("cal_initial", None)   # カレンダーで最初に表示する日
ss.setdefault("log", [])             # 修正の履歴


def _secret_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        return "" if "ここに" in key else key      # 書き換え前の見本のままなら未設定とみなす
    except Exception:
        return ""


# ------------------------------------------------------------------ サイドバー
with st.sidebar:
    st.header("設定")
    # APIキーは画面（ブラウザ）に送らない．secrets.toml / Streamlit Cloud の Secrets / 環境変数から読み込む
    api_key = _secret_key() or os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        st.success("Gemini API キー：設定済み")
    else:
        st.error("Gemini API キーが設定されていません．"
                 ".streamlit/secrets.toml（Streamlit Cloud ではアプリ設定の Secrets）に "
                 "GEMINI_API_KEY を書いてください．")
    model = st.text_input("モデル名", value=DEFAULT_MODEL,
                          help="エラーになる場合は Google AI Studio で使えるモデル名を確認して変更してください")
    ref_date = st.date_input("基準日（撮影日）", value=date.today(),
                             help="「明日」「来週火」などはこの日を基準に計算します")
    pm_range = st.slider("午後とみなす時刻（午前・午後の記載がない場合）", 0, 11, (1, 7))

    st.subheader("前処理")
    do_warp = st.checkbox("台形補正（紙の四隅を検出）", value=True)
    do_shadow = st.checkbox("影の除去", value=True)
    do_bin = st.checkbox("二値化", value=False, help="鉛筆など薄い字では精度が下がることがあります")

    st.subheader("音声")
    tts_on = st.checkbox("結果を読み上げる", value=True)
    tts_rate = st.slider("読み上げの速さ", 0.6, 1.6, 1.0, 0.1)


# ------------------------------------------------------------------ 補助関数
def speak(text: str) -> None:
    """Web Speech API で読み上げる（ブラウザの機能を使用）．"""
    if not text:
        return
    auto = ""
    if tts_on and ss.spoken_id != ss.speech_id:   # 新しい文のときだけ自動で読み上げる
        auto = "setTimeout(say, 300);"
        ss.spoken_id = ss.speech_id
    safe_text = json.dumps(text).replace("</", "<\\/")   # </script> での割り込みを防ぐ
    page = f"""
<div style="font-family:sans-serif;display:flex;gap:10px;align-items:center">
  <button id="b" style="white-space:nowrap;padding:6px 12px;border-radius:6px;border:1px solid #bbb;background:#fff;cursor:pointer">▶ 読み上げ</button>
  <span style="font-size:14px;color:#333">{html.escape(text)}</span>
</div>
<script>
const text = {safe_text};
function say() {{
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "ja-JP"; u.rate = {tts_rate};
  const v = speechSynthesis.getVoices().find(v => v.lang && v.lang.startsWith("ja"));
  if (v) u.voice = v;
  speechSynthesis.cancel(); speechSynthesis.speak(u);
}}
document.getElementById("b").onclick = say;
{auto}
// {ss.speech_id}
</script>"""
    if hasattr(st, "iframe"):          # Streamlit 1.5x 以降
        st.iframe(page, height=56)
    else:
        components.html(page, height=56)


def events_to_df(events: list[Event]) -> pd.DataFrame:
    return pd.DataFrame([{
        "登録": e.register, "日付": e.date, "開始": e.start, "終了": e.end,
        "予定名": e.title, "元の行": e.source, "確認": e.note,
    } for e in events], columns=["登録", "日付", "開始", "終了", "予定名", "元の行", "確認"])


def _val(v):
    return None if v is None or (not isinstance(v, str) and pd.isna(v)) else v


def df_to_events(df: pd.DataFrame, base: list[Event]) -> list[Event]:
    out = []
    for i, row in enumerate(df.to_dict("records")):
        d = _val(row.get("日付"))
        if isinstance(d, pd.Timestamp):
            d = d.date()
        ev = Event(date=d, start=_val(row.get("開始")), end=_val(row.get("終了")),
                   title=_val(row.get("予定名")) or "", source=_val(row.get("元の行")) or "",
                   line_index=base[i].line_index if i < len(base) else None,
                   register=bool(_val(row.get("登録"))), note=_val(row.get("確認")) or "")
        out.append(ev)
    return out


def annotate(img: np.ndarray, lines: list[dict], used: set[int]) -> np.ndarray:
    """認識した行を枠で囲み，番号を付ける（予定に使った行は緑，それ以外は灰色）．"""
    out = img.copy()
    h, w = out.shape[:2]
    for i, ln in enumerate(lines):
        box = ln.get("box_2d")
        if not box:
            continue
        y0, x0, y1, x1 = [int(v) for v in box]
        p0 = (int(x0 / 1000 * w), int(y0 / 1000 * h))
        p1 = (int(x1 / 1000 * w), int(y1 / 1000 * h))
        color = (60, 170, 60) if i in used else (150, 150, 150)
        cv2.rectangle(out, p0, p1, color, 3)
        label = str(i + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(out, (p0[0], max(p0[1] - th - 10, 0)), (p0[0] + tw + 10, max(p0[1], th + 10)), color, -1)
        cv2.putText(out, label, (p0[0] + 5, max(p0[1] - 5, th + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def show_error(e: Exception) -> None:
    st.error(str(e))


# ------------------------------------------------------------------ 画面
st.title("手書きスケジュール → カレンダー登録")
st.caption("手帳やメモを撮影するか，予定を話すと，予定を読み取ってカレンダーに登録します．")

# ---- ① 予定の入力（画像 または 音声）
st.subheader("1. 予定の入力")
mode = st.radio("入力方法", ["カメラで撮影", "画像ファイルを選択", "音声で入力"], horizontal=True,
                label_visibility="collapsed")
if mode == "カメラで撮影":
    upload = st.camera_input("メモを撮影してください")
elif mode == "画像ファイルを選択":
    upload = st.file_uploader("画像ファイル（JPEG / PNG）", type=["jpg", "jpeg", "png"])
else:
    upload = st.audio_input("予定を話してください", key="dictation")
    st.caption("例：「明日の午後3時からゼミ。11月5日の1時から2時半までミーティング。来週の火曜の3時に歯医者。」"
               "　複数の予定を続けて話せます．")

read_clicked = st.button("読み取る", type="primary", disabled=upload is None)

if read_clicked and mode == "音声で入力":
    try:
        with st.spinner("Gemini で音声を書き起こしています…"):
            texts = transcribe_schedule(make_client(api_key), model, upload.getvalue(), upload.type or "audio/wav")
        lines = [{"text": t, "box_2d": None} for t in texts]
        events = extract_events(texts, ref_date, tuple(pm_range))
        ss.update(lines=lines, final_img=None, steps={}, base_events=events,
                  editor_ver=ss.editor_ver + 1, log=[])
        ss.speech_id += 1
        ss.speech = (f"{len(events)}件の予定を聞き取りました。内容を確認してください。"
                     if events else "予定を聞き取れませんでした。もう一度話してください。")
    except GeminiError as e:
        show_error(e)

elif read_clicked:
    try:
        img = decode_image(upload.getvalue())
        res = preprocess(img, do_warp=do_warp, do_shadow=do_shadow, do_binarize=do_bin)
        with st.spinner("Gemini で手書き文字を認識しています…"):
            lines = ocr_lines(make_client(api_key), model, encode_png(res.final))
        events = extract_events([ln["text"] for ln in lines], ref_date, tuple(pm_range))
        ss.update(lines=lines, final_img=res.final, steps=res.steps, base_events=events,
                  editor_ver=ss.editor_ver + 1, log=[])
        if do_warp and not res.paper_found:
            st.info("紙の四隅を検出できなかったため，台形補正は行いませんでした．")
        ss.speech_id += 1
        ss.speech = (f"{len(events)}件の予定が見つかりました。内容を確認してください。"
                     if events else "予定を読み取れませんでした。")
    except (GeminiError, ValueError) as e:
        show_error(e)

if ss.steps:
    with st.expander("前処理の各段階を見る"):
        cols = st.columns(len(ss.steps))
        for col, (name, im) in zip(cols, ss.steps.items()):
            col.image(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB), caption=name,
                      width="stretch")

# ---- ② 認識結果
from_image = ss.final_img is not None
if from_image or ss.lines:
    st.subheader("2. 認識結果")
    rows = []
    for i, ln in enumerate(ss.lines):
        p = parse_line(ln["text"], ref_date, tuple(pm_range))
        rows.append({"行": i + 1, ("認識した文字" if from_image else "書き起こした文"): ln["text"],
                     "抽出した部分": "　".join(f"{k}:{v}" for k, v in p.spans) or "－",
                     "残り（予定名）": p.title or "－"})
    if from_image:
        used = {e.line_index for e in ss.base_events if e.line_index is not None}
        c1, c2 = st.columns([1, 1])
        c1.image(annotate(ss.final_img, ss.lines, used), caption="緑：予定として使った行　灰：使わなかった行",
                 width="stretch")
        c2.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("音声から書き起こした予定")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---- ③ 確認・修正
current: list[Event] = []
if ss.final_img is not None or ss.lines or ss.base_events:
    st.subheader("3. 予定の確認・修正")
    st.caption("表のセルをクリックすると直接修正できます．行の追加・削除もできます．")
    edited = st.data_editor(
        events_to_df(ss.base_events),
        key=f"editor_{ss.editor_ver}",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "登録": st.column_config.CheckboxColumn(width="small"),
            "日付": st.column_config.DateColumn(format="YYYY/MM/DD"),
            "開始": st.column_config.TimeColumn(format="HH:mm", step=300),
            "終了": st.column_config.TimeColumn(format="HH:mm", step=300),
            "予定名": st.column_config.TextColumn(width="medium"),
            "元の行": st.column_config.TextColumn(disabled=True),
            "確認": st.column_config.TextColumn(disabled=True),
        },
    )
    current = df_to_events(edited, ss.base_events)

    if current:
        st.markdown("**音声で修正する**")
        labels = [f"{i + 1}: {describe_event(e)}" for i, e in enumerate(current)]
        target = st.selectbox("修正する予定", range(len(current)), format_func=lambda i: labels[i])
        st.caption("例：「3時じゃなくて4時」「1時半に変更」「終わりを5時に」「11月6日に変更」「タイトルを研究会に変更」「削除」")

        def run_correction(command: str) -> None:
            new, msg = apply_correction(current[target], command, ref_date, tuple(pm_range))
            updated = list(current)
            updated[target] = new
            ss.update(base_events=updated, editor_ver=ss.editor_ver + 1, speech=msg,
                      speech_id=ss.speech_id + 1)
            ss.log.append(f"「{command}」→ {msg}")
            st.rerun()

        vc1, vc2 = st.columns(2)
        with vc1:
            audio = st.audio_input("マイクで話す", key="correction_audio")
            if audio is not None:
                data = audio.getvalue()
                digest = hashlib.md5(data).hexdigest()
                if digest != ss.last_audio:          # 同じ録音を二重に処理しない
                    ss.last_audio = digest
                    try:
                        with st.spinner("音声を書き起こしています…"):
                            text = transcribe_audio(make_client(api_key), model, data, audio.type or "audio/wav")
                        run_correction(text)
                    except GeminiError as e:
                        show_error(e)
        with vc2:
            typed = st.text_input("文字で入力（マイクが使えない場合）")
            if st.button("この指示で修正", disabled=not typed):
                run_correction(typed)

        if ss.log:
            with st.expander("修正の履歴", expanded=True):
                for line in reversed(ss.log):
                    st.write(line)

# ---- ④ 登録
if current:
    st.subheader("4. カレンダーに登録")
    targets = [e for e in current if e.register and e.date]
    st.write(f"登録する予定：{len(targets)}件")
    if st.button("カレンダーに登録", type="primary", disabled=not targets):
        added, skipped = calendar_store.add(current)
        ss.highlight_ids = {i for i, _ in added}
        ss.cal_initial = min(e.date for e in targets).isoformat()   # 登録した月を表示する
        ss.cal_ver += 1
        ss.speech_id += 1
        if added:
            names = "。".join(describe_event(e, for_speech=True) for _, e in added[:5])
            ss.speech = f"{names}。以上{len(added)}件をカレンダーに登録しました。"
            if skipped:
                ss.speech += f"{skipped}件は登録済みのため追加しませんでした。"
        else:
            ss.speech = "すべて登録済みの予定でした。"

# ---- 読み上げ
if ss.speech:
    speak(ss.speech)

# ---- ⑤ カレンダー
st.subheader("5. カレンダー")
records = calendar_store.load()

if st_calendar is None:
    st.warning("カレンダーの表示には streamlit-calendar が必要です．"
               "`pip install -r requirements.txt` を実行してください．")
else:
    st.caption("オレンジ：今回登録した予定　青：以前に登録した予定　（予定をクリックすると詳細と削除ボタンが表示されます）")
    options = {
        "initialView": "dayGridMonth",
        "initialDate": ss.cal_initial or ref_date.isoformat(),
        "locale": "ja",
        "headerToolbar": {"left": "prev,next today", "center": "title",
                          "right": "dayGridMonth,timeGridWeek,listMonth"},
        "buttonText": {"today": "今日", "month": "月", "week": "週", "list": "一覧"},
        "height": 680,
        "dayMaxEvents": 3,
        "nowIndicator": True,
        "slotMinTime": "06:00:00",
        "eventTimeFormat": {"hour": "2-digit", "minute": "2-digit", "hour12": False},
        "slotLabelFormat": {"hour": "2-digit", "minute": "2-digit", "hour12": False},
        "allDayText": "終日",
        "noEventsText": "予定はありません",
    }
    state = st_calendar(
        events=calendar_store.to_fullcalendar(records, ss.highlight_ids),
        options=options,
        custom_css=".fc-event-title, .fc-event-time { font-size: 0.85em; } .fc-toolbar-title { font-size: 1.3em; }",
        callbacks=["eventClick"],
        key=f"calendar_{ss.cal_ver}",
    )

    # 予定をクリックしたときの詳細表示
    clicked = (state or {}).get("eventClick", {}).get("event") if (state or {}).get("callback") == "eventClick" else None
    rec = next((r for r in records if clicked and r["id"] == clicked.get("id")), None)
    if rec:
        ev = calendar_store.to_events([rec])[0]
        with st.container(border=True):
            st.markdown(f"**{html.escape(rec['title'])}**　{describe_event(ev).split('、')[0]}")
            if rec.get("source"):
                st.caption(f"手書きメモの元の行：{rec['source']}")
            if st.button("この予定を削除"):
                calendar_store.delete(rec["id"])
                ss.cal_ver += 1
                ss.speech_id += 1
                ss.speech = f"{rec['title']}を削除しました。"
                st.rerun()

c1, c2 = st.columns(2)
with c1:
    st.download_button(f"すべての予定を .ics で書き出す（{len(records)}件）",
                       build_ics(calendar_store.to_events(records)) if records else b"",
                       file_name="schedule.ics", mime="text/calendar", disabled=not records,
                       help="Google カレンダー・iPhone・Outlook に取り込めるファイルです")
with c2:
    with st.popover("すべての予定を削除", disabled=not records):
        st.write("カレンダーの予定をすべて削除します．元に戻せません．")
        if st.button("削除する", type="primary"):
            calendar_store.clear()
            ss.update(cal_ver=ss.cal_ver + 1, highlight_ids=set(), speech_id=ss.speech_id + 1,
                      speech="カレンダーの予定をすべて削除しました。")
            st.rerun()
