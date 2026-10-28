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

from calendar_export import build_ics
from corrections import apply_correction
from extractor import Event, describe_event, extract_events, parse_line
from ocr_gemini import DEFAULT_MODEL, GeminiError, make_client, ocr_lines, transcribe_audio
from preprocess import decode_image, encode_png, preprocess

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
ss.setdefault("ics", None)
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
    api_key = st.text_input("Gemini API キー", value=_secret_key() or os.environ.get("GEMINI_API_KEY", ""),
                            type="password", help="Google AI Studio で発行したキー")
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
    page = f"""
<div style="font-family:sans-serif;display:flex;gap:10px;align-items:center">
  <button id="b" style="padding:6px 12px;border-radius:6px;border:1px solid #bbb;background:#fff;cursor:pointer">▶ 読み上げ</button>
  <span style="font-size:14px;color:#333">{html.escape(text)}</span>
</div>
<script>
const text = {json.dumps(text).replace("</", "<\\/")};
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
st.caption("手帳やメモを撮影すると，予定を読み取ってカレンダー用のファイル（.ics）を作ります．")

# ---- ① 画像の入力
st.subheader("1. 手書きメモの画像")
mode = st.radio("入力方法", ["カメラで撮影", "画像ファイルを選択"], horizontal=True, label_visibility="collapsed")
if mode == "カメラで撮影":
    upload = st.camera_input("メモを撮影してください")
else:
    upload = st.file_uploader("画像ファイル（JPEG / PNG）", type=["jpg", "jpeg", "png"])

if st.button("読み取る", type="primary", disabled=upload is None):
    try:
        img = decode_image(upload.getvalue())
        res = preprocess(img, do_warp=do_warp, do_shadow=do_shadow, do_binarize=do_bin)
        with st.spinner("Gemini で手書き文字を認識しています…"):
            lines = ocr_lines(make_client(api_key), model, encode_png(res.final))
        events = extract_events([ln["text"] for ln in lines], ref_date, tuple(pm_range))
        ss.update(lines=lines, final_img=res.final, steps=res.steps, base_events=events,
                  editor_ver=ss.editor_ver + 1, ics=None, log=[])
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
if ss.final_img is not None:
    st.subheader("2. 認識結果")
    used = {e.line_index for e in ss.base_events if e.line_index is not None}
    c1, c2 = st.columns([1, 1])
    c1.image(annotate(ss.final_img, ss.lines, used), caption="緑：予定として使った行　灰：使わなかった行",
             width="stretch")
    rows = []
    for i, ln in enumerate(ss.lines):
        p = parse_line(ln["text"], ref_date, tuple(pm_range))
        rows.append({"行": i + 1, "認識した文字": ln["text"],
                     "抽出した部分": "　".join(f"{k}:{v}" for k, v in p.spans) or "－",
                     "残り（予定名）": p.title or "－"})
    c2.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---- ③ 確認・修正
current: list[Event] = []
if ss.final_img is not None or ss.base_events:
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
            ss.update(base_events=updated, editor_ver=ss.editor_ver + 1, speech=msg, ics=None,
                      speech_id=ss.speech_id + 1)
            ss.log.append(f"「{command}」→ {msg}")
            st.rerun()

        vc1, vc2 = st.columns(2)
        with vc1:
            audio = st.audio_input("マイクで話す")
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
    if st.button("カレンダー用ファイル（.ics）を作成", type="primary", disabled=not targets):
        ss.ics = build_ics(current)
        names = "。".join(describe_event(e, for_speech=True) for e in targets[:5])
        ss.speech_id += 1
        ss.speech = f"{names}。以上{len(targets)}件を登録しました。"
    if ss.ics:
        st.download_button("ダウンロード（schedule.ics）", ss.ics, file_name="schedule.ics", mime="text/calendar")
        st.caption("Google カレンダー：設定 → インポート/エクスポート → ファイルを選択 → インポート．"
                   "iPhone・Outlook ではファイルを開くと取り込めます．")

# ---- 読み上げ
if ss.speech:
    st.divider()
    speak(ss.speech)
