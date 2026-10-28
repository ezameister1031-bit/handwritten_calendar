"""Gemini API を使った手書き文字認識と音声の書き起こし

方針：Gemini には「書かれている文字をそのまま書き起こす」ことだけを頼み，
日付・時刻の解釈は extractor.py（自前のルール）で行う．
比較用に「Gemini に予定の抽出までまとめて任せる」関数も用意している（評価で使用）．
"""
from __future__ import annotations

import json
from datetime import date

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

DEFAULT_MODEL = "gemini-2.5-flash"


class OcrLine(BaseModel):
    text: str = Field(description="その行に書かれている文字（書かれたとおり）")
    box_2d: list[int] = Field(description="行を囲む矩形 [ymin, xmin, ymax, xmax]．0〜1000 に正規化した座標")


class BaselineEvent(BaseModel):
    date: str = Field(description="YYYY-MM-DD．不明なら空文字")
    start: str = Field(description="HH:MM（24時間表記）．なければ空文字")
    end: str = Field(description="HH:MM（24時間表記）．なければ空文字")
    title: str


OCR_PROMPT = """この画像は手書きのスケジュール（手帳・メモ・ホワイトボードなど）です．
書かれている文字を，上から順に1行ずつ正確に書き起こしてください．

- 日付や時刻を解釈・補完・言い換えせず，書かれている文字をそのまま返してください（例：「来週火」は「来週火」のまま）．
- 数字・記号（/ : ~ - など）も書かれたとおりに返してください．
- 各行について，その行を囲む矩形を box_2d = [ymin, xmin, ymax, xmax]（画像の幅と高さを 0〜1000 に正規化した座標）で返してください．
- 取り消し線で消された文字や，予定と関係のない落書きは省いてください．
"""

BASELINE_PROMPT = """この画像は手書きのスケジュールです．書かれている予定をすべて抽出してください．
今日の日付は {today} です．年が書かれていない場合や「明日」「来週火曜」などは，この日付を基準に具体的な日付にしてください．
時刻は24時間表記（HH:MM）にしてください．"""

DICTATION_PROMPT = """この音声では，日本語で自分の予定を話しています．話された予定を書き起こしてください．

- 予定ごとに1つの文字列に分け，話された順に並べてください．
- 数字は算用数字で書いてください（例：「じゅうごじ」→「15時」，「さんじはん」→「3時半」，「じゅういちがつ いつか」→「11月5日」）．
- 「明日」「来週の火曜」などの言葉は，具体的な日付に直さず，話されたとおりに書いてください．
- 「えー」「あの」などのつなぎ言葉や，「予定を入れて」などの依頼の言葉は省いてください．
- 予定が話されていない場合は，空の配列を返してください．
"""

TRANSCRIBE_PROMPT = "この音声は日本語で予定の修正を指示しています．話された内容をそのまま文字に書き起こしてください．書き起こした文だけを返してください．"


class GeminiError(RuntimeError):
    pass


def make_client(api_key: str) -> genai.Client:
    if not api_key:
        raise GeminiError("Gemini API キーが設定されていません．.streamlit/secrets.toml を確認してください．")
    return genai.Client(api_key=api_key)


def _generate(client: genai.Client, model: str, contents, config: types.GenerateContentConfig):
    try:
        return client.models.generate_content(model=model, contents=contents, config=config)
    except Exception as e:  # API キーの誤り・モデル名の誤り・通信エラーなど
        msg = str(e)
        if "API key" in msg or "API_KEY" in msg:
            hint = "API キーが正しいか確認してください．"
        elif "not found" in msg.lower() or "404" in msg:
            hint = f"モデル名「{model}」が使えない可能性があります．Google AI Studio で利用できるモデル名を確認し，サイドバーで変更してください．"
        elif "429" in msg or "quota" in msg.lower():
            hint = "無料枠の利用回数の上限に達した可能性があります．少し時間をおいて試してください．"
        else:
            hint = "ネットワーク接続を確認してください．"
        raise GeminiError(f"Gemini API の呼び出しに失敗しました．{hint}\n（詳細：{msg[:300]}）") from e


def ocr_lines(client: genai.Client, model: str, image_png: bytes) -> list[dict]:
    """画像から手書き文字を行ごとに書き起こす．戻り値：[{text, box_2d}, ...]"""
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=list[OcrLine],
    )
    resp = _generate(client, model,
                     [types.Part.from_bytes(data=image_png, mime_type="image/png"), OCR_PROMPT],
                     config)
    try:
        data = json.loads(resp.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise GeminiError(f"Gemini の応答を読み取れませんでした：{(resp.text or '')[:200]}") from e
    lines = []
    for item in data:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        box = item.get("box_2d") or []
        lines.append({"text": text, "box_2d": box if len(box) == 4 else None})
    return lines


def transcribe_audio(client: genai.Client, model: str, audio: bytes, mime_type: str = "audio/wav") -> str:
    """録音した音声を文字に書き起こす（音声での修正指示に使用）．"""
    config = types.GenerateContentConfig(temperature=0)
    resp = _generate(client, model,
                     [types.Part.from_bytes(data=audio, mime_type=mime_type), TRANSCRIBE_PROMPT],
                     config)
    return (resp.text or "").strip()


def transcribe_schedule(client: genai.Client, model: str, audio: bytes, mime_type: str = "audio/wav") -> list[str]:
    """予定を話した音声を，予定ごとの文字列のリストに書き起こす（音声での予定入力に使用）．"""
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=list[str],
    )
    resp = _generate(client, model,
                     [types.Part.from_bytes(data=audio, mime_type=mime_type), DICTATION_PROMPT],
                     config)
    try:
        data = json.loads(resp.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise GeminiError(f"Gemini の応答を読み取れませんでした：{(resp.text or '')[:200]}") from e
    return [str(x).strip() for x in data if str(x).strip()]


def extract_events_baseline(client: genai.Client, model: str, image_png: bytes, today: date) -> list[dict]:
    """【比較用】Gemini に予定の抽出までまとめて任せる．"""
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=list[BaselineEvent],
    )
    resp = _generate(client, model,
                     [types.Part.from_bytes(data=image_png, mime_type="image/png"),
                      BASELINE_PROMPT.format(today=today.isoformat())],
                     config)
    try:
        return json.loads(resp.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise GeminiError(f"Gemini の応答を読み取れませんでした：{(resp.text or '')[:200]}") from e
