"""評価スクリプト（説明書「5. 評価方法」用）

測るもの
  (1) 文字認識の正解率（1 − 文字誤り率 CER）
  (2) 日付・時刻・予定名がすべて正しく抽出された予定の割合（再現率・適合率）
  (3) 前処理の設定による違い
  (+) Gemini に予定の抽出までまとめて任せた場合（--baseline）との比較

フォルダ構成（eval/ の例は README を参照）
  eval/images/       評価用の画像（memo01.jpg など）
  eval/truth.csv     正解の予定  列：image,date,start,end,title
  eval/transcripts/  正解の書き起こし（memo01.txt など．1行ずつ）※文字認識の評価に使用．なくてもよい

実行例
  python evaluate.py --ref 2026-10-28
  python evaluate.py --ref 2026-10-28 --baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time as time_mod
import tomllib
import unicodedata
from datetime import date, time
from pathlib import Path

from extractor import Event, extract_events
from ocr_gemini import DEFAULT_MODEL, extract_events_baseline, make_client, ocr_lines
from preprocess import decode_image, encode_png, preprocess

CONFIGS = {
    "なし": dict(do_warp=False, do_shadow=False, do_binarize=False),
    "台形補正+影除去": dict(do_warp=True, do_shadow=True, do_binarize=False),
    "台形補正+影除去+二値化": dict(do_warp=True, do_shadow=True, do_binarize=True),
}


def load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    secrets = Path(".streamlit/secrets.toml")
    if not key and secrets.exists():
        key = tomllib.loads(secrets.read_text(encoding="utf-8")).get("GEMINI_API_KEY", "")
    return key


def norm_text(s: str) -> str:
    return "".join(unicodedata.normalize("NFKC", s).split())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def parse_hm(s: str) -> time | None:
    s = (s or "").strip()
    if not s:
        return None
    h, m = s.split(":")
    return time(int(h), int(m))


def key_of(d: date | None, start: time | None, end: time | None, title: str) -> tuple:
    return (d, start, end, norm_text(title))


def match(pred: list[tuple], truth: list[tuple]) -> int:
    rest = list(pred)
    hit = 0
    for t in truth:
        if t in rest:
            rest.remove(t)
            hit += 1
    return hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval")
    ap.add_argument("--ref", required=True, help="基準日（撮影日） YYYY-MM-DD")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--baseline", action="store_true", help="Gemini に抽出まで任せる方式とも比較する")
    ap.add_argument("--sleep", type=float, default=4.0, help="API 呼び出しの間隔（秒）．無料枠の回数制限対策")
    args = ap.parse_args()

    root = Path(args.dir)
    ref = date.fromisoformat(args.ref)
    cache_path = root / "ocr_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    truth: dict[str, list[tuple]] = {}
    with open(root / "truth.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            truth.setdefault(r["image"], []).append(
                key_of(date.fromisoformat(r["date"]) if r["date"] else None,
                       parse_hm(r["start"]), parse_hm(r["end"]), r["title"]))

    if not (root / "images").exists():
        sys.exit(f"{root / 'images'} フォルダがありません．")
    images = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        sys.exit("eval/images に画像がありません．")
    client = make_client(load_api_key())

    def call(fn, *a):
        r = fn(*a)
        time_mod.sleep(args.sleep)
        return r

    rows, summary = [], []
    for cname, cfg in CONFIGS.items():
        tot_chars = tot_err = n_truth = n_pred = n_hit = 0
        for img_path in images:
            name = img_path.name
            png = encode_png(preprocess(decode_image(img_path.read_bytes()), **cfg).final)
            ck = f"{cname}|{args.model}|{name}"
            if ck not in cache:
                print(f"[{cname}] {name} を認識中…")
                cache[ck] = call(ocr_lines, client, args.model, png)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            lines = [ln["text"] for ln in cache[ck]]

            tr_path = root / "transcripts" / (img_path.stem + ".txt")
            cer = None
            if tr_path.exists():
                gt = norm_text(tr_path.read_text(encoding="utf-8"))
                err = edit_distance(norm_text("".join(lines)), gt)
                tot_chars += len(gt)
                tot_err += err
                cer = err / max(len(gt), 1)

            events: list[Event] = extract_events(lines, ref)
            pred = [key_of(e.date, e.start, e.end, e.title) for e in events]
            t = truth.get(name, [])
            hit = match(pred, t)
            n_truth += len(t); n_pred += len(pred); n_hit += hit
            rows.append([cname, name, " / ".join(lines), f"{cer:.3f}" if cer is not None else "",
                         len(t), len(pred), hit])

        summary.append([cname,
                        f"{1 - tot_err / tot_chars:.1%}" if tot_chars else "－",
                        f"{n_hit / n_truth:.1%}" if n_truth else "－",
                        f"{n_hit / n_pred:.1%}" if n_pred else "－"])

    if args.baseline:
        n_truth = n_pred = n_hit = 0
        for img_path in images:
            name = img_path.name
            ck = f"baseline|{args.model}|{name}"
            if ck not in cache:
                print(f"[Gemini一括] {name} を処理中…")
                png = encode_png(decode_image(img_path.read_bytes()))
                cache[ck] = call(extract_events_baseline, client, args.model, png, ref)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            pred = []
            for e in cache[ck]:
                try:
                    d = date.fromisoformat(e["date"]) if e.get("date") else None
                    pred.append(key_of(d, parse_hm(e.get("start")), parse_hm(e.get("end")), e.get("title", "")))
                except ValueError:
                    continue
            t = truth.get(name, [])
            hit = match(pred, t)
            n_truth += len(t); n_pred += len(pred); n_hit += hit
            rows.append(["Gemini一括", name, "", "", len(t), len(pred), hit])
        summary.append(["Gemini一括（比較）", "－",
                        f"{n_hit / n_truth:.1%}" if n_truth else "－",
                        f"{n_hit / n_pred:.1%}" if n_pred else "－"])

    out = root / "results.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["設定", "画像", "認識した文字", "文字誤り率", "正解の予定数", "抽出した予定数", "完全一致"])
        w.writerows(rows)

    print("\n=== 結果 ===")
    print(f"{'設定':<22}{'文字認識の正解率':<12}{'予定の再現率':<10}{'予定の適合率'}")
    for s in summary:
        print(f"{s[0]:<22}{s[1]:<16}{s[2]:<14}{s[3]}")
    print(f"\n詳細：{out}")


if __name__ == "__main__":
    main()
