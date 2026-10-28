"""画像の前処理（OpenCV）

撮影した手書きメモの画像に対して，
  1. 紙の四隅を検出して台形補正（射影変換）
  2. 明るさのむら（影）の除去
  3. 二値化
を順に行う．各段階の画像を返すので，画面表示や評価に使える．
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    final: np.ndarray                      # Gemini に送る画像（BGR）
    steps: dict[str, np.ndarray] = field(default_factory=dict)  # 表示用：段階ごとの画像
    paper_found: bool = False              # 紙の四隅を検出できたか


def decode_image(data: bytes) -> np.ndarray:
    """画像ファイルのバイト列を BGR 画像に変換する．"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("画像を読み込めませんでした．JPEG か PNG の画像を指定してください．")
    return img


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("画像の変換に失敗しました．")
    return buf.tobytes()


def limit_size(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """大きすぎる画像を縮小する（処理時間と API 送信量を抑えるため）．"""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 台形補正
def _order_points(pts: np.ndarray) -> np.ndarray:
    """4点を 左上・右上・右下・左下 の順に並べる．"""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def find_paper(img: np.ndarray, min_area_ratio: float = 0.2) -> np.ndarray | None:
    """画像中で一番大きい四角形（紙）の四隅を返す．見つからなければ None．"""
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img.copy()
    scale = min(scale, 1.0)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = small.shape[0] * small.shape[1] * min_area_ratio
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < area_min:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_points(approx / scale)
    return None


def warp_paper(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """四隅が長方形になるように射影変換する．"""
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, m, (width, height))


# ---------------------------------------------------------------- 影の除去・二値化
def remove_shadow(img: np.ndarray) -> np.ndarray:
    """背景の明るさを推定して割り算し，影や照明のむらを取り除く（グレースケールで返す）．"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # 文字（細い線）を膨張処理で消し，さらに平滑化して「背景だけの画像」を作る
    background = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    background = cv2.medianBlur(background, 21)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


def binarize(gray: np.ndarray) -> np.ndarray:
    """適応的しきい値処理で白黒にする．"""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)


# ---------------------------------------------------------------- まとめ
def preprocess(img: np.ndarray, *, do_warp: bool = True, do_shadow: bool = True,
               do_binarize: bool = True) -> PreprocessResult:
    img = limit_size(img)
    steps: dict[str, np.ndarray] = {"元画像": img}
    current = img
    found = False

    if do_warp:
        corners = find_paper(img)
        if corners is not None:
            current = warp_paper(img, corners)
            found = True
            steps["台形補正"] = current

    if do_shadow:
        current = remove_shadow(current)
        steps["影の除去"] = current

    if do_binarize:
        gray = current if current.ndim == 2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        current = binarize(gray)
        steps["二値化"] = current

    if current.ndim == 2:
        current = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
    return PreprocessResult(final=current, steps=steps, paper_found=found)
