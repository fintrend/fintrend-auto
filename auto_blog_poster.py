# -*- coding: utf-8 -*-
"""
FinTrend 自動投稿 完成版
- WordPress へ記事+アイキャッチ自動投稿
- 市場データ: yfinance（株価・VIX・DXY・WTI・金・米10年）
- Finnhub: 会社ニュース（403/権限不足は握りつぶして続行）
- OpenAI: 長文記事生成 / 画像生成（失敗時はMatplotlibでダッシュボード画像）
"""

from __future__ import annotations
import os, io, re, json, base64, textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import yfinance as yf

# OpenAI (v1 SDK)
try:
    from openai import OpenAI
    _openai_ok = True
except Exception:
    _openai_ok = False

# Matplotlib fallback image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========== 設定 ==========
load_dotenv()
JST = timezone(timedelta(hours=9), 'Asia/Tokyo')

WP_BASE_URL    = os.getenv("WP_BASE_URL", "").rstrip("/")
WP_POSTS_URL   = f"{WP_BASE_URL}{os.getenv('WP_API_ENDPOINT','/wp-json/wp/v2/posts')}"
WP_MEDIA_URL   = f"{WP_BASE_URL}/wp-json/wp/v2/media"
WP_TAGS_URL    = f"{WP_BASE_URL}/wp-json/wp/v2/tags"
WP_USERNAME    = os.getenv("WP_USERNAME", "")
WP_APP_PASSWORD= os.getenv("WP_APP_PASSWORD", "")
WP_CATEGORY_ID = os.getenv("WP_CATEGORY_ID", "1")
POST_STATUS    = os.getenv("POST_STATUS", "publish")
WP_TAGS_RAW    = os.getenv("WP_TAGS", "米国株,仮想通貨,VIX,WTI").strip()

FINNHUB_API_KEY= os.getenv("FINNHUB_API_KEY","")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
SLUG_PREFIX    = os.getenv("SLUG_PREFIX", "market-report")

LOG_FILE       = "logs.txt"

# OpenAI client
client = None
if _openai_ok and OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None

# ========== ユーティリティ ==========
def log(msg: str) -> None:
    ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass

def wp_auth():
    return HTTPBasicAuth(WP_USERNAME, WP_APP_PASSWORD)

def ensure_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"

def http_json(url: str, params: Optional[dict]=None) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        # Finnhub 403など
        log(f"http_json 失敗: {url} [{r.status_code}] {r.text[:150]}")
    except Exception as e:
        log(f"http_json例外: {url} | {e}")
    return None

# ========== データ取得 ==========
def price_of(ticker: str) -> Optional[float]:
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if len(data) == 0: return None
        return float(round(data["Close"].iloc[-1], 2))
    except Exception as e:
        log(f"yfinance 価格取得失敗: {ticker} | {e}")
        return None

def market_dashboard() -> Dict[str, Optional[float]]:
    # 代表指標
    tickers = {
        "VIX": "^VIX",
        "ドルインデックス": "DX-Y.NYB",
        "WTI原油": "CL=F",
        "金先物": "GC=F",
        "米10年債利回り(%)": "^TNX"
    }
    out = {}
    for name,tkr in tickers.items():
        out[name] = price_of(tkr)
    return out

def m7_prices() -> Dict[str, Optional[float]]:
    names = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA"]
    return { t: price_of(t) for t in names }

def finnhub_company_news(ticker: str) -> List[dict]:
    if not FINNHUB_API_KEY:
        return []
    frm = (datetime.utcnow() - timedelta(days=4)).date().isoformat()
    to  = datetime.utcnow().date().isoformat()
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={frm}&to={to}&token={FINNHUB_API_KEY}"
    js = http_json(url)
    if not js or isinstance(js, dict) and js.get("error"):
        return []
    # 上位2件
    return js[:2] if isinstance(js, list) else []

# ========== 記事生成 ==========
def build_context_text() -> str:
    dash = market_dashboard()
    m7   = m7_prices()

    parts = []
    parts.append("【主要指標】")
    for k,v in dash.items():
        parts.append(f"- {k}: {v}")
    parts.append("\n【マグニフィセントセブン】")
    for k,v in m7.items():
        parts.append(f"- {k}: {v}")

    parts.append("\n【社名ベースの話題銘柄ニュース】")
    for t in ["AAPL","NVDA","TSLA","MSFT","META","GOOGL","AMZN"]:
        ns = finnhub_company_news(t)
        if ns:
            parts.append(f"- {t}:")
            for n in ns:
                ttl = n.get("headline","").strip()
                url = n.get("url","")
                parts.append(f"  - {ttl} ({url})")
    return "\n".join(parts)

def long_article_by_openai(context_text: str) -> Optional[str]:
    if not client:
        return None
    try:
        sys_prompt = (
            "あなたは日本語の投資ライターです。"
            "以下の市場データ・ニュースを使って、投資家向けに読みやすい長文の相場レポートを作成してください。"
            "見出しは「1. 2. 3. …」の番号見出しを使い、## 等のMarkdown見出し記号は付けないでください。"
            "投資アドバイスは一般的・教育的な観点にとどめ、特定銘柄の推奨は避けます。"
            "安全資産（ゴールド・債券・投資信託）にも必ず言及し、VIX・ドルインデックス・WTI・金利にも触れてください。"
            "最後に、マグニフィセントセブンのハイライトと社名ベースの話題銘柄を箇条書きで要約してください。"
            "本文は2000〜2600字程度。"
        )
        user_prompt = f"本日の市場データと材料:\n{context_text}\n\nこれを使って記事を書いてください。"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1800,
            messages=[
                {"role":"system","content": sys_prompt},
                {"role":"user","content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log(f"OpenAI本文生成 失敗: {e}")
        return None

def long_article_fallback(context_text: str) -> str:
    # OpenAI が使えない時のテンプレ長文
    today = datetime.now(JST).strftime("%Y年%m月%d日")
    return textwrap.dedent(f"""
    1. 本日の市場サマリー（{today}）
    本レポートでは、米国株式・為替・コモディティ・債券を横断して俯瞰します。主要指標の動き・ニュースの背景を整理し、投資家が押さえるべき観点を解説します。

    2. 主要指標の現状
    {context_text}

    3. 安全資産の位置づけ
    市場の不確実性が高まる局面では、金や良質な債券、インデックス型の投資信託が相対的に堅調となる傾向があります。VIXが上昇する場合はボラティリティ・リスクが高まるため、分散とキャッシュ余力を重視した運用が有効です。

    4. 金利・ドルインデックスの示唆
    米10年債利回りの変動は株式の割引率を通じてバリュエーションに影響します。ドルインデックスはコモディティ価格の逆風/追い風となり、WTI原油や金のトレンドに波及する点にも留意が必要です。

    5. マグニフィセントセブンの動向
    高い収益性とAI/クラウド等の成長領域を背景に、依然として指数寄与度が高いセクターです。ただしイベント・決算期には変動率が上がるため、リスク管理を徹底したいところです。

    6. 投資家にとっての実務的ポイント
    - ボラティリティ上昇下では、ポジションサイズ調整・ディフェンシブ資産の組み込み・ヘッジ手段の検討が有用
    - 長期投資の軸は分散・低コスト・規律を基本に
    - テーマ投資は中核資産と衛星資産を明確に分離し、想定外の変動に備える

    7. まとめ
    本日の材料を踏まえ、過度な強気/弱気に振れず、データに基づく判断を積み重ねることが重要です。
    """).strip()

def build_article() -> str:
    ctx = build_context_text()
    text = long_article_by_openai(ctx)
    return text if text else long_article_fallback(ctx)

# ========== 画像生成 ==========
def generate_feature_image() -> Optional[str]:
    """OpenAI 画像 → 失敗したらMatplotlib ダッシュボード画像"""
    # まず OpenAI 画像
    if client:
        try:
            prompt = (
                "米国市場のダッシュボード。VIX・ドルインデックス・WTI・金・金利と、"
                "マグニフィセントセブンのテック感を想起させる安全・知的なイメージ。"
                "ミニマルで品のある青系。テキスト要素やロゴは入れない。"
            )
            img = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024"
            )
            # 画像データ（base64_b64json）形式にも対応
            if hasattr(img, "data") and img.data and hasattr(img.data[0], "b64_json"):
                png_bytes = base64.b64decode(img.data[0].b64_json)
                out = os.path.join(os.getcwd(), f"feature_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.png")
                with open(out, "wb") as f:
                    f.write(png_bytes)
                return out
        except Exception as e:
            log(f"OpenAI画像 失敗: {e}")

    # フォールバック: Matplotlib で簡易ダッシュボード
    dash = market_dashboard()
    m7   = m7_prices()

    fig, ax = plt.subplots(figsize=(8,6), dpi=160)
    ax.axis('off')
    ax.set_title("Market Dashboard", loc='left', fontsize=16, fontweight='bold')
    y = 0.95
    ax.text(0.02, y, datetime.now(JST).strftime("%Y-%m-%d %H:%M JST"),
            transform=ax.transAxes, fontsize=10); y -= 0.08
    ax.text(0.02, y, "■ 主要指標", transform=ax.transAxes, fontsize=12, fontweight='bold'); y -= 0.06
    for k,v in dash.items():
        ax.text(0.04, y, f"{k: <12}: {v}", transform=ax.transAxes, fontsize=11); y -= 0.05
    y -= 0.02
    ax.text(0.02, y, "■ Magnificent 7", transform=ax.transAxes, fontsize=12, fontweight='bold'); y -= 0.06
    for k,v in m7.items():
        ax.text(0.04, y, f"{k: <5}: {v}", transform=ax.transAxes, fontsize=11); y -= 0.05

    out = os.path.join(os.getcwd(), f"feature_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.png")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

# ========== WordPress ==========
def wp_tag_id_by_name(name: str) -> Optional[int]:
    try:
        r = requests.get(WP_TAGS_URL, params={"search": name}, auth=wp_auth(), timeout=20)
        if r.status_code == 200 and isinstance(r.json(), list) and r.json():
            return int(r.json()[0]["id"])
        # 作成試行
        r2 = requests.post(WP_TAGS_URL, auth=wp_auth(), json={"name": name}, timeout=20)
        if r2.status_code in (200,201):
            return int(r2.json()["id"])
        log(f"タグ作成失敗: {name} [{r2.status_code}] {r2.text[:120]}")
    except Exception as e:
        log(f"タグID取得/作成 例外: {e}")
    return None

def upload_media(filepath: str) -> Optional[int]:
    try:
        fname = os.path.basename(filepath)
        headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
        with open(filepath, "rb") as f:
            files = {"file": (fname, f, "image/png")}
            r = requests.post(WP_MEDIA_URL, auth=wp_auth(), headers=headers, files=files, timeout=60)
        if r.status_code in (200,201):
            mid = int(r.json().get("id"))
            log(f"WP 画像アップ成功: id={mid}")
            return mid
        log(f"WP 画像アップ失敗: [{r.status_code}] {r.text[:200]}")
    except Exception as e:
        log(f"WP 画像アップ例外: {e}")
    return None

def post_exists(slug: str) -> bool:
    try:
        r = requests.get(WP_POSTS_URL, params={"slug": slug}, timeout=20)
        if r.status_code == 200 and isinstance(r.json(), list):
            return len(r.json()) > 0
    except Exception as e:
        log(f"既存チェック例外: {e}")
    return False

def post_to_wp(title: str, slug: str, content_md: str, featured_media: Optional[int], tag_ids: List[int]) -> Optional[str]:
    data = {
        "title": title,
        "slug":  slug,
        "content": content_md,
        "status": POST_STATUS,
        "categories": [int(WP_CATEGORY_ID)] if str(WP_CATEGORY_ID).isdigit() else [],
        "tags": tag_ids,
    }
    if featured_media: data["featured_media"] = featured_media

    try:
        r = requests.post(WP_POSTS_URL, auth=wp_auth(), json=data, timeout=60)
        if r.status_code in (200,201):
            j = r.json()
            url = j.get("link") or j.get("guid",{}).get("rendered","")
            log(f"投稿成功: {url}")
            return url
        log(f"投稿失敗: [{r.status_code}] {r.text[:300]}")
    except Exception as e:
        log(f"投稿例外: {e}")
    return None

# ========== メイン ==========
def main():
    log("=== 自動投稿タスク開始 ===")

    # 環境確認
    if not all([WP_BASE_URL, WP_USERNAME, WP_APP_PASSWORD]):
        log("WP 設定不足。WP_BASE_URL, WP_USERNAME, WP_APP_PASSWORD を確認してください。")
        return

    # タイトル・スラッグ
    now = datetime.now(JST)
    title = f"{now.strftime('%Y年%m月%d日')} 米国株・仮想通貨の市場レポート"
    slug  = f"{SLUG_PREFIX}-{now.strftime('%Y%m%d-%H%M')}"
    if post_exists(slug):
        log("同スラッグ記事がすでに存在します。終了。")
        return

    # 本文
    log("本文生成 フェーズ開始")
    content_md = build_article()

    # タグ
    tag_ids: List[int] = []
    if WP_TAGS_RAW:
        for nm in [s.strip() for s in WP_TAGS_RAW.split(",") if s.strip()]:
            tid = wp_tag_id_by_name(nm)
            if tid: tag_ids.append(tid)

    # アイキャッチ生成 → アップロード
    fm_id = None
    img_path = generate_feature_image()
    if img_path:
        fm_id = upload_media(img_path)

    # 投稿
    url = post_to_wp(title, slug, content_md, fm_id, tag_ids)
    if url:
        log(f"完了: {url}")
    log("=== 自動投稿タスク終了 ===")

if __name__ == "__main__":
    main()
