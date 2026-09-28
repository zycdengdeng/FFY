from __future__ import annotations

import hashlib
import html
import http.client
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import http.cookiejar
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)


ROOT = Path(r"C:\Users\ZihanWANG\Desktop\FFY\FFY")
SOURCE_DIR = Path(r"C:\Users\ZihanWANG\Desktop\FFY\Stilt论文调研")
TMP_DIR = ROOT / "tmp" / "pdfs" / "translation"
OUT_DIR = ROOT / "output" / "pdf"

PAPERS = [
    {
        "source": "A terradynamics of legged locomotion on granular media.pdf",
        "output": "颗粒介质上腿式运动的地面动力学_中文版.pdf",
        "title": "颗粒介质上腿式运动的地面动力学",
        "original_title": "A Terradynamics of Legged Locomotion on Granular Media",
        "authors": "Chen Li, Tingnan Zhang, Daniel I. Goldman",
        "citation": "Science 339, 1408-1412 (2013)",
        "short": "腿式运动的地面动力学",
        "first_page": 1,
        "last_page": 5,
    },
    {
        "source": "A_scalable_pipeline_for_designing_reconfigurable_o.pdf",
        "output": "设计可重构生物体的可扩展流程_中文版.pdf",
        "title": "设计可重构生物体的可扩展流程",
        "original_title": "A Scalable Pipeline for Designing Reconfigurable Organisms",
        "authors": "Sam Kriegman, Douglas Blackiston, Michael Levin, Josh Bongard",
        "citation": "PNAS 117(4), 1853-1859 (2020)",
        "short": "可重构生物体设计流程",
        "first_page": 1,
        "last_page": 7,
    },
    {
        "source": "Why the seahorse tail is square.pdf",
        "output": "海马的尾巴为什么是方形的_中文版.pdf",
        "title": "海马的尾巴为什么是方形的",
        "original_title": "Why the Seahorse Tail Is Square",
        "authors": "Michael M. Porter, Dominique Adriaens, Ross L. Hatton, Marc A. Meyers, Joanna McKittrick",
        "citation": "Science 349(6243), aaa6683 (2015)",
        "short": "海马尾巴为何是方形",
        "first_page": 2,
        "last_page": 9,
    },
]

NOISE_PATTERNS = [
    re.compile(r"^Downloaded (?:by|from) ", re.I),
    re.compile(r"^arXiv:\S+", re.I),
    re.compile(r"^PNAS Latest Articles", re.I),
    re.compile(r"^www\.pnas\.org/cgi/doi", re.I),
    re.compile(r"^SCIENCE\s+sciencemag\.org", re.I),
    re.compile(r"^RESEARCH\s*$", re.I),
    re.compile(r"^RESEARCH ARTICLE SUMMARY\s*$", re.I),
]

SECTION_WORDS = {
    "Abstract", "Introduction", "Results", "Discussion", "Conclusion", "Conclusions",
    "Methods", "Materials and Methods", "References", "References and Notes",
    "Acknowledgments", "Acknowledgements", "Data Availability", "Significance",
    "Supplementary Materials", "Transferability Filter", "Evolutionary Algorithm",
}


def extract_pages(pdf: Path) -> list[str]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    txt = TMP_DIR / f"{pdf.stem}.txt"
    subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), str(txt)], check=True)
    return txt.read_text(encoding="utf-8", errors="replace").split("\f")


def clean_line(line: str) -> str:
    line = line.replace("\u00ad", "").replace("", "").replace("◥", "")
    line = re.sub(r"\s+", " ", line).strip()
    compact_letters = re.sub(r"[^A-Za-z]", "", line).upper()
    if re.fullmatch(r"(?:RESEARCH)+(?:ARTICLESUMMARY|ARTICLE)?", compact_letters):
        return ""
    return line


def looks_like_heading(line: str) -> bool:
    s = line.strip().rstrip(":")
    if s in SECTION_WORDS:
        return True
    if re.match(r"^(Fig(?:ure)?|Table|Movie)\s+[S]?\d+[A-Za-z]?\.?$", s, re.I):
        return True
    if 2 <= len(s) <= 55 and s.isupper() and sum(c.isalpha() for c in s) >= 3:
        return True
    if re.match(r"^(Results|Methods|Discussion|Conclusion|Data Availability|ACKNOWLEDGMENTS?)\b", s, re.I) and len(s) < 90:
        return True
    return False


def page_to_paragraphs(raw: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = " ".join(current)
        text = re.sub(r"(?<=[a-z])-[ ]+(?=[a-z])", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            paragraphs.append(text)
        current = []

    for source_line in raw.splitlines():
        line = clean_line(source_line)
        if not line:
            flush()
            continue
        if any(p.search(line) for p in NOISE_PATTERNS):
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        if looks_like_heading(line):
            flush()
            paragraphs.append(line)
            continue
        if current and current[-1].endswith("-") and line[:1].islower():
            current[-1] = current[-1][:-1] + line
        else:
            current.append(line)
    flush()
    return paragraphs


def split_for_translation(text: str, limit: int = 2800) -> list[str]:
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        if len(sentence) > limit:
            pieces = [sentence[i:i + limit] for i in range(0, len(sentence), limit)]
        else:
            pieces = [sentence]
        for piece in pieces:
            candidate = (buf + " " + piece).strip()
            if buf and len(candidate) > limit:
                chunks.append(buf)
                buf = piece
            else:
                buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


class BingTranslator:
    def __init__(self):
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookie_jar))
        self.ig = ""
        self.iid = ""
        self.token = ""
        self.key = ""

    def refresh(self) -> None:
        request = urllib.request.Request(
            "https://www.bing.com/translator",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with self.opener.open(request, timeout=45) as response:
            page = response.read().decode("utf-8", errors="ignore")
        self.ig = re.search(r'IG:"([^"]+)', page).group(1)
        self.iid = re.search(r'data-iid="([^"]+)', page).group(1)
        token_match = re.search(r'params_AbusePreventionHelper = \[(\d+),"([^"]+)', page)
        self.key, self.token = token_match.group(1), token_match.group(2)

    def translate(self, text: str) -> str:
        if not self.token:
            self.refresh()
        data = urllib.parse.urlencode({
            "fromLang": "en",
            "text": text,
            "to": "zh-Hans",
            "token": self.token,
            "key": self.key,
            "tryFetchingGenderDebiasedTranslations": "true",
        }).encode("utf-8")
        url = f"https://www.bing.com/ttranslatev3?isVertical=1&IG={self.ig}&IID={self.iid}"
        request = urllib.request.Request(
            url, data=data, method="POST",
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.bing.com/translator"},
        )
        with self.opener.open(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload[0]["translations"][0]["text"]


BING_TRANSLATOR = BingTranslator()


def argos_translate(text: str) -> str:
    from argostranslate import translate as argos
    return argos.translate(text, "en", "zh")


def translate_chunk(text: str) -> str:
    return argos_translate(text)


def translate(text: str, cache: dict[str, str], cache_path: Path) -> str:
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]
    translated = "".join(translate_chunk(part) for part in split_for_translation(text))
    cache[key] = translated
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return translated


def translate_paragraphs(paragraphs: list[str], cache: dict[str, str], cache_path: Path) -> list[str]:
    results: list[str | None] = [None] * len(paragraphs)
    missing: list[tuple[int, str, str]] = []
    for index, paragraph in enumerate(paragraphs):
        key = hashlib.sha256(paragraph.encode("utf-8")).hexdigest()
        if key in cache:
            results[index] = cache[key]
        else:
            missing.append((index, paragraph, key))

    batches: list[list[tuple[int, str, str]]] = []
    current: list[tuple[int, str, str]] = []
    current_size = 0
    for item in missing:
        index, paragraph, key = item
        addition = len(paragraph) + 24
        if current and (current_size + addition > 2500 or len(paragraph) > 2200):
            batches.append(current)
            current = []
            current_size = 0
        if len(paragraph) > 2200:
            batches.append([item])
        else:
            current.append(item)
            current_size += addition
    if current:
        batches.append(current)

    for batch in batches:
        if len(batch) == 1:
            index, paragraph, key = batch[0]
            translated = "".join(translate_chunk(part) for part in split_for_translation(paragraph))
            results[index] = translated
            cache[key] = translated
        else:
            marker = "\ue000"
            payload = f"\n\n{marker}\n\n".join(item[1] for item in batch)
            translated_batch = translate_chunk(payload)
            parts = [part.strip() for part in translated_batch.split(marker)]
            if len(parts) == len(batch):
                for item, translated in zip(batch, parts):
                    index, _, key = item
                    results[index] = translated
                    cache[key] = translated
            else:
                for index, paragraph, key in batch:
                    translated = "".join(translate_chunk(part) for part in split_for_translation(paragraph))
                    results[index] = translated
                    cache[key] = translated
                    time.sleep(0.8)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.8)

    return [item or "" for item in results]


def normalize_translation(text: str) -> str:
    replacements = {
        "陆地动力学": "地面动力学",
        "可重新配置的生物": "可重构生物体",
        "可重新配置的生物体": "可重构生物体",
        "海马尾部": "海马尾巴",
        "颗粒媒体": "颗粒介质",
        "图。": "图",
        "无花果。": "图",
        "补充图": "补充图",
        "体内": "体内",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def safe_markup(text: str) -> str:
    text = html.escape(text, quote=False)
    text = text.replace("\n", "<br/>")
    return text


def make_styles():
    pdfmetrics.registerFont(TTFont("Deng", r"C:\Windows\Fonts\Deng.ttf"))
    pdfmetrics.registerFont(TTFont("Deng-Bold", r"C:\Windows\Fonts\Dengb.ttf"))
    styles = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=styles["Title"], fontName="Deng-Bold", fontSize=23,
            leading=34, alignment=TA_CENTER, textColor=colors.HexColor("#17324D"),
            spaceAfter=16,
        ),
        "original_title": ParagraphStyle(
            "OriginalTitle", parent=styles["Normal"], fontName="Deng", fontSize=10.5,
            leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#536777"),
            spaceAfter=18,
        ),
        "authors": ParagraphStyle(
            "Authors", parent=styles["Normal"], fontName="Deng", fontSize=11,
            leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#263746"),
            spaceAfter=10,
        ),
        "note": ParagraphStyle(
            "Note", parent=styles["Normal"], fontName="Deng", fontSize=9.3,
            leading=15, alignment=TA_LEFT, textColor=colors.HexColor("#475866"),
            backColor=colors.HexColor("#F2F6F8"), borderColor=colors.HexColor("#D6E1E7"),
            borderWidth=0.6, borderPadding=10, spaceBefore=10, spaceAfter=12,
        ),
        "page_heading": ParagraphStyle(
            "PageHeading", parent=styles["Heading2"], fontName="Deng-Bold", fontSize=13,
            leading=20, textColor=colors.HexColor("#177E89"), spaceBefore=6, spaceAfter=8,
            keepWithNext=True,
        ),
        "heading": ParagraphStyle(
            "Heading", parent=styles["Heading3"], fontName="Deng-Bold", fontSize=12,
            leading=19, textColor=colors.HexColor("#17324D"), spaceBefore=9, spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body", parent=styles["BodyText"], fontName="Deng", fontSize=10.2,
            leading=16.5, alignment=TA_JUSTIFY, textColor=colors.HexColor("#20282E"),
            firstLineIndent=2.0 * 10.2, spaceAfter=7, allowWidows=0, allowOrphans=0,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=styles["BodyText"], fontName="Deng", fontSize=9.2,
            leading=14.5, alignment=TA_LEFT, textColor=colors.HexColor("#43515C"),
            backColor=colors.HexColor("#F7F9FA"), borderPadding=6, spaceBefore=4, spaceAfter=8,
        ),
    }


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, short_title: str):
        super().__init__(
            filename, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
            topMargin=20 * mm, bottomMargin=19 * mm,
            title=short_title, author="中文翻译阅读版",
        )
        self.short_title = short_title
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(colors.HexColor("#D9E2E7"))
            canvas.setLineWidth(0.5)
            canvas.line(22 * mm, A4[1] - 14 * mm, A4[0] - 22 * mm, A4[1] - 14 * mm)
            canvas.setFont("Deng", 8)
            canvas.setFillColor(colors.HexColor("#667783"))
            canvas.drawString(22 * mm, A4[1] - 11 * mm, self.short_title)
        canvas.setFont("Deng", 8)
        canvas.setFillColor(colors.HexColor("#667783"))
        canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
        canvas.restoreState()


def build_paper(meta: dict, styles: dict) -> Path:
    source = SOURCE_DIR / meta["source"]
    output = OUT_DIR / meta["output"]
    cache_path = TMP_DIR / f"{source.stem}.translation-cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    pages = extract_pages(source)
    numbered_pages = [
        (number, raw)
        for number, raw in enumerate(pages, start=1)
        if meta["first_page"] <= number <= meta["last_page"]
    ]
    story = [
        Spacer(1, 30 * mm),
        Paragraph(safe_markup(meta["title"]), styles["cover_title"]),
        Paragraph(safe_markup(meta["original_title"]), styles["original_title"]),
        Paragraph(safe_markup(meta["authors"]), styles["authors"]),
        Paragraph(safe_markup(meta["citation"]), styles["authors"]),
        Spacer(1, 10 * mm),
        Paragraph(
            "<b>版本说明</b><br/>本文件为便于阅读和写作调研制作的中文翻译版。正文、图表说明与方法按原 PDF 页序翻译；公式、变量、单位、引文序号、网址和专有缩写尽量保留。由于原文采用双栏排版，个别图注或侧栏在本版中可能与正文相邻出现。正式引用请以英文原文为准。",
            styles["note"],
        ),
        PageBreak(),
    ]

    total_pages = len([p for p in pages if p.strip()])
    for page_number, raw_page in numbered_pages:
        if not raw_page.strip():
            continue
        paragraphs = page_to_paragraphs(raw_page)
        if not paragraphs:
            continue
        story.append(Paragraph(f"原文第 {page_number} 页 / 共 {total_pages} 页", styles["page_heading"]))
        translated_paragraphs = translate_paragraphs(paragraphs, cache, cache_path)
        for para, raw_translation in zip(paragraphs, translated_paragraphs):
            translated = normalize_translation(raw_translation)
            if not translated:
                continue
            if looks_like_heading(para):
                story.append(Paragraph(safe_markup(translated), styles["heading"]))
            elif re.match(r"^(Fig(?:ure)?|Table|Movie)\s+", para, re.I):
                story.append(Paragraph(safe_markup(translated), styles["caption"]))
            else:
                story.append(Paragraph(safe_markup(translated), styles["body"]))
        if page_number < meta["last_page"]:
            story.append(Spacer(1, 4 * mm))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PaperDocTemplate(str(output), meta["short"]).build(story)
    return output


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    for paper in PAPERS:
        output = build_paper(paper, styles)
        print(output)


if __name__ == "__main__":
    main()
