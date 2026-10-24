"""Oral-defence style deck for the professor (2026-10-24): 緒論 -> 相關研究 -> 研究方法 -> 實驗結果 -> 結論, with an
outline, section dividers, one-sentence claim titles backed by a figure or diagram, references, backup slides and speaker
notes on every slide. Figures come from make_figures.py (deck/figs); every number is from SUMMARY.md.

    <python with python-pptx> make_defense_pptx.py [out.pptx]
"""
import os, re, sys
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "VCC_defense_style_20261024.pptx")
LATIN, EA, MONO = "Arial", "Microsoft JhengHei", "Courier New"
NAVY, DARK, TEAL, TEAL_TX, TEAL_LT, TEAL_BG = "1F3A5F", "1A2E4C", "2A9D8F", "1E7D72", "7FD1C4", "E4F3F1"
ORANGE, ORANGE_TX, ORANGE_BG = "E76F51", "C4512F", "FCEBE5"
TEXT, MUTED, GRAY, LIGHT, PANEL, LINE = "2B2D42", "5C6475", "8D99AE", "B4BCC8", "F4F6F8", "DDE2E8"
WHITE, ON_DARK, ON_DARK3 = "FFFFFF", "B8C7D9", "8DA2BC"
CAT = {"血球類": "2A9D8F", "上皮類": "E9C46A", "幹細胞類": "B5838D"}
L, CW = 0.6, 12.133                                   # left margin, content width (16:9, 13.333 x 7.5 in)
COL3 = [0.6, 4.717, 8.833]                            # three equal columns, 3.9 in wide
SECTIONS = [("緒論", "為什麼要預測基因擾動的結果？要預測什麼、怎麼評分？"),
            ("相關研究", "別人怎麼做？我們之前試過什麼？"),
            ("研究方法", "模型如何把知識轉移到沒看過的細胞？"),
            ("實驗結果", "在沒看過的細胞上表現如何？瓶頸在哪裡？"),
            ("結論與未來工作", "做到了什麼？接下來要做什麼？")]
BREAD = ["緒論", "相關研究", "研究方法", "實驗結果", "結論"]
ROUND, OVAL, RECT = MSO_SHAPE.ROUNDED_RECTANGLE, MSO_SHAPE.OVAL, MSO_SHAPE.RECTANGLE
TOP, MID = MSO_ANCHOR.TOP, MSO_ANCHOR.MIDDLE
LEFT, CENTER, RIGHT = PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
LAYOUT = prs.slide_layouts[5]                         # "Title Only": the claim title stays a real title placeholder


def patch_theme():
    """Theme fonts (Arial + 微軟正黑體) and colours, so text and shapes added later in PowerPoint match the deck."""
    part = prs.slide_master.part.part_related_by(RT.THEME)
    xml = part.blob.decode("utf-8")
    for tag in ("majorFont", "minorFont"):
        xml = re.sub(rf'(<a:{tag}>\s*<a:latin typeface=")[^"]*(")', rf"\g<1>{LATIN}\g<2>", xml)
        xml = re.sub(rf'(<a:{tag}>\s*<a:latin [^>]*>\s*<a:ea typeface=")[^"]*(")', rf"\g<1>{EA}\g<2>", xml)
    for k, v in {"dk2": NAVY, "lt2": PANEL, "accent1": NAVY, "accent2": TEAL, "accent3": ORANGE, "accent4": GRAY,
                 "accent5": "E9C46A", "accent6": "B5838D"}.items():
        xml = re.sub(rf'(<a:{k}>\s*<a:srgbClr val=")[0-9A-Fa-f]{{6}}(")', rf"\g<1>{v}\g<2>", xml)
    part._blob = xml.encode("utf-8")


def rgb(h):
    return RGBColor.from_string(h)


def blend(c1, c2, t):
    """c1 over c2 with opacity t (hex in, hex out)."""
    a = [int(c1[i:i + 2], 16) for i in (0, 2, 4)]; b = [int(c2[i:i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{round(x * t + y * (1 - t)):02X}" for x, y in zip(a, b))


def text_w(s, size):
    """Rough rendered width (in) of a one-line string."""
    w = 0.0
    for ch in s:
        if ord(ch) > 0x2E80: w += 1.0
        elif ch == " " or ch in "il.,:;|!'()": w += 0.3
        elif ch.isupper() or ch.isdigit(): w += 0.64
        else: w += 0.52
    return w * size / 72


def style_run(r, size, color, bold=False, italic=False, font=None, pos=None):
    f = r.font; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = rgb(color); f.name = font or LATIN
    rPr = r._r.get_or_add_rPr(); rPr.set("lang", "zh-TW"); rPr.set("altLang", "en-US")
    for tag, face in (("a:ea", EA), ("a:cs", font or LATIN)):          # CJK glyphs use 微軟正黑體
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {}); rPr.append(el)
        el.set("typeface", face)
    if pos: rPr.set("baseline", {"sub": "-25000", "sup": "30000"}[pos])


def set_bullet(p, color, indent):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent)))); pPr.set("indent", str(-int(Inches(indent))))
    clr = pPr.makeelement(qn("a:buClr"), {}); clr.append(clr.makeelement(qn("a:srgbClr"), {"val": color}))
    for el in (clr, pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}), pPr.makeelement(qn("a:buChar"), {"char": "•"})):
        pPr.append(el)


def fill(tf, paras, size=16, color=TEXT, bold=False, align=LEFT, spacing=1.15, after=0, font=None, italic=False):
    """paras: list of paragraphs; a paragraph is a str, a list of (text, opts) runs, or a dict with "runs" plus
    paragraph options (align, after, before, spacing, bullet, size, color, bold). Run opts: size, color, bold, italic,
    font, pos ("sub" / "sup")."""
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        po = para if isinstance(para, dict) else {}
        runs = po.get("runs", para) if isinstance(para, dict) else para
        if isinstance(runs, str): runs = [(runs, {})]
        p.alignment = po.get("align", align); p.line_spacing = po.get("spacing", spacing)
        p.space_after = Pt(po.get("after", after))
        if "before" in po: p.space_before = Pt(po["before"])
        if po.get("bullet"): set_bullet(p, po.get("bullet_color", GRAY), po.get("indent", 0.2))
        elif po.get("hang"):
            pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(int(Inches(po["hang"])))); pPr.set("indent", str(-int(Inches(po["hang"]))))
        for t, o in runs:
            r = p.add_run(); r.text = t
            style_run(r, o.get("size", po.get("size", size)), o.get("color", po.get("color", color)),
                      o.get("bold", po.get("bold", bold)), o.get("italic", italic), o.get("font", font), o.get("pos"))


def _frame(tf, anchor, margin, wrap=True):
    tf.word_wrap = wrap; tf.vertical_anchor = anchor; tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Inches(margin[0]); tf.margin_top = tf.margin_bottom = Inches(margin[1])


def tbox(sl, x, y, w, h, paras, anchor=TOP, margin=(0.04, 0.03), name=None, **kw):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: tb.name = name
    _frame(tb.text_frame, anchor, margin)
    fill(tb.text_frame, paras if isinstance(paras, list) else [paras], **kw)
    return tb


def shp(sl, x, y, w, h, fill_c=None, line=None, lw=1.0, kind=RECT, radius=0.1, name=None, paras=None, anchor=MID,
        margin=(0.12, 0.08), dash=False, rot=0, **kw):
    s = sl.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if name: s.name = name
    if fill_c: s.fill.solid(); s.fill.fore_color.rgb = rgb(fill_c)
    else: s.fill.background()
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(lw)
        if dash: s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    else: s.line.fill.background()
    if kind == ROUND: s.adjustments[0] = min(0.5, radius / min(w, h))
    if rot: s.rotation = rot
    s.shadow.inherit = False
    er = s._element.find(".//" + qn("a:effectRef"))
    if er is not None: er.set("idx", "0")
    _frame(s.text_frame, anchor, margin)
    if paras is not None: fill(s.text_frame, paras if isinstance(paras, list) else [paras], **kw)
    return s


def num_circle(sl, x, y, n, d=0.46, fill_c=NAVY, color=WHITE, size=None):
    return shp(sl, x, y, d, d, fill_c, kind=OVAL, paras=[[(str(n), {})]], align=CENTER, margin=(0, 0),
               size=size or round(d * 30), bold=True, color=color, name=f"Number {n}")


def arrow(sl, x1, y1, x2, y2, color=GRAY, lw=1.75, head=True, dash=False, name=None):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color); c.line.width = Pt(lw)
    if dash: c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if head:
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    if name: c.name = name
    return c


def pic(sl, fname, x, y, w, h, alt):
    """Picture fitted (aspect kept) and centred in the box; returns its rectangle."""
    path = os.path.join(FIG, fname); iw, ih = Image.open(path).size; ar = iw / ih
    pw, ph = (h * ar, h) if w / h > ar else (w, w / ar)
    px, py = x + (w - pw) / 2, y + (h - ph) / 2
    p = sl.shapes.add_picture(path, Inches(px), Inches(py), Inches(pw), Inches(ph))
    p.name = fname.rsplit(".", 1)[0]; p._element.nvPicPr.cNvPr.set("descr", alt)
    return px, py, pw, ph


def table(sl, x, y, col_w, rows, row_h=0.45, size=12.5, header=True, head_fill=NAVY, name=None, cell_fill=None,
          bold_first=False, align_cols=None):
    """rows: list of rows; a cell is a str or a list of paragraphs (as in fill). Thin rules between rows, no grid."""
    n, m = len(rows), len(rows[0])
    gf = sl.shapes.add_table(n, m, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * n))
    if name: gf.name = name
    tbl = gf.table
    tbl._tbl.tblPr.find(qn("a:tableStyleId")).text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"   # No Style, No Grid
    tbl.first_row = header; tbl.horz_banding = False
    for j, cw in enumerate(col_w): tbl.columns[j].width = Inches(cw)
    for i in range(n):
        tbl.rows[i].height = Inches(row_h)
        for j in range(m):
            c = tbl.cell(i, j); val = rows[i][j]
            c.margin_left = c.margin_right = Inches(0.1); c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MID; c.text_frame.word_wrap = True
            head = header and i == 0
            paras = [val] if isinstance(val, str) else val
            al = (align_cols or {}).get(j, LEFT)
            fill(c.text_frame, paras, size=size, color=WHITE if head else TEXT, bold=head or (bold_first and j == 0),
                 align=al, spacing=1.1)
            fc = head_fill if head else ((cell_fill or {}).get((i, j)) or WHITE)
            c.fill.solid(); c.fill.fore_color.rgb = rgb(fc)
            tcPr = c._tc.get_or_add_tcPr()                              # bottom rule; borders precede the fill
            ln = tcPr.makeelement(qn("a:lnB"), {"w": str(int(Pt(0.75))), "cmpd": "sng"})
            sf = ln.makeelement(qn("a:solidFill"), {}); sf.append(sf.makeelement(qn("a:srgbClr"), {"val": LINE}))
            ln.append(sf); tcPr.insert(0, ln)
    return tbl


def bg(s, color):
    f = s.background.fill; f.solid(); f.fore_color.rgb = rgb(color)


def title(s, paras, x=L, y=0.62, w=CW, h=1.0, size=26, color=NAVY, spacing=1.0, align=LEFT, anchor=TOP):
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(x), Inches(y), Inches(w), Inches(h)
    _frame(t.text_frame, anchor, (0.02, 0.02))
    fill(t.text_frame, paras if isinstance(paras, list) else [paras], size=size, color=color, bold=True,
         spacing=spacing, align=align)


def slide_no(s, color=GRAY):
    tb = s.shapes.add_textbox(Inches(11.73), Inches(7.03), Inches(1.0), Inches(0.28)); tb.name = "Slide number"
    _frame(tb.text_frame, MID, (0.04, 0.02))
    p = tb.text_frame.paragraphs[0]; p.alignment = RIGHT
    fld = p._p.makeelement(qn("a:fld"), {"id": "{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}", "type": "slidenum"})
    rPr = fld.makeelement(qn("a:rPr"), {"lang": "en-US", "sz": "1000"})
    sf = rPr.makeelement(qn("a:solidFill"), {}); sf.append(sf.makeelement(qn("a:srgbClr"), {"val": color}))
    rPr.append(sf); rPr.append(rPr.makeelement(qn("a:latin"), {"typeface": LATIN}))
    t = fld.makeelement(qn("a:t"), {}); t.text = str(len(prs.slides))
    fld.append(rPr); fld.append(t)
    end = p._p.find(qn("a:endParaRPr"))
    if end is not None: end.addprevious(fld)
    else: p._p.append(fld)


def footer(s, dark=False):
    tbox(s, L, 7.03, 8.0, 0.28, "王獻霆｜Arc Virtual Cell Challenge 2026 期末成果報告", size=10, color=ON_DARK3 if dark else GRAY,
         anchor=MID, name="Footer")
    slide_no(s, ON_DARK3 if dark else GRAY)


def notes(s, txt):
    s.notes_slide.notes_text_frame.text = txt


def content(sec, ttl, note, tag=None):
    """White content slide: section tag + breadcrumb, claim title, footer, speaker notes. sec None = no breadcrumb."""
    s = prs.slides.add_slide(LAYOUT); bg(s, WHITE)
    label = tag or (f"{sec + 1:02d}｜{SECTIONS[sec][0]}" if sec is not None else None)
    if label: tbox(s, L, 0.28, 5.5, 0.32, label, size=12, bold=True, color=TEAL_TX, anchor=MID, name="Section tag")
    if sec is not None:
        runs = []
        for i, b in enumerate(BREAD):
            if i: runs.append(("　·　", {"color": LIGHT}))
            runs.append((b, {"color": NAVY if i == sec else LIGHT, "bold": i == sec}))
        tbox(s, 6.3, 0.28, 6.433, 0.32, [runs], size=11, align=RIGHT, anchor=MID, name="Breadcrumb")
    title(s, ttl)
    footer(s); notes(s, note)
    return s


def divider(sec, note):
    s = prs.slides.add_slide(LAYOUT); bg(s, DARK)
    tbox(s, 0.9, 1.35, 4.0, 1.5, f"{sec + 1:02d}", size=88, bold=True, color=TEAL_LT, name="Section number")
    title(s, SECTIONS[sec][0], x=0.9, y=3.2, w=7.2, h=0.9, size=40, color=WHITE)
    tbox(s, 0.9, 4.25, 7.0, 1.0, SECTIONS[sec][1], size=20, color=ON_DARK, spacing=1.3, name="Guiding question")
    for i, (nm, _) in enumerate(SECTIONS):
        y = 1.95 + i * 0.8; cur = i == sec
        shp(s, 8.75, y + 0.15, 0.16, 0.16, TEAL_LT if cur else "3A5275", name="Outline marker")
        tbox(s, 9.1, y, 3.7, 0.46, f"{i + 1:02d}　{nm}", size=18, bold=cur, color=WHITE if cur else ON_DARK3, anchor=MID)
    footer(s, dark=True); notes(s, note)
    return s


def chips(sl, x, y, maxw, items, line_c, size=11.5, h=0.36, gap=0.08, vgap=0.1):
    """Wrapped row of rounded labels; returns the bottom y."""
    cx, cy = x, y
    for it in items:
        w = text_w(it, size) + 0.3
        if cx + w > x + maxw and cx > x: cx, cy = x, cy + h + vgap
        shp(sl, cx, cy, w, h, blend(line_c, WHITE, 0.22), line=line_c, lw=1.0, kind=ROUND, radius=0.18,
            paras=[[(it, {})]], align=CENTER, margin=(0.04, 0.0), size=size, color=TEXT, name=f"Chip {it}")
        cx += w + gap
    return cy + h


patch_theme()

# ---------------------------------------------------------------------------------------------------------------
# 1. title
s = prs.slides.add_slide(LAYOUT); bg(s, DARK)
tbox(s, 0.75, 0.85, 7.6, 0.4, [[("ARC VIRTUAL CELL CHALLENGE 2026", {"bold": True}), ("　｜　期末成果報告", {})]],
     size=13, color=TEAL_LT, name="Event")
title(s, ["以可遷移的擾動預測模型", "預測新細胞類型的 CRISPRi 反應"], x=0.75, y=1.4, w=7.8, h=1.75, size=34, color=WHITE,
      spacing=1.12)
tbox(s, 0.75, 3.3, 7.3, 0.75, "Predicting CRISPRi responses in unseen cell types with a transferable perturbation model",
     size=15, color=ON_DARK, spacing=1.2, name="English title")
tbox(s, 0.75, 4.55, 7.0, 1.5, [[("報告人　", {"color": TEAL_LT}), ("王獻霆", {})],
                               [("指導教授　", {"color": TEAL_LT}), ("黃宣誠 教授", {})],
                               [("國立臺灣大學 生醫電子與資訊學研究所", {"color": ON_DARK})]], size=17, color=WHITE, spacing=1.45,
     name="Presenter")
tbox(s, 0.75, 6.45, 5.0, 0.4, "2026 年 10 月 24 日", size=13, color=ON_DARK3, name="Date")
grp = s.shapes.add_group_shape(); grp.name = "Decor: response matrix"
rng = np.random.default_rng(7); NR, NC = 10, 13
M = (np.outer(rng.normal(size=NR), rng.normal(size=NC)) * 0.55 + np.outer(rng.normal(size=NR), rng.normal(size=NC)) * 0.35
     + rng.normal(scale=0.22, size=(NR, NC)))
M = np.clip(M / np.abs(M).max() * 1.5, -1, 1)
for i in range(NR):
    for j in range(NC):
        v = M[i, j]
        c = blend(WHITE, DARK, 0.07) if abs(v) < 0.12 else blend(TEAL if v > 0 else ORANGE, DARK, 0.15 + 0.8 * abs(v))
        shp(grp, 8.55 + j * 0.335, 1.15 + i * 0.335, 0.27, 0.27, c, kind=ROUND, radius=0.05)
tbox(s, 8.55, 4.6, 4.3, 0.35, "擾動 × 基因的反應矩陣（示意）", size=10.5, color=ON_DARK3, name="Decor caption")
notes(s, "各位老師好，今天報告我們參加 Arc Virtual Cell Challenge 2026 的期末成果。題目是：用一個可以遷移的模型，"
         "去預測「沒有做過實驗的新細胞類型」中，關掉某個基因之後細胞會怎麼變化。右邊的方格是示意圖：每一列是一個被關掉的基因，"
         "每一行是一個被量測的基因，綠色代表上升、橘色代表下降，這就是我們要預測的東西。")

# 2. outline
s = content(None, "大綱", "今天的報告分成五個部分：先說明研究背景與要解決的問題，再介紹相關研究和我們之前的嘗試，"
            "接著是方法、實驗結果，最後是結論與未來工作。右邊是一句話的摘要和三個最重要的數字，等一下會逐一說明。")
for i, (n, t, d) in enumerate([("01", "緒論", "研究背景、挑戰、研究問題、任務與評分方式"),
                               ("02", "相關研究", "現有擾動預測方法與我們的前期實驗"),
                               ("03", "研究方法", "訓練資料、AdaptNet、訓練方式、GeneNet4、產生細胞"),
                               ("04", "實驗結果", "三種驗證、消融實驗、瓶頸分析、未採用的方法"),
                               ("05", "結論與未來工作", "貢獻、限制與下一步")]):
    y = 1.85 + i * 0.97
    tbox(s, L, y, 1.0, 0.75, n, size=30, bold=True, color=TEAL)
    tbox(s, 1.6, y + 0.02, 5.9, 0.45, t, size=21, bold=True, color=NAVY)
    tbox(s, 1.6, y + 0.48, 5.9, 0.4, d, size=13.5, color=MUTED)
shp(s, 7.95, 1.85, 4.78, 4.8, NAVY, kind=ROUND, radius=0.12, name="Summary panel")
tbox(s, 8.3, 2.1, 4.1, 0.35, "一句話摘要", size=13, bold=True, color=TEAL_LT)
tbox(s, 8.3, 2.5, 4.15, 1.5, "我們訓練一個只有約 2 萬個參數的小模型，把 15 個公開 CRISPRi 實驗的知識，轉移到沒有任何擾動資料的新細胞類型。",
     size=16.5, color=WHITE, spacing=1.3)
for k, (big, lab) in enumerate([("+37%", "藏起細胞類型時的方向相似度（AdaptNet＋RPE1 資料，相對原方法）"), ("0.306", "官方評分模擬考總分（原方法 0.272）"),
                                ("v5", "期末版本已於 10/24 上傳")]):
    y = 4.2 + k * 0.78
    tbox(s, 8.3, y, 1.45, 0.6, big, size=24, bold=True, color=TEAL_LT, anchor=MID)
    tbox(s, 9.75, y, 2.8, 0.6, lab, size=12.5, color=ON_DARK, anchor=MID, spacing=1.1)

# ---------------------------------------------------------------------------------------------------------------
# 01 緒論
divider(0, "第一部分是緒論：先說明這個實驗是什麼、為什麼值得預測，再定義比賽要我們預測什麼、分數怎麼算。")

s = content(0, "CRISPRi 擾動實驗：一次量測「關掉一個基因」後，數千個基因的變化",
            "這一頁介紹我們要預測的實驗。CRISPRi 是一種把特定基因的表現量『調低』的技術，它不切斷 DNA，而是把基因關小聲。"
            "把它和單細胞定序結合（Perturb-seq），一次實驗就能看到：關掉某個基因後，每顆細胞裡一萬八千多個基因各自怎麼變化。"
            "和沒被處理的對照組比較，就知道哪些基因上升、哪些下降。問題是每換一種細胞就要重做，又貴又慢，"
            "所以我們想用電腦，從已經做過的實驗推測新細胞的反應。")
steps = [("1", "細胞群", "一群同類型的細胞，例如 K562"),
         ("2", "關掉基因 X", "用 CRISPRi 降低目標基因的表現（不切斷 DNA）"),
         ("3", "單細胞定序", "每顆細胞量 18,533 個基因的讀數"),
         ("4", "與對照組比較", "哪些基因上升、哪些基因下降")]
for k, (n, lab, desc) in enumerate(steps):
    x = L + k * 3.19; cx, cy = x + 1.275, 2.675
    shp(s, cx - 0.725, cy - 0.725, 1.45, 1.45, PANEL, kind=OVAL, name=f"Icon {n}")
    if k == 0:
        for dx, dy in [(-0.3, -0.24), (0.06, -0.33), (0.36, -0.06), (-0.36, 0.14), (0.0, 0.08), (0.2, 0.34), (-0.14, 0.4)]:
            shp(s, cx + dx - 0.15, cy + dy - 0.15, 0.3, 0.3, TEAL_BG, line=TEAL, lw=1.25, kind=OVAL)
    elif k == 1:
        shp(s, cx - 0.52, cy - 0.36, 1.04, 0.32, NAVY, kind=ROUND, radius=0.08, paras=[[("基因 X", {})]], align=CENTER,
            margin=(0, 0), size=11, bold=True, color=WHITE)
        shp(s, cx - 0.17, cy + 0.04, 0.34, 0.42, ORANGE, kind=MSO_SHAPE.DOWN_ARROW)
    elif k == 2:
        r2 = np.random.default_rng(3)
        for i in range(4):
            for j in range(5):
                shp(s, cx - 0.455 + j * 0.19, cy - 0.36 + i * 0.19, 0.15, 0.15, blend(NAVY, WHITE, 0.15 + 0.75 * r2.random()))
    else:
        arrow(s, cx - 0.52, cy + 0.04, cx + 0.52, cy + 0.04, color=GRAY, lw=1.25, head=False)
        for j, hgt in enumerate([0.36, -0.22, 0.18, -0.4, -0.12]):
            bx = cx - 0.43 + j * 0.19
            if hgt > 0: shp(s, bx, cy + 0.04 - hgt, 0.13, hgt, TEAL)
            else: shp(s, bx, cy + 0.04, 0.13, -hgt, ORANGE)
    tbox(s, x, 3.55, 2.55, 0.42, f"{n}　{lab}", size=16.5, bold=True, color=NAVY, align=CENTER)
    tbox(s, x, 4.0, 2.55, 0.9, desc, size=13.5, color=MUTED, align=CENTER, spacing=1.25)
    if k < 3: arrow(s, x + 2.62, cy, x + 3.12, cy, color=GRAY, lw=2)
shp(s, L, 5.3, 5.9, 1.4, PANEL, kind=ROUND, radius=0.1, margin=(0.28, 0.1), name="Limitation",
    paras=[{"runs": [("限制", {"bold": True, "color": ORANGE_TX})], "after": 4},
           "每換一種細胞就要重做一次實驗：昂貴、耗時，而且一次只能測有限數量的基因。"], size=14, spacing=1.25)
shp(s, 6.833, 5.3, 5.9, 1.4, TEAL_BG, kind=ROUND, radius=0.1, margin=(0.28, 0.1), name="Motivation",
    paras=[{"runs": [("研究動機", {"bold": True, "color": TEAL_TX})], "after": 4},
           "能不能用已經做過的公開實驗，預測「還沒做過實驗的細胞類型」會怎麼反應？"], size=14, spacing=1.25)

s = content(0, "挑戰：新細胞類型完全沒有擾動資料，只能從別的細胞「轉移」知識",
            "左邊是我們手上有的資料：15 個公開實驗、9 類細胞，依照細胞種類分成血球、上皮、幹細胞三大類。"
            "右邊是期末考的三種細胞 D、E、F，主辦單位只給沒被處理過的對照組細胞，完全沒有擾動資料，所以模型必須把別的細胞學到的東西轉移過去。"
            "難的地方有三個：沒有任何樣本可以參考；D、E、F 和訓練細胞差很多；而且期末考的基因大多是非必需基因，被關掉後變化很小、雜訊很大。")
shp(s, L, 1.85, 6.35, 3.15, PANEL, kind=ROUND, radius=0.1, name="Known panel")
tbox(s, 0.85, 1.98, 5.9, 0.4, "已知：15 個公開實驗（9 類細胞）", size=15, bold=True, color=NAVY)
yy = 2.55
for cat, items in [("血球類", ["K562", "K562ess", "VIPK562", "Jurkat", "CD4 休息", "CD4 刺激 8h", "CD4 刺激 48h"]),
                   ("上皮類", ["HCT116", "HEK293T", "HepG2", "RPE1"]), ("幹細胞類", ["H1", "KOLF", "FengNF", "FengF"])]:
    shp(s, 0.88, yy + 0.1, 0.16, 0.16, CAT[cat], kind=OVAL)
    tbox(s, 1.1, yy, 1.0, 0.36, cat, size=12, bold=True, color=TEXT, anchor=MID)
    yy = chips(s, 2.1, yy, 4.7, items, CAT[cat]) + 0.16
shp(s, 7.08, 3.15, 0.75, 0.55, TEAL, kind=MSO_SHAPE.RIGHT_ARROW, name="Transfer arrow")
tbox(s, 6.95, 2.7, 1.0, 0.4, "轉移", size=13, bold=True, color=TEAL_TX, align=CENTER)
shp(s, 7.95, 1.85, 4.78, 3.15, ORANGE_BG, kind=ROUND, radius=0.1, name="Unknown panel")
tbox(s, 8.2, 1.98, 4.3, 0.4, "未知：期末考 D、E、F", size=15, bold=True, color=ORANGE_TX)
for k, (c, d) in enumerate([("D", "類骨髓細胞（myeloid）"), ("E", "間質細胞（類纖維母細胞）"), ("F", "上皮／間質混合型")]):
    y = 2.52 + k * 0.62
    shp(s, 8.2, y, 3.65, 0.5, WHITE, line=ORANGE, kind=ROUND, radius=0.12, margin=(0.15, 0),
        paras=[[(c, {"bold": True, "size": 15, "color": NAVY}), ("　" + d, {"size": 12.5, "color": TEXT})]])
    shp(s, 12.0, y + 0.04, 0.42, 0.42, ORANGE, kind=OVAL, paras=[[("?", {})]], align=CENTER, margin=(0, 0), size=15,
        bold=True, color=WHITE)
tbox(s, 8.2, 4.42, 4.4, 0.4, "只有未處理的對照組細胞（各 18,400 顆）", size=12, color=MUTED)
for k, (h, d) in enumerate([("零樣本", "新細胞沒有任何擾動資料，只能從別的細胞轉移知識"),
                            ("細胞差異大", "D 最接近的訓練細胞（CD4、K562）相關也只有 0.26–0.35"),
                            ("訊號弱、雜訊大", "300 個目標多為非必需基因：被關掉後變化小，單細胞讀數又很稀疏")]):
    x = COL3[k]
    num_circle(s, x, 5.32, k + 1, d=0.48, fill_c=ORANGE)
    tbox(s, x + 0.62, 5.3, 3.25, 0.42, h, size=15.5, bold=True, color=NAVY)
    tbox(s, x + 0.62, 5.72, 3.25, 1.05, d, size=13, color=TEXT, spacing=1.25)

s = content(0, "研究問題：能否用一個通用模型，預測任何細胞中任何基因被關掉後的反應？",
            "這是我們的研究問題。我們的目標不是只把期末考考好，而是做出一個通用的模型：給它任何一種細胞、任何一個基因，都能給出預測。"
            "為了確認模型真的通用，我們設計三種驗證方式交叉確認；最後才是在期末考的 D、E、F 三種細胞上提交預測。"
            "下方是我們一直遵守的原則：D、E、F 只是一次考試，每個改動都要在其他細胞上也成立，才會放進模型。")
for k, (h, d) in enumerate([("通用模型", "任何細胞類型、任何基因都能預測：有測量的基因用 AdaptNet，沒測過的基因用 GeneNet4"),
                            ("嚴格驗證", "藏起整種細胞、藏起一批基因、官方評分模擬考，三種方式交叉確認"),
                            ("期末提交", "在 D、E、F 三種新細胞上產生 36 萬顆預測細胞，截止前完成上傳")]):
    x = COL3[k]
    shp(s, x, 1.95, 3.9, 3.15, PANEL, kind=ROUND, radius=0.12, name=f"Goal {k + 1}")
    num_circle(s, x + 0.3, 2.2, k + 1, d=0.62)
    tbox(s, x + 0.3, 2.95, 3.3, 0.45, f"目標 {k + 1}｜{h}", size=19, bold=True, color=NAVY)
    tbox(s, x + 0.3, 3.5, 3.3, 1.55, d, size=15.5, color=TEXT, spacing=1.3)
shp(s, L, 5.4, CW, 1.25, TEAL_BG, kind=ROUND, radius=0.1, margin=(0.3, 0.1), name="Principle",
    paras=[[("核心原則　", {"bold": True, "color": TEAL_TX}),
            ("D、E、F 只是一次考試。每個改動都必須在其他細胞上同樣成立才放進模型——目標是通用，而不是只對期末考調參數。", {})]],
    size=15, spacing=1.3)

s = content(0, "任務定義：輸入新細胞的對照組與目標基因，輸出 400 顆被擾動的細胞",
            "把比賽寫成正式的任務：輸入有兩個，一是新細胞類型 c 的對照組細胞，每種細胞一萬八千四百顆、每顆量一萬八千多個基因；"
            "二是要關掉的目標基因 p，三種細胞考同一組 300 個基因。模型要輸出 400 顆『假裝被關掉 p』的細胞，每個值是整數讀數。"
            "總共 3 種細胞乘 300 個基因，是 900 組預測、36 萬顆細胞。上傳每天最多兩次，而且只算最後一次。")
shp(s, L, 1.95, 3.85, 1.42, PANEL, kind=ROUND, radius=0.1, margin=(0.22, 0.08), name="Input 1",
    paras=[{"runs": [("輸入 1｜對照組細胞", {"bold": True, "color": NAVY, "size": 15})], "after": 3},
           [("X", {"bold": True, "size": 16}), ("c", {"bold": True, "size": 16, "pos": "sub"}),
            ("：18,400 顆細胞 × 18,533 個基因", {"size": 13.5})],
           [("新細胞類型 c 中，沒有被擾動的細胞", {"size": 12.5, "color": MUTED})]])
shp(s, L, 3.6, 3.85, 1.42, PANEL, kind=ROUND, radius=0.1, margin=(0.22, 0.08), name="Input 2",
    paras=[{"runs": [("輸入 2｜目標基因", {"bold": True, "color": NAVY, "size": 15})], "after": 3},
           [("p", {"bold": True, "italic": True, "size": 16}), ("：要關掉的基因（共 300 個）", {"size": 13.5})],
           [("三種細胞考同一組 300 個基因", {"size": 12.5, "color": MUTED})]])
shp(s, 5.25, 2.45, 2.85, 2.05, NAVY, kind=ROUND, radius=0.12, name="Model", align=CENTER,
    paras=[{"runs": [("模型 f", {"size": 24, "bold": True, "color": WHITE})], "after": 4},
           [("AdaptNet ＋ 產生細胞", {"size": 13, "color": TEAL_LT})]])
shp(s, 8.9, 2.12, 3.833, 2.7, TEAL_BG, kind=ROUND, radius=0.1, margin=(0.22, 0.1), name="Output",
    paras=[{"runs": [("輸出｜被擾動的細胞", {"bold": True, "color": TEAL_TX, "size": 15})], "after": 3},
           [("Ŷ", {"bold": True, "size": 16}), ("c,p", {"bold": True, "size": 16, "pos": "sub"}),
            ("：400 顆細胞 × 18,533 個基因", {"size": 13.5})],
           [("每個值是整數讀數（UMI），直接用官方程式評分", {"size": 12.5, "color": MUTED})]], spacing=1.2)
arrow(s, 4.5, 2.66, 5.2, 3.2); arrow(s, 4.5, 4.31, 5.2, 3.75); arrow(s, 8.15, 3.475, 8.85, 3.475)
for k, (big, lab) in enumerate([("3", "種新細胞類型：D、E、F"), ("900", "組（細胞類型, 基因）要預測"),
                                ("36 萬", "顆細胞要產生並上傳"), ("2 次／天", "上傳上限，只算最後一次")]):
    x = L + k * 3.08
    tbox(s, x, 5.3, 2.9, 0.7, big, size=30, bold=True, color=NAVY)
    tbox(s, x, 6.0, 2.9, 0.6, lab, size=12.5, color=MUTED)

s = content(0, "評分方式：六項指標各自換算成「0 = 平均基準、1 = 實驗重複」再取平均",
            "分數怎麼算：官方用六項指標，每一項都換算成同一個尺度——0 分代表和『一律猜這種細胞的平均反應』一樣，"
            "1 分代表和『把真實實驗分成兩半互相預測』一樣好，也就是實驗本身的水準。六項分成兩類：前兩項只看每個基因的總量，"
            "後四項看哪些基因『顯著改變』，需要用每顆細胞的分布做統計檢定。我們在 H1 模擬考的總分是 0.306。")
shp(s, L, 1.95, 5.5, 1.2, NAVY, kind=ROUND, radius=0.1, align=CENTER, name="Formula",
    paras=[[("換算分數 = (s − s", {}), ("基準", {"pos": "sub"}), (") ÷ (s", {}), ("重複", {"pos": "sub"}),
            (" − s", {}), ("基準", {"pos": "sub"}), (")", {})]], size=19, bold=True, color=WHITE)
tbox(s, L, 3.35, 5.5, 1.4, [{"runs": [("s", {"bold": True}), ("基準", {"bold": True, "pos": "sub"}),
                                      ("　一律預測「這種細胞的平均反應」", {})], "after": 8},
                            {"runs": [("s", {"bold": True}), ("重複", {"bold": True, "pos": "sub"}),
                                      ("　真實細胞分成兩半、互相預測（實驗本身的水準）", {})]}], size=14.5, spacing=1.3)
X0, X1, V0, V1 = 0.85, 5.85, -0.15, 1.1
xv = lambda v: X0 + (v - V0) / (V1 - V0) * (X1 - X0)
arrow(s, X0, 5.75, X1, 5.75, color=GRAY, lw=2, head=False, name="Number line")
for v, lab in [(0, "0｜平均基準"), (1, "1｜實驗重複")]:
    arrow(s, xv(v), 5.62, xv(v), 5.88, color=GRAY, lw=2, head=False)
    tbox(s, xv(v) - 0.9, 5.95, 1.8, 0.35, lab, size=12.5, color=MUTED, align=CENTER)
shp(s, xv(0.306) - 0.13, 5.38, 0.26, 0.3, TEAL, kind=MSO_SHAPE.ISOSCELES_TRIANGLE, rot=180, name="Our score")
tbox(s, xv(0.306) - 1.3, 4.98, 2.6, 0.36, "我們：0.306（H1 模擬考）", size=12.5, bold=True, color=TEAL_TX, align=CENTER)
mt = table(s, 6.45, 1.95, [1.15, 2.05, 3.083],
           [["類別", "指標", "看什麼"],
            ["總量", "區分度（pds）", "能否分辨「關掉 A」與「關掉 B」"],
            ["", "整體誤差（mse）", "每個基因的平均表現量準不準"],
            ["顯著基因", "方向正確（fid）", "判定顯著的基因，升降方向對不對"],
            ["", "方向排序（reach）", "真正顯著的基因，是否排在前面"],
            ["", "清單重疊（jaccard）", "顯著基因清單和真實清單的重疊"],
            ["", "變化大小（nmae）", "顯著基因的變化倍數誤差"]], row_h=0.5, size=12.5, name="Metric table",
           cell_fill={(1, 0): blend(NAVY, WHITE, 0.1), (3, 0): TEAL_BG})
mt.cell(1, 0).merge(mt.cell(2, 0)); mt.cell(3, 0).merge(mt.cell(6, 0))
for (r, c_) in [(1, 0), (3, 0)]:
    for run in mt.cell(r, c_).text_frame.paragraphs[0].runs:
        run.font.bold = True; run.font.color.rgb = rgb(NAVY if r == 1 else TEAL_TX)
tbox(s, 6.45, 5.6, 6.283, 0.9, "總量類只看每個基因在 400 顆細胞的總和；顯著基因類用每顆細胞的分布做統計檢定（Wilcoxon），判斷哪些基因真的改變。",
     size=12, color=MUTED, spacing=1.25)

# ---------------------------------------------------------------------------------------------------------------
# 02 相關研究
divider(1, "第二部分是相關研究：先整理現有的擾動預測方法，再說明我們一開始的嘗試，以及從中學到的設計原則。")

s = content(1, "相關研究：多數方法需要目標細胞的擾動資料，跨細胞類型預測仍是難題",
            "這頁整理幾個代表性的方法。GEARS 和 CPA 是早期的深度學習方法，主要在同一種細胞裡預測新的基因組合，或需要該細胞的部分資料。"
            "scGPT 這類基礎模型需要在目標資料上微調；主辦單位 Arc 的 State 模型可以跨情境，但需要大量資料和運算。"
            "我們一開始重現了 scDFM，在新細胞上接近亂猜。另外 2025 年 Nature Methods 的研究指出，深度模型還沒有穩定勝過簡單的線性基準，"
            "這讓我們決定從小而可驗證的模型開始。")

def cite(name, ref):
    return [[(name, {"bold": True, "color": NAVY, "size": 14})], [(ref, {"size": 10.5, "color": MUTED})]]

table(s, L, 1.85, [2.75, 3.15, 3.0, 3.233],
      [["方法", "核心做法", "能否預測新細胞類型", "與本研究的關係"],
       [cite("GEARS", "Roohani et al., 2024"), "基因關係圖＋圖神經網路", "否：在同一種細胞內預測新的基因組合", "借鏡基因知識圖 → GeneNet4"],
       [cite("CPA", "Lotfollahi et al., 2023"), "自編碼器，拆開細胞狀態與擾動效果", "有限：需要該細胞類型的部分擾動資料", "細胞狀態調整擾動效果 → AdaptNet"],
       [cite("scGPT", "Cui et al., 2024"), "單細胞基礎模型，預訓練後微調", "需要在目標資料上微調", "大型預訓練需要大量同類資料"],
       [cite("State", "Adduri et al., 2025"), "大型 Transformer，以細胞群為單位預測", "可以，但需要大量資料與運算", "主辦單位 Arc 的模型；我們改走小模型"],
       [cite("scDFM", "Yu et al., ICLR 2026"), "Flow matching 生成單細胞", "我們重現：新細胞上接近亂猜", "本研究的起點（見下頁）"],
       [cite("線性基準", "Ahlmann-Eltze et al., 2025"), "簡單線性模型、平均法", "—", "深度模型尚未穩定勝過簡單基準 → 從小模型出發"]],
      row_h=0.6, size=12.5, name="Related work table")
shp(s, L, 6.2, CW, 0.6, WHITE, line=NAVY, lw=1.25, kind=ROUND, radius=0.1, margin=(0.25, 0.04), name="Positioning",
    paras=[[("本研究定位　", {"bold": True, "color": NAVY}),
            ("小模型（約 2 萬個參數）＋ 跨實驗知識轉移 ＋「藏起整種細胞」的嚴格驗證", {})]], size=14.5)

s = content(1, "前期實驗：大型生成模型學不到擾動訊號，簡單的跨細胞平均反而有效",
            "這是我們在九、十月初做的前期實驗。第一個是照論文設定重現 scDFM 生成模型，訓練兩萬步，在藏起來的 H1 細胞上區分度一直是 0.5，等於亂猜，"
            "連訓練過的 K562 都學不到。第二個是讓每個基因都有自己參數的網路，訓練越久在新細胞上越差，代表它把訓練細胞背起來了。"
            "反而最簡單的『把其他細胞的反應平均起來』拿到官方總分 0.267。所以我們的設計原則是：不要從零學每個基因，只學怎麼組合已有的測量。")
for k, (h, big, bc, lab, extra) in enumerate([
        ("scDFM 生成模型（重現論文設定，訓練 2 萬步）", "0.50", ORANGE_TX, "藏起 H1 時的區分度＝亂猜，整個訓練過程都沒有進步",
         ["連訓練過的 K562 都學不到（相關 0.03）", "產生的細胞幾乎沒有 0（真實資料 76% 是 0）"]),
        ("每個基因自由學習的網路（ResponseNet）", "0.76 → 0.69", ORANGE_TX, "訓練越久，藏起 H1 的區分度越差：把訓練細胞「背起來」",
         ["同一個測試，簡單平均法為 0.83"]),
        ("跨細胞平均（簡單基準）", "0.267", TEAL_TX, "藏起 H1 的官方評分總分（平均反應放大 1.5 倍）",
         ["區分度 0.72、方向正確 0.67（換算分數）"])]):
    x = COL3[k]
    shp(s, x, 1.95, 3.9, 3.5, PANEL, kind=ROUND, radius=0.12, name=f"Pilot {k + 1}")
    tbox(s, x + 0.25, 2.1, 3.4, 0.75, h, size=14.5, bold=True, color=NAVY, spacing=1.15)
    tbox(s, x + 0.25, 2.85, 3.4, 0.8, big, size=34, bold=True, color=bc, anchor=MID)
    tbox(s, x + 0.25, 3.7, 3.4, 0.75, lab, size=13, color=TEXT, spacing=1.2)
    tbox(s, x + 0.25, 4.5, 3.4, 0.9, [{"runs": e, "bullet": True, "after": 2} for e in extra], size=11.5, color=MUTED,
         spacing=1.15)
shp(s, L, 5.7, CW, 1.0, TEAL_BG, kind=ROUND, radius=0.1, margin=(0.3, 0.06), name="Design principle",
    paras=[[("設計原則　", {"bold": True, "color": TEAL_TX}),
            ("模型不從零學每個基因，只學「怎麼組合、調整已有的測量」→ 參數少、不易過度擬合 → AdaptNet", {})]], size=15)

# ---------------------------------------------------------------------------------------------------------------
# 03 研究方法
divider(2, "第三部分是方法：先看整體架構，再依序介紹訓練資料、核心模型 AdaptNet、訓練方式、處理沒測過基因的 GeneNet4，以及怎麼產生細胞。")

s = content(2, "整體架構：從公開實驗學會「如何轉移」，再為新細胞產生 400 顆細胞",
            "這是整個系統的流程。第一步把 15 個公開實驗整理成同一種格式；第二步對每個要預測的基因，找出它在其他細胞裡被關掉時的測量，我們叫它『來源』；"
            "第三步是核心模型 AdaptNet，它看新細胞的對照組長什麼樣子，決定要相信哪些來源、怎麼調整，預測平均變化；第四步依預測產生 400 顆細胞；"
            "最後打包上傳。如果某個基因從來沒被任何實驗測過，就改由 GeneNet4 從功能相近的基因推測。")
stages = [("資料整合", "15 個公開實驗 → 每組（細胞, 基因）的平均變化"), ("查詢來源", "目標基因在其他細胞中的測量（來源）"),
          ("AdaptNet", "依新細胞的狀態組合、調整來源，預測平均變化"), ("產生細胞", "總量＋單細胞分布 → 每個基因 400 顆細胞"),
          ("打包上傳", "36 萬顆細胞 → .vcc 檔，官方格式檢查")]
for k, (h, d) in enumerate(stages):
    x = L + k * 2.483; core = k == 2
    shp(s, x, 2.05, 2.15, 2.0, NAVY if core else PANEL, kind=ROUND, radius=0.12, name=f"Stage {k + 1}")
    num_circle(s, x + 0.17, 2.22, k + 1, d=0.42, fill_c=TEAL_LT if core else TEAL, color=NAVY if core else WHITE)
    tbox(s, x + 0.15, 2.72, 1.9, 0.42, h, size=16, bold=True, color=WHITE if core else NAVY)
    tbox(s, x + 0.15, 3.15, 1.9, 0.85, d, size=12.5, color=ON_DARK if core else TEXT, spacing=1.2)
    if k < 4: arrow(s, x + 2.17, 3.05, x + 2.465, 3.05, color=GRAY, lw=2)
shp(s, 3.083, 4.8, 2.15, 1.1, WHITE, line=ORANGE, lw=1.25, kind=ROUND, radius=0.1, dash=True, name="GeneNet4 input",
    paras=[[("沒測過的基因", {"bold": True, "color": ORANGE_TX})], [("GeneNet4：STRING、GO 鄰居", {"size": 11.5})]],
    align=CENTER, size=13, spacing=1.15)
shp(s, 5.567, 4.8, 2.15, 1.1, WHITE, line=TEAL, lw=1.25, kind=ROUND, radius=0.1, name="Cell state input",
    paras=[[("新細胞的對照組", {"bold": True, "color": TEAL_TX})], [("提供「細胞狀態」", {"size": 11.5})]],
    align=CENTER, size=13, spacing=1.15)
arrow(s, 4.158, 4.78, 4.158, 4.1, color=ORANGE, lw=1.75); arrow(s, 6.642, 4.78, 6.642, 4.1, color=TEAL, lw=1.75)
tbox(s, L, 6.2, CW, 0.55, "訓練時每一步都藏起一種細胞來模擬「新細胞」（見後頁）；期末考的 300 個基因都有測量，主要走第 2、3 步。",
     size=12.5, color=MUTED)

s = content(2, "訓練資料：15 個公開 CRISPRi 實驗、約 3.8 萬組擾動測量",
            "訓練資料是 15 個公開的 CRISPRi 實驗，涵蓋 9 類細胞，總共約三萬八千組『某種細胞裡關掉某個基因』的測量，左圖是每個實驗的大小。"
            "每組資料的處理分四步：把同一組細胞加總後和對照組比較，算出每個基因的變化倍數；只留至少 20 顆細胞的擾動；"
            "扣掉每個實驗自己的共同反應，因為不同平台、批次會有系統性差異；最後記下對照組的平均表現，當作模型看到的細胞狀態。"
            "另外用 STRING 和 Gene Ontology 兩個基因知識庫，處理沒被測過的基因。")
pic(s, "fig_data.png", L, 1.8, 6.5, 5.05, "15 個訓練實驗的擾動數，依血球類、上皮類、幹細胞類著色")
tbox(s, 7.45, 1.9, 5.28, 0.4, "資料處理（每個實驗、每個被關掉的基因）", size=15, bold=True, color=NAVY)
for k, d in enumerate(["加總同一組細胞的讀數，和對照組比較，算出每個基因的變化倍數（取對數）",
                       "只保留至少 20 顆細胞的擾動；表現太低的基因標為缺值",
                       "扣掉該實驗所有擾動的共同反應（不同實驗平台、批次的差異）",
                       "記錄對照組的平均表現，當作模型看到的「細胞狀態」"]):
    y = 2.48 + k * 0.84
    num_circle(s, 7.45, y + 0.02, k + 1, d=0.4, fill_c=TEAL)
    tbox(s, 8.0, y - 0.03, 4.73, 0.8, d, size=13.5, color=TEXT, spacing=1.2)
shp(s, 7.45, 5.9, 5.283, 0.9, PANEL, kind=ROUND, radius=0.1, margin=(0.2, 0.06), name="Knowledge bases",
    paras=[[("基因知識庫　", {"bold": True, "color": NAVY}), ("STRING v12 蛋白交互網路、Gene Ontology 功能註解 → 推測沒測過的基因", {})]],
    size=13, spacing=1.2)

s = content(2, "AdaptNet：依新細胞的狀態，學會「相信哪個來源、如何調整」",
            "這是核心模型 AdaptNet。左邊三個輸入：目標基因在其他細胞的測量（來源）、每個來源的特徵（例如定序深度、擊倒強度、和其他來源是否一致、細胞像不像），"
            "以及新細胞的狀態。中間三個模組：先把每個來源拆成 200 個主要的基因程式加上剩下的細節；再由一個小網路給每個來源打分數，決定相信誰；"
            "第三個模組依新細胞的狀態，把每個程式放大或減弱。右邊把它們組合成預測。整個模型只有約兩萬個參數，因為它不學每個基因，只學怎麼組合來源。")
for t_, x_ in [("輸入", L), ("模組", 4.15), ("輸出", 8.05)]:
    tbox(s, x_, 1.78, 2.0, 0.3, t_, size=12, bold=True, color=GRAY)
def subs(txt, **o):
    """'w_s' -> runs with the s as subscript."""
    out = []
    for part in re.split(r"(_[a-z])", txt):
        if part.startswith("_") and len(part) == 2: out.append((part[1], dict(o, pos="sub")))
        elif part: out.append((part, dict(o)))
    return out


rows3 = [(("來源反應 y", "s"), "目標基因在其他細胞 s 的測量", "拆成基因程式", "200 個主要程式 V（SVD）→ 係數 A_s ＋ 細節 R_s"),
         (("來源特徵", ""), "定序深度、擊倒強度、來源間一致度、細胞相似度", "來源權重 w_s", "小型網路打分 → softmax，決定相信誰"),
         (("新細胞狀態 z(c)", ""), "對照組平均表現的主成分", "程式增益 g", "依細胞狀態，把每個程式調強或調弱")]



for k, ((t1, sub1), d1, t2, d2) in enumerate(rows3):
    y = 2.15 + k * 1.45; cy = y + 0.6
    head = [(t1, {"bold": True, "color": NAVY, "size": 14.5})] + ([(sub1, {"bold": True, "color": NAVY, "size": 14.5, "pos": "sub"})] if sub1 else [])
    shp(s, L, y, 2.9, 1.2, PANEL, kind=ROUND, radius=0.1, margin=(0.16, 0.06), name=f"Input {k + 1}",
        paras=[{"runs": head, "after": 2}, [(d1, {"size": 12, "color": MUTED})]], spacing=1.15)
    shp(s, 4.15, y, 3.15, 1.2, WHITE, line=TEAL, lw=1.5, kind=ROUND, radius=0.1, margin=(0.16, 0.06), name=f"Module {k + 1}",
        paras=[{"runs": subs(t2, bold=True, color=TEAL_TX, size=14.5), "after": 2}, subs(d2, size=12, color=MUTED)], spacing=1.15)
    arrow(s, 3.53, cy, 4.12, cy)
    arrow(s, 7.33, cy, 8.02, 3.2 + k * 0.6)
shp(s, 8.05, 2.35, 4.683, 2.9, NAVY, kind=ROUND, radius=0.12, margin=(0.25, 0.12), name="Prediction",
    paras=[{"runs": [("預測：目標基因在新細胞的平均變化", {"bold": True, "color": TEAL_LT, "size": 14.5})], "after": 10},
           {"runs": [("ŷ = α·[(Σ w", {}), ("s", {"pos": "sub"}), (" A", {}), ("s", {"pos": "sub"}), (") × g] V", {}),
                     ("T", {"pos": "sup"}), (" + α·β·Σ w", {}), ("s", {"pos": "sub"}), (" R", {}), ("s", {"pos": "sub"})],
            "after": 10, "size": 16, "bold": True, "color": WHITE},
           subs("A_s：程式係數　R_s：程式以外的細節　α：整體大小　×：逐項相乘", size=11.5, color=ON_DARK)],
    align=CENTER, spacing=1.2)
tbox(s, 8.05, 5.45, 4.683, 1.0, [{"runs": [("約 2 萬個參數：不學每個基因，只學怎麼組合來源", {})], "bullet": True, "after": 4},
                                 {"runs": [("期末版本：β = 1（保留全部細節），3 個模型取平均", {})], "bullet": True}],
     size=12.5, color=MUTED, spacing=1.2)

s = content(2, "訓練方式：每一步藏起一種細胞，模擬「遇到沒看過的細胞」",
            "訓練時，我們讓模型一直練習期末考的情境：每一步隨機挑一種細胞，把它藏起來當作新細胞，只能用其他細胞中同一個基因的測量去預測它。"
            "同一組實驗的兄弟細胞也不能當來源，例如藏起 CD4 休息狀態時，另外兩個 CD4 狀態也不能用。每個來源還有 20% 機率被拿掉，讓模型習慣來源不齊全。"
            "損失函數有兩項：一是預測誤差，二是對比學習，要求每個基因的預測最像它自己的答案，這直接對應評分的區分度。右邊是主要設定，一個模型大約一分鐘就訓練完。")
legend = [(ORANGE, "藏起（當作新細胞）"), (TEAL, "來源（提供測量）"), (LINE, "同組實驗（不當來源）")]
lx = L
for c_, t_ in legend:
    shp(s, lx, 2.02, 0.22, 0.22, c_, kind=OVAL); tbox(s, lx + 0.28, 1.93, 2.3, 0.4, t_, size=12, color=TEXT, anchor=MID)
    lx += 0.28 + text_w(t_, 12) + 0.45
episodes = [("第 1 步", 4, [5, 6]), ("第 2 步", 9, []), ("第 3 步", 11, []), ("…", None, []), ("第 1,500 步", 7, [8])]
for r_, (lab, hid, sib) in enumerate(episodes):
    y = 2.6 + r_ * 0.56
    tbox(s, L, y - 0.05, 1.3, 0.4, lab, size=13, bold=True, color=NAVY, anchor=MID)
    if hid is None: continue
    for j in range(15):
        c_ = ORANGE if j == hid else (LINE if j in sib else TEAL)
        shp(s, 1.95 + j * 0.38, y, 0.3, 0.3, c_, kind=OVAL)
tbox(s, L, 5.5, 7.1, 1.2, "每一步隨機藏起一種細胞，用其他細胞中同一個基因的測量去預測它；每個來源再有 20% 機率被拿掉，讓模型習慣來源不齊全。",
     size=12.5, color=MUTED, spacing=1.25)
shp(s, 8.1, 1.95, 4.633, 2.0, PANEL, kind=ROUND, radius=0.1, margin=(0.22, 0.12), anchor=TOP, name="Loss",
    paras=[{"runs": [("損失函數", {"bold": True, "color": NAVY, "size": 14.5})], "after": 6},
           {"runs": [("① 預測誤差：", {"bold": True}), ("和真實平均變化的差距（以簡單平均法為基準）", {})], "after": 6},
           {"runs": [("② 對比學習：", {"bold": True}), ("一批 96 個基因中，每個預測要最像自己的答案（對應區分度）", {})]}],
    size=13, spacing=1.2)
table(s, 8.1, 4.15, [2.0, 2.633], [["基因程式數 k", "200"], ["訓練步數", "1,500"], ["每步基因數", "96"], ["參數量", "約 2 萬"],
                                   ["模型數", "3（隨機種子 0、1、2）"], ["訓練時間", "約 1 分鐘／模型"]],
      row_h=0.42, size=12.5, header=False, bold_first=True, name="Hyperparameters")

s = content(2, "沒測過的基因：借用 32 個功能相近基因的反應來推測（GeneNet4）",
            "如果某個基因從來沒被任何實驗關掉過，就沒有來源可以參考。GeneNet4 的做法是：用 STRING 蛋白交互網路和 GO 功能註解，找出功能最像的 32 個基因，"
            "用它們被關掉時的反應，透過學到的注意力加權平均，來推測這個基因。右圖是在藏起細胞類型的測試裡的結果：有測量的基因方向相似度 0.152，"
            "沒測過的基因只有 0.025，區分度 0.572 略高於亂猜，代表有訊號但很弱。合起來就是我們的通用模型：任何細胞、任何基因都能給出預測。")
cx0, cy0, rad = 2.45, 3.5, 1.2
pts = [(cx0 + rad * np.cos(np.radians(a)), cy0 + rad * np.sin(np.radians(a))) for a in range(-68, 292, 45)]
for px_, py_ in pts: arrow(s, cx0, cy0, px_, py_, color=LIGHT, lw=1.25, head=False)
for px_, py_ in pts: shp(s, px_ - 0.27, py_ - 0.27, 0.54, 0.54, TEAL_BG, line=TEAL, lw=1.25, kind=OVAL)
shp(s, cx0 - 0.62, cy0 - 0.62, 1.24, 1.24, ORANGE, kind=OVAL, align=CENTER, margin=(0, 0), name="Unseen gene",
    paras=[[("基因 g", {"bold": True, "size": 14})], [("沒測過", {"size": 11.5})]], color=WHITE, spacing=1.05)
arrow(s, 4.05, cy0, 4.65, cy0, color=GRAY, lw=2)
shp(s, 4.72, cy0 - 0.7, 2.25, 1.4, NAVY, kind=ROUND, radius=0.12, align=CENTER, margin=(0.12, 0.08), name="Attention",
    paras=[{"runs": [("學到的注意力", {"bold": True, "size": 14, "color": WHITE})], "after": 3},
           [("加權平均鄰居的反應 → 預測 g", {"size": 11.5, "color": TEAL_LT})]], spacing=1.15)
tbox(s, L, 4.95, 6.4, 0.4, [[("●", {"color": TEAL}), (" 功能相近、有測量的基因（STRING＋GO，最多 32 個）", {})]], size=12, color=MUTED)
_, py0, _, ph0 = pic(s, "fig_genenet.png", 7.35, 1.9, 5.383, 2.75, "藏起細胞類型時，有測量與沒測過基因的方向相似度和區分度")
tbox(s, 7.35, py0 + ph0 + 0.02, 5.383, 0.35, "藏起細胞類型（7 折）；區分度 0.5 = 亂猜", size=11.5, color=MUTED, align=CENTER)
shp(s, L, 5.55, CW, 1.2, TEAL_BG, kind=ROUND, radius=0.1, margin=(0.3, 0.08), name="General model",
    paras=[[("通用模型 GeneNet4　", {"bold": True, "color": TEAL_TX}),
            ("＝ AdaptNet（有測量的基因）＋ 知識分支（沒測過的基因）：任何細胞、任何基因都能預測。沒測過的基因仍有訊號但很弱；期末 300 個基因都有測量，主要靠 AdaptNet。", {})]],
    size=14, spacing=1.3)

s = content(2, "產生細胞：「總量」與「單細胞分布」分開處理，各自對應一類評分",
            "有了預測之後要產生真正的細胞。我們發現兩類評分需要的東西不一樣，所以分兩條線處理。上面一條管『總量』：用 AdaptNet 的預測，"
            "移除 5 個對照組裡的雜訊方向、依來源一致度加權，再把整體大小調到評分最佳的位置，讓 400 顆細胞每個基因的總讀數剛好等於預測，這對應區分度和整體誤差。"
            "下面一條管『單細胞分布』：把各來源一致的變化放大 4.5 倍，總量的差額集中放進 2% 的細胞，讓真正改變的基因能通過統計檢定，這對應其餘四項。")
lanes = [("總量", "每個基因的平均", NAVY, ["AdaptNet 預測平均變化", "移除 5 個對照組雜訊方向，依來源一致度加權", "依評分規則調整整體大小"],
          ("400 顆細胞的總讀數＝預測值", "→ 區分度、整體誤差")),
         ("單細胞", "每顆細胞的分布", TEAL, ["共識模型的變化（各來源平均）", "放大 4.5 倍 × 各來源方向一致度", "總量差額集中放進 2% 的細胞"],
          ("顯著基因能通過統計檢定", "→ 方向正確、方向排序、清單重疊、變化大小"))]
for k, (lab, sub_, col, boxes, (res, met)) in enumerate(lanes):
    y = 1.95 + k * 1.9; cy = y + 0.8
    shp(s, L, y, 1.3, 1.6, col, kind=ROUND, radius=0.12, align=CENTER, margin=(0.06, 0.06), name=f"Lane {lab}",
        paras=[{"runs": [(lab, {"bold": True, "size": 17})], "after": 2}, [(sub_, {"size": 11})]], color=WHITE, spacing=1.1)
    for j, b in enumerate(boxes):
        x = 2.15 + j * 2.6
        shp(s, x, y, 2.3, 1.6, PANEL, kind=ROUND, radius=0.1, margin=(0.1, 0.08), paras=[[(b, {})]], align=CENTER,
            size=13, color=TEXT, spacing=1.2)
        arrow(s, x - 0.27 if j else 1.93, cy, x - 0.03, cy)
    shp(s, 9.95, y, 2.783, 1.6, blend(col, WHITE, 0.12), line=col, lw=1.25, kind=ROUND, radius=0.1, margin=(0.16, 0.08),
        align=CENTER, paras=[{"runs": [(res, {"bold": True, "color": TEXT})], "after": 4},
                             [(met, {"size": 11.5, "color": NAVY if col == NAVY else TEAL_TX, "bold": True})]],
        size=13, spacing=1.2)
    arrow(s, 9.68, cy, 9.92, cy)
tbox(s, L, 5.9, CW, 0.8, [[("另外：", {"bold": True}),
                           ("被關掉的基因本身固定下降約 85%（比賽保證 > 80%）；基因組上 50 kb 內的鄰近基因（CRISPRi 的旁及效應）由共識模型另外處理。", {})]],
     size=12.5, color=MUTED, spacing=1.25)

# ---------------------------------------------------------------------------------------------------------------
# 04 實驗結果
divider(3, "第四部分是實驗結果：先說明驗證怎麼設計，再依序看模型比較、消融實驗、參數選擇、分項分數，最後是瓶頸分析和沒有採用的方法。")

s = content(3, "驗證設計：三種「藏起答案」的考法，避免只對期末考調參數",
            "期末考的成績要比賽結束才公布，所以我們自己設計三種考法。第一種把整種細胞藏起來，七組細胞輪流當新細胞；第二種把一批基因的資料全部拿掉，"
            "測試沒測過的基因；第三種是模擬考：拿答案已知的細胞當期末考，跑完整個流程，用官方的評分程式打分。模擬考有三套，H1 的讀數深度最接近期末考，所以最可靠；"
            "另外兩套資料比較淺，換算分數不穩定，所以看原始分數。採用規則是：改動要在 H1 多數組別進步，其他兩套也沒有明顯退步，才放進期末版本。")
for k, (h, d, m) in enumerate([("藏起整種細胞", "7 組細胞實驗輪流藏起，用其餘資料訓練後預測", "沒看過的細胞準不準"),
                               ("藏起一批基因", "5 組基因的資料全部移除，再預測它們", "沒測過的基因準不準"),
                               ("官方評分模擬考", "答案已知的細胞當作期末考，跑完整流程並用官方程式評分", "真正的分數")]):
    x = COL3[k]
    shp(s, x, 1.95, 3.9, 2.45, PANEL, kind=ROUND, radius=0.12, name=f"Protocol {k + 1}")
    num_circle(s, x + 0.25, 2.13, k + 1, d=0.5)
    tbox(s, x + 0.88, 2.12, 2.9, 0.5, h, size=16.5, bold=True, color=NAVY, anchor=MID)
    tbox(s, x + 0.25, 2.8, 3.4, 0.95, d, size=13.5, color=TEXT, spacing=1.25)
    tbox(s, x + 0.25, 3.8, 3.4, 0.45, [[("衡量　", {"bold": True}), (m, {})]], size=12.5, color=TEAL_TX)
table(s, L, 4.65, [1.25, 1.75, 1.95, 1.35, 1.8],
      [["模擬考", "細胞", "規模", "讀數／細胞", "說明"], ["H1", "胚胎幹細胞", "5 組 × 約 60 基因", "約 24,000", "最接近期末考"],
       ["HCT116", "大腸癌細胞", "4 組 × 約 66 基因", "約 12,000", "看原始分數"],
       ["KOLF200", "誘導多能幹細胞", "4 組 × 約 70 基因", "約 3,900", "看原始分數"]], row_h=0.44, size=12.5,
      bold_first=True, name="Mock sets")
tbox(s, L, 6.45, 8.1, 0.35, "讀數深度：期末考 D、E、F 約 20,000／細胞", size=11.5, color=MUTED)
shp(s, 9.0, 4.65, 3.733, 1.76, TEAL_BG, kind=ROUND, radius=0.1, margin=(0.22, 0.1), name="Adoption rule",
    paras=[{"runs": [("採用規則", {"bold": True, "color": TEAL_TX, "size": 14.5})], "after": 4},
           "改動要在 H1 多數組別進步，而且 HCT116、KOLF200 沒有明顯退步，才放進期末版本。"], size=13.5, spacing=1.25)

s = content(3, "結果一：AdaptNet 在同樣資料下高 18%，加入 RPE1 後合計高 37%",
            "第一個結果是模型比較。灰色是原本的方法，把各來源直接平均；藍色是 AdaptNet；深藍色是再加入 RPE1 資料。"
            "左圖是方向相似度，也就是預測的變化方向和真實的有多像。用同樣的資料，AdaptNet 從 0.111 提升到 0.131，進步 18%；"
            "再加入 RPE1 資料升到 0.152，合計進步 37%，模型和新資料大約各貢獻一半。右圖是區分度，0.5 是亂猜，從 0.738 提升到 0.766。"
            "這是七組細胞輪流藏起來的平均，所以代表模型對沒看過的細胞確實比較準。")
pic(s, "fig_benchmark.png", L, 1.95, 8.7, 3.9, "藏起細胞類型時，原方法、AdaptNet、AdaptNet 加 RPE1 的方向相似度與區分度")
tbox(s, 9.65, 2.0, 3.08, 0.9, "+18%", size=44, bold=True, color=TEAL_TX, anchor=MID)
tbox(s, 9.65, 2.9, 3.08, 0.75, ["同樣資料：AdaptNet", "方向相似度 0.111 → 0.131"], size=13.5, color=TEXT, spacing=1.2)
tbox(s, 9.65, 3.85, 3.08, 0.9, "+37%", size=44, bold=True, color=NAVY, anchor=MID)
tbox(s, 9.65, 4.75, 3.08, 0.75, ["再加入 RPE1 資料", "方向相似度 → 0.152"], size=13.5, color=TEXT, spacing=1.2)
tbox(s, L, 5.95, CW, 0.8, [[("設定：", {"bold": True}),
                            ("7 折交叉驗證，每次藏起一組細胞實驗；方向相似度＝扣除雜訊後的餘弦相似度（1 = 完全一致）；區分度 0.5 = 亂猜，0.738 → 0.762 → 0.766。RPE1 不涵蓋期末考的基因，對期末考影響很小。", {})]],
     size=12, color=MUTED, spacing=1.25)

s = content(3, "消融實驗：改用訓練的模型預測總量貢獻最大，模擬考總分 0.272 → 0.306",
            "這是消融實驗，把每個設計一步一步加上去，看官方模擬考的總分怎麼變。最左邊是原方法 0.272；改用 AdaptNet 預測總量貢獻最大，加了 0.019；"
            "之後加入 RPE1 資料、調整對照組方向的處理、加上目標權重、三個模型取平均，各自再加一點，最後是 0.306。三個模型取平均在五組裡有四組進步，代表結果比較穩定。")
pic(s, "fig_ablation.png", L, 1.95, 9.0, 3.95, "消融實驗瀑布圖：H1 模擬考總分從 0.272 逐步提升到 0.306")
for k, (h, d) in enumerate([("最大增益", "改用 AdaptNet 預測總量：+0.019"), ("更穩定", "3 個模型取平均：+0.004，5 組中 4 組進步"),
                            ("評分方式", "官方評分程式，H1 模擬考 5 組、共 298 個基因")]):
    y = 2.15 + k * 1.2
    tbox(s, 9.9, y, 2.833, 0.38, h, size=14, bold=True, color=NAVY)
    tbox(s, 9.9, y + 0.38, 2.833, 0.75, d, size=12.5, color=TEXT, spacing=1.2)
tbox(s, L, 6.05, CW, 0.65, "總量的設定（對照組方向、目標權重）另在 HCT116、KOLF200 上交叉確認。", size=12, color=MUTED)

s = content(3, "參數選擇：只移除 5 個雜訊方向，區分度明顯提升、誤差小幅增加",
            "這頁解釋 v5 的一個設定：要從預測裡移除幾個『對照組的雜訊方向』。橫軸是移除的數量，左圖是區分度、越高越好，右圖是整體誤差、越低越好。"
            "從 0 個到 5 個，在讀數比較淺的 HCT116 和 KOLF200 上，區分度大幅提升，H1 的誤差只增加一點點；再多移除，區分度幾乎不再提升，誤差卻一直上升。"
            "所以 v5 選 5 個。這是三套模擬考一起看的結果，不是只看 H1。")
_, py0, _, ph0 = pic(s, "fig_ctrl_axes.png", L, 1.9, CW, 4.1, "移除 0 到 50 個對照組方向時，三套模擬考的區分度和整體誤差")
tbox(s, L, py0 + ph0 + 0.1, CW, 0.7, [[("0 → 5 個方向：", {"bold": True}),
                                       ("HCT116 區分度 0.705 → 0.747、KOLF200 0.598 → 0.656，H1 誤差只增加 0.008；再多移除幾乎沒有增益，誤差卻持續上升（快速試算，官方模擬考結果一致）。", {})]],
     size=12.5, color=TEXT, spacing=1.25)

s = content(3, "分項分數：區分度 0.87 接近實驗重複水準，顯著基因類仍接近基準",
            "這是 v5 在 H1 模擬考的六項分數。區分度 0.87，表示模型很會分辨不同基因被關掉的結果，已經接近實驗重做一次的水準；方向正確 0.55 也不錯。"
            "但方向排序只有 0.075、變化大小接近 0，表示『哪些基因顯著改變、往哪個方向、變多少』這件事，很難從別的細胞轉移過來。下一頁分析原因。")
pic(s, "fig_metrics_v5.png", L, 1.95, 7.3, 4.05, "v5 在 H1 模擬考的六項換算分數")
shp(s, 8.25, 2.0, 4.483, 1.85, TEAL_BG, kind=ROUND, radius=0.1, margin=(0.22, 0.12), name="Strength",
    paras=[{"runs": [("強項：總量", {"bold": True, "color": TEAL_TX, "size": 15})], "after": 4},
           "區分度 0.871：模型能清楚分辨不同基因被關掉的結果；方向正確 0.548。"], size=13.5, spacing=1.25)
shp(s, 8.25, 4.05, 4.483, 1.95, ORANGE_BG, kind=ROUND, radius=0.1, margin=(0.22, 0.12), name="Weakness",
    paras=[{"runs": [("弱項：顯著基因", {"bold": True, "color": ORANGE_TX, "size": 15})], "after": 4},
           "方向排序 0.075、變化大小 −0.022：哪些基因顯著、往哪個方向、變多少，很難從別的細胞轉移。"], size=13.5, spacing=1.25)
tbox(s, L, 6.15, CW, 0.55, "v5 在 H1 模擬考（官方評分，5 組平均）的六項換算分數，總分 0.306；0 = 平均基準，1 = 實驗重複水準。",
     size=12, color=MUTED)

s = content(3, "瓶頸在「方向」：非必需基因的變化方向幾乎無法跨細胞轉移",
            "為什麼顯著基因的分數低？左圖看每組藏起的細胞裡，模型最有把握的 10 個基因，方向猜對的比例。必需基因的實驗（HepG2、Jurkat）有七成以上，"
            "但以非必需基因為主的實驗只有五成多，接近丟銅板，而期末考多是非必需基因。右圖是假設：如果方向全部猜對、其他都不變，方向正確會從 0.55 變成 1.17，"
            "方向排序從 0.04 變成 0.60。所以目前的上限在『方向』，不是在產生細胞的方式。")
tbox(s, L, 1.85, 6.2, 0.4, "(a) 最有把握的 10 個基因：方向猜對率（細胞層用的預測）", size=13.5, bold=True, color=NAVY)
tbox(s, 7.0, 1.85, 5.733, 0.4, "(b) 如果方向全對：分數上限（H1 模擬考）", size=13.5, bold=True, color=NAVY)
pic(s, "fig_sign_accuracy.png", L, 2.3, 6.2, 2.9, "藏起各組細胞時，最有把握的 10 個基因方向猜對率")
pic(s, "fig_true_signs.png", 7.0, 2.3, 5.733, 2.9, "實際預測與假設方向全對時的方向正確和方向排序分數")
shp(s, L, 5.42, CW, 1.33, PANEL, kind=ROUND, radius=0.1, margin=(0.3, 0.08), name="Interpretation",
    paras=[[("解讀　", {"bold": True, "color": NAVY}),
            ("必需基因被關掉時反應強而一致，方向能轉移（72–74%）；以非必需基因為主的實驗只有 54–56%，接近丟銅板，而期末考多為非必需基因。"
             "目前的上限在「方向」，不在產生細胞的方式。", {})]], size=14, spacing=1.3)

s = content(3, "試過但未採用的方法：都經過模擬考驗證，沒有穩定增益",
            "這頁列出我們試過、但最後沒有放進期末版本的方法。每一個都有合理的動機，例如加入新細胞的共同反應：如果事先知道答案可以加 0.03 分，"
            "但我們沒辦法從其他細胞預測它。其他像細胞週期、專門學方向的損失函數、對照組的自然變異、加權訓練、更多的資料等等，都在模擬考上沒有穩定的進步。"
            "這些負面結果也很重要，它們說明瓶頸不在這些地方。")
table(s, L, 1.85, [3.35, 3.55, 5.233],
      [["想法", "為什麼可能有用", "結果"],
       ["加入新細胞的「共同反應」", "69% 的顯著基因和共同反應同方向", "事先知道答案可 +0.03，但無法從其他細胞預測（相關 0.27）"],
       ["用細胞週期組成解釋反應", "擾動可能改變細胞週期的比例", "幾乎解釋不了（平均相關 0.04）"],
       ["專門學「方向」的損失函數", "瓶頸在方向", "和原本相同：模型只能重新組合來源"],
       ["對照組細胞的自然差異", "基因自然的高低起伏可能類似擊倒", "區分度 0.47＝亂猜（目標基因表現太低、太稀疏）"],
       ["依表現量加權訓練", "評分比較重視高表現的基因", "差異在隨機誤差範圍內"],
       ["預測每個擾動影響幾個基因", "真實顯著基因數差異很大", "少報會被重扣分，結果反而變差"],
       ["更多資料：ELEM-1200RS、LINCS、神經細胞", "資料越多應該越好", "沒有幫助或變差（實驗類型不同）"],
       ["更真實的「下降基因」產生方式", "下降可能只發生在部分細胞", "三套模擬考結果不一致，保留為選項"]],
      row_h=0.5, size=12.5, bold_first=True, name="Negative results")
tbox(s, L, 6.45, CW, 0.35, "詳細數字見 SUMMARY.md；每個比較都使用同一組模擬考或交叉驗證。", size=11.5, color=MUTED)

# ---------------------------------------------------------------------------------------------------------------
# 05 結論與未來工作
divider(4, "最後一部分是結論與未來工作。")

s = content(4, "結論：小型可轉移模型能穩定預測「總量」，方向轉移是下一個關鍵",
            "總結三個貢獻。第一，我們做出一個通用模型，任何細胞類型、任何基因都能預測，而且只有兩萬個參數、一分鐘就能訓練。"
            "第二，我們建立了嚴格的驗證：在藏起的細胞類型上方向相似度提升 37%，其中模型和新增的 RPE1 資料大約各占一半；官方評分模擬考從 0.272 提升到 0.306。"
            "第三，期末版本 v5 已經在 10 月 24 日上傳並通過格式檢查。主要的發現是：區分度已經接近實驗重複水準，剩下的差距主要來自非必需基因的變化方向很難跨細胞轉移。")
for k, (big, h, d) in enumerate([("2 萬", "通用模型", "AdaptNet＋GeneNet4：任何細胞類型、任何基因都能預測；參數少，約 1 分鐘訓練完成"),
                                 ("+37%", "嚴格驗證", "藏起細胞類型的方向相似度提升 37%（模型與新增資料約各占一半）；模擬考總分 0.272 → 0.306"),
                                 ("v5", "期末提交", "D、E、F 期末版本於 10/24 上傳並通過格式檢查；成績於賽後公布")]):
    x = COL3[k]
    shp(s, x, 1.95, 3.9, 3.2, PANEL, kind=ROUND, radius=0.12, name=f"Contribution {k + 1}")
    tbox(s, x + 0.28, 2.1, 3.4, 0.9, big, size=40, bold=True, color=TEAL_TX, anchor=MID)
    tbox(s, x + 0.28, 3.0, 3.4, 0.45, h, size=17, bold=True, color=NAVY)
    tbox(s, x + 0.28, 3.5, 3.4, 1.6, d, size=15, color=TEXT, spacing=1.3)
shp(s, L, 5.4, CW, 1.3, WHITE, line=NAVY, lw=1.5, kind=ROUND, radius=0.1, margin=(0.3, 0.08), name="Main finding",
    paras=[[("主要發現　", {"bold": True, "color": NAVY}),
            ("區分度的換算分數已達 0.87（1 ＝ 實驗重複水準）；剩下的差距主要來自「非必需基因的變化方向」難以跨細胞轉移，而不是模型大小或產生細胞的方式。", {})]],
    size=14.5, spacing=1.3)

s = content(4, "未來工作：補上相近細胞的資料，並讓模型知道自己「沒把握」的地方",
            "未來有三個方向。第一，取得和 D、E、F 相近的細胞類型的擾動資料，例如骨髓細胞、纖維母細胞、上皮細胞，這最能直接改善方向的轉移。"
            "第二，讓模型估計自己的把握程度，只在有把握的基因宣告顯著，但要配合評分規則校準，因為少報也會被扣分。"
            "第三，想辦法預測新細胞的共同反應，模擬考顯示上限大約 0.03 分。右邊是時程：期末截止是台灣時間 11 月 6 日早上 7 點 59 分，只算最後一次上傳。")
for k, (h, d) in enumerate([("取得相近細胞的擾動資料", "D、E、F 類似骨髓、間質、上皮細胞；這些細胞類型的公開 CRISPRi 資料，最能直接改善方向轉移。"),
                            ("讓模型估計自己的把握", "為每個基因給出信心分數，只在有把握時宣告顯著；需要配合評分規則校準（少報也會被扣分）。"),
                            ("預測新細胞的共同反應", "模擬考上限約 +0.03；目前從其他細胞推測的相關只有 0.27，需要新的資訊來源。")]):
    y = 1.95 + k * 1.55
    num_circle(s, L, y + 0.04, k + 1, d=0.55, fill_c=TEAL)
    tbox(s, 1.35, y, 6.9, 0.45, h, size=16.5, bold=True, color=NAVY)
    tbox(s, 1.35, y + 0.47, 6.9, 0.95, d, size=13.5, color=TEXT, spacing=1.25)
tbox(s, 8.9, 1.9, 3.8, 0.4, "時程", size=15, bold=True, color=NAVY)
arrow(s, 9.15, 2.55, 9.15, 6.25, color=LINE, lw=2.5, head=False, name="Timeline")
for k, (d_, c_, t_) in enumerate([("10/24", NAVY, "v5 上傳（目前計分的版本）"), ("11/06 07:59", ORANGE_TX, ["期末截止（台灣時間）", "只算最後一次上傳"]),
                                  ("賽後", NAVY, "公布成績與得獎名單")]):
    y = 2.45 + k * 1.75
    shp(s, 9.0, y, 0.3, 0.3, c_, kind=OVAL)
    tbox(s, 9.5, y - 0.08, 3.23, 0.45, d_, size=15, bold=True, color=c_)
    tbox(s, 9.5, y + 0.38, 3.23, 0.9, t_, size=12.5, color=TEXT, spacing=1.2)

s = content(None, "參考文獻", "這頁列出報告中提到的方法和使用的公開資料。")
refs = ["Replogle et al., Cell (2022)：全基因組 Perturb-seq（K562、RPE1）", "Roohani et al., Nat. Biotechnol. (2024)：GEARS",
        "Lotfollahi et al., Mol. Syst. Biol. (2023)：CPA", "Cui et al., Nat. Methods (2024)：scGPT", "Adduri et al., bioRxiv (2025)：State",
        "Yu et al., ICLR (2026)：scDFM", "Ahlmann-Eltze et al., Nat. Methods (2025)：深度模型與線性基準的比較",
        "Nadig et al., Nat. Genet. (2025)：HepG2、Jurkat 必需基因 Perturb-seq",
        "Xaira Therapeutics (2025)：X-Atlas/Orion（HCT116、HEK293T）", "Marson lab：人類 CD4 T 細胞全基因組 CRISPRi Perturb-seq",
        "Feng et al. (2025)：34 位捐贈者 iPSC 的 CRISPRi 實驗", "KOLF2.1J 全基因組 CRISPRi，Nat. Biotechnol. (2026)",
        "Satija lab (2026)：VIPerturb-seq（K562，10x Flex）", "Szklarczyk et al., Nucleic Acids Res. (2023)：STRING v12",
        "Gene Ontology Consortium, Genetics (2023)：GO 知識庫", "Arc Institute：Virtual Cell Challenge 2026 與 cell-eval2 評分程式"]
for col in range(2):
    tbox(s, L + col * 6.18, 1.85, 5.95, 4.9,
         [{"runs": [(f"[{i + 1}]\t", {"bold": True, "color": NAVY}), (refs[i], {})], "after": 10, "hang": 0.5} for i in range(col * 8, col * 8 + 8)],
         size=13, color=TEXT, spacing=1.2)

s = prs.slides.add_slide(LAYOUT); bg(s, DARK)
title(s, "謝謝聆聽　敬請指教", x=0.9, y=2.45, w=11.5, h=1.0, size=40, color=WHITE, align=CENTER)
tbox(s, 0.9, 3.5, 11.5, 0.6, "Q & A", size=26, bold=True, color=TEAL_LT, align=CENTER)
tbox(s, 0.9, 5.4, 11.5, 0.45, "程式與實驗紀錄：github.com/scnpg/VCC_2026_NTU_JHLab", size=14, color=ON_DARK, align=CENTER)
footer(s, dark=True)
notes(s, "以上是我們的報告，謝謝各位老師，請老師們指教。後面附錄有評分細節、運算資源和提交紀錄，可以在問答時使用。")

# ---------------------------------------------------------------------------------------------------------------
# backup
s = content(None, "顯著基因類指標的計算方式", "如果老師問到評分細節，可以用這頁說明：方向正確會同時懲罰多報和少報；"
            "方向排序看真正顯著的基因有沒有被我們排在前面；整體誤差會扣掉抽樣雜訊的修正，所以變化幅度要調到剛好；空集合的規則在期末考很少發生。", tag="附錄 A")
for k, (chip, f_, d_) in enumerate([
        ("方向正確（fid）", "k ÷ max（我們宣告的顯著基因數, 真實顯著基因數）", "k = 宣告顯著、方向正確、而且真實也顯著的基因數 → 多報、少報都會被扣分"),
        ("方向排序（reach）", "依我們的 p 值，替真實顯著的基因排序", "在方向正確率維持 90% 以上的前提下，能涵蓋多少真實顯著的基因"),
        ("整體誤差：變化幅度", "評分會扣掉抽樣雜訊的修正，修正上限取決於預測的變化幅度", "幅度調到「上限＝修正量」時誤差最低；快速試算確認放大 1.25 倍或縮小 0.75 倍都變差"),
        ("空集合規則", "真實沒有顯著基因時，不宣告任何基因可得滿分", "期末考很少出現；我們對每個擾動宣告約 3,000 個顯著基因最穩健")]):
    y = 1.95 + k * 1.2
    shp(s, L, y, 2.6, 1.0, NAVY, kind=ROUND, radius=0.1, paras=[[(chip, {})]], align=CENTER, size=14, bold=True, color=WHITE)
    tbox(s, 3.45, y + 0.02, 9.28, 0.45, f_, size=14.5, bold=True, color=NAVY)
    tbox(s, 3.45, y + 0.5, 9.28, 0.5, d_, size=13, color=TEXT)

s = content(None, "運算資源與可重現性", "這頁說明我們用的運算資源和怎麼重現結果。大部分工作在實驗室的一張 RTX 4060 Ti 上完成，"
            "只有在記憶體不夠時才用雲端，總共大約 6 美元。所有程式、實驗紀錄和重建期末版本的指令都在 GitHub 上。", tag="附錄 B")
table(s, L, 1.95, [2.0, 3.9], [["項目", "內容"], ["本機 GPU", "RTX 4060 Ti 16 GB：訓練、模擬考"], ["雲端", "Brev，只在記憶體不足時使用，共約 US$6"],
                                ["模型訓練", "約 1 分鐘／模型"], ["官方評分模擬考", "約 100 次"], ["交叉驗證實驗", "60 多組"]],
      row_h=0.5, size=13, bold_first=True, name="Compute")
tbox(s, 6.85, 1.9, 5.883, 1.5, [{"runs": [("重現方式", {"bold": True, "color": NAVY, "size": 15})], "after": 6},
                                {"runs": [("程式與紀錄：github.com/scnpg/VCC_2026_NTU_JHLab（general_model/）", {})], "bullet": True, "after": 3},
                                {"runs": [("每個數字的來源：SUMMARY.md（實驗紀錄）", {})], "bullet": True, "after": 3},
                                {"runs": [("重建期末 v5（約 20 分鐘）：", {})], "bullet": True}], size=13, color=TEXT, spacing=1.2)
code = ["cd scDFM", "RPE1=1 SEEDS=0,1,2 .venv/bin/python -I \\", "  repro/model_v2_20261022/build_v2.py v4", "cd ..",
        "PRECOMP=$PWD/data/stage3/effects/gm_final_v4pre \\", "  RENDER=g1 SHAREDK=0 GAMMA=1 CTRLAXES=5 TWPOW=0.5 \\",
        "  CELLAGREE=1 CELLAMP=1.5 TOPFRAC=0.02 \\", "  bash tools/final_pipeline.sh data/vcc2026_final \\", "  gm_final_v5 D E F"]
shp(s, 6.85, 3.5, 5.883, 2.3, "EEF1F5", kind=ROUND, radius=0.08, margin=(0.2, 0.12), anchor=TOP, name="Rebuild commands",
    paras=[[(c_, {})] for c_ in code], size=11, color=TEXT, font=MONO, spacing=1.15)

s = content(None, "期末提交紀錄與公開排行榜", "這頁是期末提交的紀錄。每天最多上傳兩次、只算最後一次，成績在比賽結束才公開。"
            "目前計分的是 10 月 24 日凌晨上傳的 v5。另外在前一階段的公開排行榜，我們用舊方法拿到 0.239，在 1,635 隊中排第 165 名。", tag="附錄 C")
pic(s, "fig_mock_progress.png", L, 1.95, 6.6, 3.0, "各版本的 H1 模擬考總分")
table(s, 7.45, 1.95, [0.65, 1.4, 2.33, 0.903],
      [["版本", "上傳時間", "內容", "模擬考"], ["v1", "10/22 19:44", "各來源平均（保底版）", "0.272"], ["v2", "10/23 01:30", "改用 AdaptNet 預測總量", "0.291"],
       ["v4", "10/23 18:40", "＋RPE1、設定調整、3 模型平均", "0.306"],
       [[[("v5", {"bold": True})]], [[("10/24 00:24", {"bold": True})]], [[("＋只移除 5 個雜訊方向（目前計分）", {"bold": True})]],
        [[("0.306", {"bold": True})]]]],
      row_h=0.55, size=12.5, name="Submissions", align_cols={3: RIGHT}, cell_fill={(4, j): TEAL_BG for j in range(4)})
tbox(s, 7.45, 4.8, 5.283, 0.35, "上傳時間為台灣時間", size=11, color=MUTED)
shp(s, L, 5.3, CW, 1.4, PANEL, kind=ROUND, radius=0.1, margin=(0.3, 0.1), name="Rules",
    paras=[{"runs": [("規則　", {"bold": True, "color": NAVY}), ("每天最多上傳 2 次、只算最後一次；分數在比賽結束、公布得獎者時才公開。", {})], "after": 6},
           [("A/B/C 驗證期（公開排行榜）　", {"bold": True, "color": NAVY}), ("舊方法最佳 0.239，第 165 名／1,635 隊；前 40 名約 0.30–0.35。", {})]],
    size=14, spacing=1.25)

prs.save(OUT)
print(f"{len(prs.slides)} slides -> {OUT}")
