"""PowerPoint version of the professor progress deck (2026-10-24), same content as the Slides artifact
1oFtaxuSjLczEm8McQjDkm (deck/project/slides/*.html), for presenting without the link.

    <python with python-pptx> make_pptx.py [out.pptx]
"""
import sys
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu

OUT = sys.argv[1] if len(sys.argv) > 1 else "VCC_progress_report_20261024.pptx"
FONT = "Microsoft JhengHei"
NAVY, NAVY2, WHITE = "16213A", "22304F", "FFFFFF"
TEXT, MUTED, BLUE, ORANGE, ORANGE_L = "3B4658", "5E6878", "2563A8", "C8691E", "E9A35C"
TINT, TINT2, LIGHT, LIGHT2, ICE = "F1F4F8", "DCE6F2", "F6F5F1", "C9D3E0", "8CB8E8"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
TITLE_ONLY = prs.slide_layouts[5]
W = 12.133                                   # content width between 0.6" margins


def rgb(h):
    return RGBColor.from_string(h)


def style_run(run, size, color, bold=False, italic=False):
    f = run.font; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = rgb(color); f.name = FONT
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):                                   # East Asian glyphs use the same face
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {}); rPr.append(el)
        el.set("typeface", FONT)


def text(slide, x, y, w, h, paras, size=16, color=TEXT, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.25, after=0, name=None):
    """paras: list of paragraphs; a paragraph is a str or a list of (text, {size, color, bold}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: tb.name = name
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = spacing; p.space_after = Pt(after)
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            r = p.add_run(); r.text = t
            style_run(r, o.get("size", size), o.get("color", color), o.get("bold", bold))
    return tb


def box(slide, x, y, w, h, fill, line=None, radius=0.08, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_w=1.5, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name: s.name = name
    if fill: s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    else: s.fill.background()
    if line: s.line.color.rgb = rgb(line); s.line.width = Pt(line_w)
    else: s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def arrow(slide, x, y, w=0.55, h=0.38, color=BLUE):
    return box(slide, x, y, w, h, color, shape=MSO_SHAPE.RIGHT_ARROW)


def new_slide(title, dark=False, n=None, foot=None, title_size=30):
    s = prs.slides.add_slide(TITLE_ONLY)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = rgb(NAVY if dark else WHITE)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.42), Inches(W), Inches(0.85)
    tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.05)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = title; style_run(r, title_size, LIGHT if dark else NAVY, bold=True)
    if n is not None:
        label = f"VCC 2026 進度報告　·　{n}" + (f"　·　{foot}" if foot else "")
        text(s, 0.6, 6.95, W, 0.35, [label], size=11, color="9FB0C6" if dark else MUTED, name="Footer")
    return s


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def card(slide, x, y, w, h, head, body, head_color=BLUE, fill=TINT, head_size=19, body_size=15, body_color=TEXT, num=None):
    box(slide, x, y, w, h, fill)
    top = y + 0.25
    if num is not None:
        c = box(slide, x + 0.3, top, 0.55, 0.55, head_color, shape=MSO_SHAPE.OVAL)
        tf = c.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = str(num); style_run(r, 18, WHITE, True)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        top += 0.7
    text(slide, x + 0.25, top, w - 0.5, 0.5, [head], size=head_size, color=head_color, bold=True)
    text(slide, x + 0.25, top + 0.55, w - 0.5, h - (top - y) - 0.7, [body], size=body_size, color=body_color, spacing=1.3)


def table(slide, x, y, w, col_w, rows, size=15, row_h=0.5, highlight_last=False, bold_last=False, right_cols=()):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = shp.table
    for j, cw in enumerate(col_w): tbl.columns[j].width = Inches(cw)
    for i in range(nr):
        tbl.rows[i].height = Inches(row_h)
        for j in range(nc):
            cell = tbl.cell(i, j); cell.text = ""
            cell.margin_left = cell.margin_right = Inches(0.12); cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT if j in right_cols else PP_ALIGN.LEFT
            r = p.add_run(); r.text = rows[i][j]
            head = i == 0; last = i == nr - 1
            style_run(r, size, WHITE if head else NAVY, bold=head or (bold_last and last))
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(NAVY if head else (TINT2 if (highlight_last and last) else (TINT if i % 2 == 0 else WHITE)))
    return shp


def stat(slide, x, y, w, h, number, caption, label=None, fill=NAVY, num_color=ORANGE_L, cap_color=LIGHT, num_size=44, cap_size=16):
    box(slide, x, y, w, h, fill)
    top = y + 0.25
    if label:
        text(slide, x + 0.3, top, w - 0.6, 0.45, [label], size=16, color=cap_color); top += 0.5
    text(slide, x + 0.3, top, w - 0.6, 0.95, [number], size=num_size, color=num_color, bold=True, spacing=1.0)
    text(slide, x + 0.3, top + num_size / 72 * 1.25 + 0.05, w - 0.6, 0.9, [caption], size=cap_size, color=cap_color, spacing=1.3)


# 1 cover -----------------------------------------------------------------------------------------------
s = prs.slides.add_slide(TITLE_ONLY)
s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(NAVY)
text(s, 0.9, 1.45, 11.5, 0.5, ["Arc Virtual Cell Challenge 2026｜進度報告"], size=18, color=ORANGE_L, bold=True)
t = s.shapes.title; t.left, t.top, t.width, t.height = Inches(0.9), Inches(2.05), Inches(11.6), Inches(1.9)
tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
for i, line in enumerate(["虛擬細胞挑戰賽：", "訓練一個通用的擾動預測模型"]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.LEFT; p.line_spacing = 1.1
    r = p.add_run(); r.text = line; style_run(r, 42, LIGHT, True)
text(s, 0.9, 4.25, 11.3, 1.1, ["目標：給一種「沒看過的細胞」和一個要關掉的基因，預測關掉後這種細胞的基因表現會怎麼變"], size=21, color=LIGHT2, spacing=1.35)
text(s, 0.9, 5.8, 11.3, 0.5, ["2026/10/24　　報告人：[姓名]"], size=16, color=LIGHT2)
notes(s, "開場：這是我們參加 Arc 研究所「虛擬細胞挑戰賽 2026」的進度報告。一句話說明：我們訓練了一個模型，給它一種新的細胞（只有它沒被處理過的樣子）和一個要關掉的基因，它就要預測關掉之後，這種細胞裡一萬八千多個基因的表現會怎麼改變。期末考是三種模型從沒看過的細胞，截止日是 11 月 5 日。")

# 2 summary ---------------------------------------------------------------------------------------------
s = new_slide("一頁重點", n=2)
cw = (W - 0.3) / 2
items = [("做了什麼", "訓練一個通用模型（AdaptNet），用 15 個公開 CRISPRi 實驗、約 3.8 萬組「細胞 × 被關掉的基因」資料學習", BLUE),
         ("怎麼驗證", "訓練時藏起整種細胞或一批基因再預測；並用官方評分程式，在三種細胞上做「模擬考」", BLUE),
         ("目前成績", "H1 模擬考從原方法 0.272 進步到 0.306；期末版本 v5 已於 10/24 上傳並評分成功（官方分數比賽結束才公布）", BLUE),
         ("主要限制", "「哪些基因會變、往哪個方向變」受資料限制：對非必需基因的方向只猜對約 55%", ORANGE)]
for k, (h, b, c) in enumerate(items):
    card(s, 0.6 + (k % 2) * (cw + 0.3), 1.55 + (k // 2) * 2.65, cw, 2.4, h, b, head_color=c, head_size=21, body_size=17)
notes(s, "先給教授一頁總結。第一，我們訓練了一個通用模型，叫 AdaptNet，資料來自 15 個公開的 CRISPRi 基因關閉實驗，總共大約三萬八千組「在某種細胞裡關掉某個基因」的測量結果。第二，驗證方式是故意把整種細胞或一批基因藏起來，看模型能不能預測；另外用比賽官方的評分程式，在三種細胞上做模擬考。第三，模擬考從原本方法的 0.272 進步到 0.306，期末版本已經上傳並評分成功，官方分數要等比賽結束才公布。第四，主要限制是：預測「哪些基因會變、往哪個方向變」的準確度，被現有公開資料限制住了。")

# 3 task ------------------------------------------------------------------------------------------------
s = new_slide("比賽任務：預測「關掉一個基因」後細胞的樣子", n=3)
for k, (h, b) in enumerate([("輸入 1：新細胞的對照組", "約 18,400 顆沒被處理過的細胞"), ("輸入 2：要關掉的基因", "用 CRISPRi 降低它的表現（每種細胞 300 個）")]):
    y = 1.6 + k * 1.25
    box(s, 0.6, y, 3.75, 1.1, TINT)
    text(s, 0.8, y + 0.12, 3.4, 0.4, [h], size=17, color=NAVY, bold=True)
    text(s, 0.8, y + 0.52, 3.4, 0.55, [b], size=14, color=TEXT)
arrow(s, 4.5, 2.35)
box(s, 5.2, 1.85, 2.7, 1.6, NAVY)
text(s, 5.35, 2.05, 2.4, 0.5, ["我們的模型"], size=22, color=LIGHT, bold=True, align=PP_ALIGN.CENTER)
text(s, 5.35, 2.65, 2.4, 0.5, ["AdaptNet"], size=16, color=LIGHT2, align=PP_ALIGN.CENTER)
arrow(s, 8.05, 2.35)
box(s, 8.75, 1.6, 3.98, 2.35, WHITE, line=BLUE, line_w=2)
text(s, 8.95, 1.8, 3.6, 0.45, ["輸出：預測的細胞"], size=19, color=BLUE, bold=True)
text(s, 8.95, 2.35, 3.6, 1.4, ["每個被關掉的基因，產生 400 顆細胞、每顆 18,533 個基因的讀數"], size=15, color=TEXT, spacing=1.3)
sw = (W - 0.6) / 3
for k, (num, cap) in enumerate([("3 種新細胞", "期末考 D、E、F，模型從沒看過"), ("36 萬顆細胞", "3 × 300 個基因 × 400 顆"), ("11/05 截止", "只算最後一次上傳，分數最後才公布")]):
    x = 0.6 + k * (sw + 0.3)
    box(s, x, 4.45, sw, 1.95, TINT)
    text(s, x + 0.3, 4.65, sw - 0.6, 0.7, [num], size=30, color=NAVY, bold=True)
    text(s, x + 0.3, 5.45, sw - 0.6, 0.8, [cap], size=15, color=TEXT)
notes(s, "比賽的任務是這樣：主辦單位給我們一種新的細胞，但只給「沒被處理過」的對照組細胞，大約一萬八千四百顆。再告訴我們要關掉哪 300 個基因。關掉的方法叫 CRISPRi，可以把某個基因的表現大幅降低。我們要預測：每關掉一個基因，這種細胞會變成什麼樣子，而且要產生 400 顆細胞、每顆細胞 18,533 個基因的讀數。期末考有 D、E、F 三種細胞，總共 36 萬顆細胞，11 月 5 日截止，只算最後一次上傳，分數要比賽結束才公布。重點是：這三種細胞我們完全沒有它們被擾動過的資料，所以是「零樣本」預測。")

# 4 scoring ---------------------------------------------------------------------------------------------
s = new_slide("評分方式：六項分數取平均", n=4, foot="數字來自 H1 模擬考（官方評分程式）")
text(s, 0.6, 1.4, W, 0.5, [[("每項都換算成：", {}), ("0 分", {"bold": True, "color": NAVY}), (" = 每個擾動都猜「平均反應」的基準；", {}),
                           ("1 分", {"bold": True, "color": NAVY}), (" = 同一個實驗重做一次的水準", {})]], size=16)
table(s, 0.6, 2.05, W, [3.0, 7.0, 2.133], [
    ["項目", "白話意思", "我們（v5）"],
    ["區分度 pds", "能不能分辨「關掉 A」和「關掉 B」的不同", "0.87"],
    ["整體誤差 mse", "每個基因平均表現量猜得準不準", "0.17"],
    ["變化大小 nmae", "真正有變的基因，變化幅度猜得準不準", "−0.02"],
    ["方向正確 fid", "我們說「有顯著改變」的基因，升降方向對不對", "0.55"],
    ["方向排序 reach", "最有把握的那些基因，方向是否連續都對", "0.08"],
    ["清單重疊 jaccard", "我們的顯著基因清單和真實清單重疊多少", "0.20"],
    ["總分", "六項平均", "0.306"]], size=16, row_h=0.56, highlight_last=True, bold_last=True, right_cols=(2,))
notes(s, "評分有六項，每一項都換算成 0 到 1：0 分代表「不管關掉哪個基因，都猜成平均反應」這種最簡單的基準；1 分代表「同一個實驗重做一次」能達到的一致程度。六項平均就是總分。表格右邊是我們目前版本在 H1 模擬考的分數。可以看到：區分度很高（0.87），表示模型很會分辨不同擾動；整體誤差 0.17；但是跟「哪些基因有顯著改變、方向對不對」有關的幾項比較低，這是後面要說明的主要限制。")

# 5 data ------------------------------------------------------------------------------------------------
s = new_slide("訓練資料：15 個公開的基因關閉實驗", n=5, foot="試過但拿掉：ELEM-1200RS、LINCS、神經細胞資料（驗證後沒幫助或變差）")
table(s, 0.6, 1.5, 7.7, [3.9, 2.1, 1.7], [
    ["細胞類型", "實驗", "擾動數"],
    ["K562 慢性骨髓性白血病", "3 組", "9,838"],
    ["HCT116 大腸癌、HEK293T 腎", "X-Atlas", "3,387"],
    ["H1 胚胎幹細胞", "VCC 2025", "300"],
    ["CD4 T 細胞", "3 種活化狀態", "2,022"],
    ["HepG2 肝癌", "必需基因", "2,160"],
    ["Jurkat T 細胞白血病", "必需基因", "2,270"],
    ["iPSC 誘導多能幹細胞", "3 組", "16,122"],
    ["RPE1 視網膜上皮", "必需基因", "2,204"]], size=15, row_h=0.56, right_cols=(2,))
stat(s, 8.6, 1.5, 4.133, 2.45, "38,303", "組「某細胞中關掉某基因」的測量，涵蓋 9 類細胞")
card(s, 8.6, 4.2, 4.133, 1.95, "基因知識庫", "STRING 蛋白交互網路、Gene Ontology 功能註解", head_color=NAVY, head_size=18, body_size=15)
notes(s, "訓練資料全部來自公開的 CRISPRi 實驗，一共 15 組、9 類細胞，加起來有 38,303 組「在某種細胞裡關掉某個基因」的測量。每一組都是把被關掉那個基因的細胞，跟對照組比較，得到所有基因的變化量。另外加了兩個基因知識庫：STRING 是蛋白質之間的交互作用網路，Gene Ontology 是基因功能的註解，用來處理從沒被實驗測過的基因。我們也試過其他資料，例如 ELEM-1200RS、LINCS、神經細胞的資料，但驗證後沒有幫助或讓結果變差，所以拿掉了。")

# 6 model -----------------------------------------------------------------------------------------------
s = new_slide("模型架構：AdaptNet（自己訓練的神經網路）", n=6, foot="期末版本：3 個不同亂數種子訓練的模型取平均")
steps = [("輸入", "同一個基因在其他細胞被關掉的測量", TINT, NAVY, TEXT, None),
         ("① 來源加權", "依細胞像不像、資料品質、來源是否一致", WHITE, BLUE, TEXT, BLUE),
         ("② 基因程式", "拆成 200 組會一起變動的基因加細節", WHITE, BLUE, TEXT, BLUE),
         ("③ 依新細胞調整", "看新細胞對照組狀態，調整各程式強弱", WHITE, BLUE, TEXT, BLUE),
         ("輸出", "新細胞中每個基因的變化量", NAVY, LIGHT, LIGHT2, None)]
bw = (W - 4 * 0.34 - 8 * 0.06) / 5
x = 0.6
for k, (h, b, fill, hc, bc, line) in enumerate(steps):
    box(s, x, 1.55, bw, 2.35, fill, line=line, line_w=1.75)
    text(s, x + 0.15, 1.72, bw - 0.3, 0.5, [h], size=15, color=hc, bold=True)
    text(s, x + 0.15, 2.25, bw - 0.3, 1.55, [b], size=14, color=bc, spacing=1.3)
    x += bw + 0.06
    if k < 4:
        arrow(s, x, 2.56, w=0.34, h=0.28); x += 0.34 + 0.06
cw3 = (W - 0.6) / 3
for k, (h, b) in enumerate([("怎麼訓練", "每一步隨機藏起一種細胞，要求模型只用其他細胞的資料預測它，等於一直在練習「新細胞」"),
                            ("學什麼", "預測誤差越小越好，同時讓不同擾動的預測彼此可以分辨（對比學習）"),
                            ("為什麼小", "約 2 萬個參數；大模型只會背下 15 種細胞，換新細胞就失準")]):
    card(s, 0.6 + k * (cw3 + 0.3), 4.2, cw3, 2.5, h, b, head_color=ORANGE, head_size=18, body_size=15)
notes(s, "這是我們自己設計、自己訓練的模型 AdaptNet。核心想法：要預測「在新細胞裡關掉基因 X」，最好的線索是「其他細胞裡關掉 X 的實驗結果」。但每個來源的可信度不同，所以模型分三步：第一，替每個來源打分數加權，看那個細胞跟新細胞像不像、資料品質好不好、跟其他來源一不一致；第二，把反應拆成 200 組會一起變動的基因（基因程式），再加上細節；第三，根據新細胞對照組的狀態，調整每組程式的強弱。訓練時每一步都隨機藏起一種細胞，要模型只用其他細胞來預測它，這樣模型一直在練習預測「沒看過的細胞」。模型故意做得很小，大約兩萬個參數，因為我們試過大模型，它會把 15 種細胞背起來，換到新細胞反而更差。期末版本是用三個不同亂數種子訓練的模型取平均，讓結果更穩定。")

# 7 genenet ---------------------------------------------------------------------------------------------
s = new_slide("從沒被實驗測過的基因：GeneNet4", n=7, foot="也試過：TF→目標基因知識（CollecTRI）、蛋白質語言模型向量（ProtT5），都沒有再提升")
text(s, 0.6, 1.6, 7.1, 4.9, ["如果某個基因從來沒被任何實驗關掉過，就沒有「來源」可以參考。",
                            [("GeneNet4 改找", {}), ("功能最像的 32 個基因", {"bold": True, "color": NAVY}), ("（STRING 蛋白網路 + GO 功能註解），用它們被關掉時的反應來推測，並學會該相信多少。", {})],
                            [("期末考的 300 個基因都有測量資料，所以期末主要靠 AdaptNet；GeneNet4 讓模型能處理「任何基因」。", {"color": MUTED, "size": 16})]],
     size=19, spacing=1.4, after=14)
stat(s, 8.1, 1.6, 4.633, 2.2, "0.152", "方向相似度（1 = 完全一致）", label="有測量的基因（藏起細胞類型）", fill=TINT, num_color=BLUE, cap_color=TEXT, num_size=40, cap_size=14)
stat(s, 8.1, 4.05, 4.633, 2.2, "0.025", "試過的方法中最好，但訊號很弱", label="沒測量的基因＋沒看過的細胞", fill=TINT, num_color=ORANGE, cap_color=TEXT, num_size=40, cap_size=14)
notes(s, "通用模型還要能處理「從來沒被實驗關掉過的基因」。這種基因沒有任何來源資料，所以我們做了 GeneNet4：它會從 STRING 蛋白交互網路和 Gene Ontology 功能註解，找出功能最像的 32 個基因，拿它們被關掉時的反應來推測，並且學會這種推測要相信多少。右邊是驗證結果：有測量資料的基因，在沒看過的細胞上方向相似度約 0.15；完全沒測量的基因，再加上沒看過的細胞，只有 0.025。這是我們試過的方法裡最好的，但訊號很弱，代表只靠基因知識很難預測。期末考的 300 個基因都有測量資料，所以期末主要靠 AdaptNet。")

# 8 render ----------------------------------------------------------------------------------------------
s = new_slide("從預測到 400 顆細胞：三個步驟", n=8)
for k, (h, b) in enumerate([("算總量", "模型預測每個基因在 400 顆細胞的總量變化；整體幅度依評分規則校準，把握度高的擾動給大一點"),
                            ("抽真實細胞當底", "從新細胞的對照組隨機抽 400 顆真實細胞，保留真實的細胞差異和雜訊"),
                            ("調整計數", "依預測調高或調低每顆細胞的基因計數，最後讓每個基因的總量剛好等於預測值")]):
    card(s, 0.6 + k * (cw3 + 0.3), 1.55, cw3, 3.25, h, b, head_color=BLUE, head_size=20, body_size=15, num=k + 1)
box(s, 0.6, 5.05, W, 1.55, TINT2)
text(s, 0.85, 5.2, W - 0.5, 0.45, ["為什麼要分開控制「總量」和「單顆細胞」"], size=18, color=NAVY, bold=True)
text(s, 0.85, 5.7, W - 0.5, 0.8, ["評分同時看平均表現（誤差、區分度）和單顆細胞的分布（哪些基因統計上顯著），兩者的最佳做法不同"], size=15, color=TEXT)
notes(s, "模型輸出的是「每個基因平均會變多少」，但比賽要的是一顆一顆的細胞，所以我們有一套產生細胞的流程。第一步，算出每個基因在 400 顆細胞裡的總量；整體變化幅度要依評分規則校準，因為評分程式對「預測得太誇張」會扣分；把握度高的擾動給大一點、把握度低的給小一點。第二步，從新細胞自己的對照組隨機抽 400 顆真實細胞當底，這樣細胞之間的真實差異和雜訊都保留下來。第三步，依照預測，把每顆細胞的計數調高或調低，最後讓每個基因的總量剛好等於預測。之所以分開控制，是因為評分同時看平均表現和單顆細胞的統計顯著性，兩者的最佳做法不一樣。")

# 9 validation ------------------------------------------------------------------------------------------
s = new_slide("驗證方法：故意藏起答案再考模型", n=9)
for k, (h, b) in enumerate([("藏起整種細胞", "7 組細胞輪流藏起來，用其餘資料訓練後預測它（7 折交叉驗證），量方向相似度與區分度"),
                            ("藏起一批基因", "一批基因完全不給模型看，再預測它們，測試「從沒被測過的基因」"),
                            ("模擬考", "把答案已知的 H1、HCT116、KOLF 細胞當成期末考，跑完整流程，用官方評分程式打分")]):
    card(s, 0.6 + k * (cw3 + 0.3), 1.55, cw3, 3.4, h, b, head_color=BLUE, head_size=20, body_size=15, num=k + 1)
box(s, 0.6, 5.2, W, 1.35, TINT2)
text(s, 0.85, 5.35, W - 0.5, 1.1, [[("採用規則：", {"bold": True, "color": NAVY}), ("每個改動都要在模擬考的多數組別上進步、且其他細胞沒有明顯退步，才會放進期末版本", {})]], size=17, spacing=1.35)
notes(s, "因為期末考的答案看不到，所有改進都必須先自己驗證。我們用三種方法：第一，把整種細胞藏起來，用其他細胞訓練後預測它，7 組細胞輪流，這叫 7 折交叉驗證。第二，把一批基因藏起來，測試模型能不能預測從沒被測過的基因。第三，模擬考：拿答案已知的細胞，例如 H1、HCT116、KOLF，假裝它們是期末考，跑完整個流程，再用比賽官方的評分程式打分。我們的規則是：一個改動要在模擬考大多數組別都進步、其他細胞也沒有明顯退步，才放進期末版本，避免只對某一種細胞有效。")

# 10 benchmark ------------------------------------------------------------------------------------------
s = new_slide("結果一：在沒看過的細胞上，訓練的模型比較準", n=10, foot="scDFM（現成的生成模型）在專案初期測過，在新細胞上學不到擾動訊號，故未列入")
cd = CategoryChartData(); cd.categories = ["各來源直接平均（原方法）", "AdaptNet", "AdaptNet＋RPE1 資料"]
cd.add_series("方向相似度", (0.111, 0.131, 0.152))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5), Inches(7.7), Inches(4.4), cd)
ch = gf.chart; ch.has_legend = False; ch.has_title = True
ch.chart_title.text_frame.text = "方向相似度（藏起細胞，7 折平均）"
style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 16, NAVY, True)
pl = ch.plots[0]; pl.gap_width = 80; pl.has_data_labels = True
dl = pl.data_labels; dl.number_format = "0.000"; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(16); dl.font.bold = True; dl.font.color.rgb = rgb(NAVY)
ser = pl.series[0]
for i, c in enumerate(["B8C4D3", "6D8BB2", BLUE]):
    pt = ser.points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
va = ch.value_axis; va.maximum_scale = 0.18; va.minimum_scale = 0; va.visible = False; va.has_major_gridlines = False
ca = ch.category_axis; ca.tick_labels.font.size = Pt(14); ca.tick_labels.font.color.rgb = rgb(TEXT); ca.format.line.color.rgb = rgb("C7CFDA")
stat(s, 8.6, 1.5, 4.133, 2.3, "+37%", "方向相似度比原方法提升")
box(s, 8.6, 4.05, 4.133, 1.85, TINT)
text(s, 8.85, 4.2, 3.7, 1.6, ["區分度：0.738 → 0.762 → 0.766", [("（0.5 = 亂猜，1 = 每個擾動都分得出來）", {"size": 13, "color": MUTED})]], size=16, color=NAVY, spacing=1.35)
notes(s, "第一個結果：在「藏起整種細胞」的測試裡，我們訓練的 AdaptNet 比原本「把各來源直接平均」的方法準。方向相似度從 0.111 提升到 0.131，加入 RPE1 資料後到 0.152，大約提升 37%；區分度從 0.738 提升到 0.766。專案一開始我們也重現並測試過一個現成的生成模型 scDFM，但在新細胞上它基本上學不到擾動的訊號，接近亂猜，所以後來改走自己設計的小模型。要誠實說明：0.15 這個數字本身不高，代表「把別種細胞的結果轉移到新細胞」本來就很難。")

# 11 progress -------------------------------------------------------------------------------------------
s = new_slide("結果二：模擬考總分一路進步", n=11, foot="H1 模擬考 5 組平均、官方評分程式；座標軸從 0.25 開始")
cd = CategoryChartData()
cd.categories = ["原方法（各來源平均）", "v2：訓練模型預測總量", "v3：＋RPE1 資料", "v4：設定調整＋3 模型平均", "v5：只移除 5 個雜訊方向"]
cd.add_series("總分", (0.272, 0.291, 0.293, 0.306, 0.306))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.45), Inches(W), Inches(4.15), cd)
ch = gf.chart; ch.has_legend = False; ch.has_title = False
pl = ch.plots[0]; pl.gap_width = 55; pl.has_data_labels = True
dl = pl.data_labels; dl.number_format = "0.000"; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(16); dl.font.bold = True; dl.font.color.rgb = rgb(NAVY)
ser = pl.series[0]
for i, c in enumerate(["B8C4D3", "8FA6C2", "6D8BB2", "3F6FA6", BLUE]):
    pt = ser.points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
va = ch.value_axis; va.minimum_scale = 0.25; va.maximum_scale = 0.32; va.visible = False; va.has_major_gridlines = False
ca = ch.category_axis; ca.reverse_order = True; ca.tick_labels.font.size = Pt(15); ca.tick_labels.font.color.rgb = rgb(TEXT); ca.format.line.color.rgb = rgb("C7CFDA")
box(s, 0.6, 5.8, W, 0.95, TINT2)
text(s, 0.85, 5.9, W - 0.5, 0.75, ["v5 在 H1 持平，但在另外兩種細胞的區分度大幅提升：HCT116 0.70 → 0.75、KOLF 0.60 → 0.65（原始分數）"], size=16, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "第二個結果是模擬考的總分。原方法是 0.272；換成訓練的模型來預測總量（v2）到 0.291；加入 RPE1 資料（v3）0.293；接著重新調整流程設定、並用三個模型取平均（v4）到 0.306；最後的 v5 只改一個設定，在 H1 持平，但在另外兩種細胞 HCT116 和 KOLF 上，區分度大幅提升，所以是比較穩健的版本。每一步都是用官方評分程式驗證過才採用。注意這張圖的座標軸從 0.25 開始，所以差距看起來比較明顯；總分進步不大，是因為六項裡有幾項（跟顯著基因方向有關的）很難動，後面會說明原因。")

# 12 submissions ----------------------------------------------------------------------------------------
s = new_slide("期末提交紀錄（D、E、F 三種細胞）", n=12, foot="每次上傳前都先在本機用官方格式檢查器驗證")
table(s, 0.6, 1.5, W, [1.2, 2.2, 6.6, 2.133], [
    ["版本", "上傳時間", "內容", "模擬考"],
    ["v1", "10/22 19:44", "原方法：各來源平均（保底版）", "0.272"],
    ["v2", "10/23 01:30", "改用訓練的 AdaptNet 預測總量", "0.291"],
    ["v4", "10/23 18:40", "＋RPE1、流程設定調整、3 模型平均", "0.306"],
    ["v5", "10/24 00:24", "＋只移除 5 個雜訊方向（目前計分版本）", "0.306"]], size=16, row_h=0.58, highlight_last=True, bold_last=True, right_cols=(3,))
for k, (h, b) in enumerate([("規則", "每天最多上傳 2 次，只算最後一次；分數在比賽結束、公布得獎者時才公開"),
                            ("驗證期排行榜（A/B/C 細胞）", "舊方法最佳 0.239，第 165 名 / 1,635 隊；前 40 名約 0.30–0.35")]):
    card(s, 0.6 + k * (cw + 0.3), 4.6, cw, 1.95, h, b, head_color=NAVY, head_size=18, body_size=15)
notes(s, "這是期末考的提交紀錄。期末考從 10 月 22 日開始，每天最多傳兩次、只算最後一次，而且分數要等比賽結束才公布，所以我們只能靠自己的模擬考判斷好壞。v1 是保底版本，用原本的方法；v2 改成訓練的模型；v4 再加上資料和設定調整；v5 是目前計分的版本，已經評分成功。另外，在之前的驗證期（細胞 A、B、C）有公開排行榜，我們用舊方法最好是 0.239，在 1,635 隊裡排第 165 名，前 40 名大約 0.30 到 0.35。新方法是在驗證期結束後才做的，所以排行榜上沒有新方法的成績。")

# 13 findings -------------------------------------------------------------------------------------------
s = new_slide("主要發現：瓶頸在「方向」猜不準", dark=True, n=13, foot="在 7 種藏起來的細胞上測量；模擬實驗：方向若全對，這幾項分數可大幅提升")
stat(s, 0.6, 1.55, cw, 2.75, "70–80%", "最有把握的基因，升／降方向猜對", label="必需基因被關掉時", fill=NAVY2, num_color=ICE, cap_color=LIGHT2, num_size=54)
stat(s, 0.6 + cw + 0.3, 1.55, cw, 2.75, "53–58%", "接近丟銅板（50%）", label="非必需基因被關掉時（期末考）", fill=NAVY2, num_color=ORANGE_L, cap_color=LIGHT2, num_size=54)
tb = text(s, 0.6, 4.6, W, 1.9, ["「區分度」已達實驗重複水準的約 87%：模型很會分辨不同擾動",
                                "「哪些基因顯著改變、往哪變」接近基準：別種細胞的結果，轉移到新細胞時方向常常不同"], size=18, color=LIGHT, spacing=1.35, after=10)
for p in tb.text_frame.paragraphs:
    pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(Emu(Inches(0.3)))); pPr.set("indent", str(-Emu(Inches(0.3))))
    bu = pPr.makeelement(qn("a:buChar"), {"char": "•"}); pPr.append(bu)
notes(s, "這頁是最重要的發現。我們仔細分析了為什麼「顯著基因、方向」這幾項分數低。結論是：問題不在產生細胞的方式，而在每個基因到底是「上升」還是「下降」猜不準。當被關掉的是必需基因（細胞生存必需的基因）時，模型最有把握的那些預測，方向可以猜對 70 到 80%；但當被關掉的是非必需基因時，只有 53 到 58%，接近丟銅板。期末考的 300 個基因都是非必需基因，所以這幾項很難提升。相對地，區分度已經達到實驗重複水準的大約 87%。我們也做了一個模擬實驗：如果方向全部猜對，這幾項分數會大幅提升，證明瓶頸就在方向。")

# 14 tried ----------------------------------------------------------------------------------------------
s = new_slide("試過但沒有採用的方法（都有驗證數據）", n=14, foot="詳細數字記錄在專案筆記 SUMMARY.md")
table(s, 0.6, 1.45, W, [5.0, 7.133], [
    ["嘗試", "結果"],
    ["加入這種細胞的「共同反應」", "若事先知道答案可 +0.03，但無法從其他細胞預測"],
    ["用細胞週期比例解釋反應", "只能解釋約 4% 的變化"],
    ["專門學「方向」的訓練目標", "結果和原本相同"],
    ["用對照組細胞的自然差異推測", "等於亂猜（基因表現太低、太稀疏）"],
    ["更真實的「下降基因」產生方式", "三種細胞結果不一致，先不採用"],
    ["訓練時加重高表現量基因", "在誤差範圍內，沒有差別"],
    ["預測每個擾動會影響幾個基因", "反而變差（少報顯著基因會被扣分）"],
    ["更多資料：ELEM-1200RS、LINCS、神經細胞", "沒有幫助或變差"]], size=15, row_h=0.56)
notes(s, "這頁列出試過但沒有採用的方法，每一個都有驗證數據。例如：加入細胞的「共同反應」——如果事先知道答案，分數可以多 0.03，但我們用其他細胞去預測它，準確度太低，有時甚至猜反；用細胞週期比例來解釋反應，只能解釋大約 4%；專門為「方向」設計訓練目標，結果完全一樣，因為模型主要是在重新分配各來源的權重，方向本身由來源資料決定；用對照組細胞之間的自然差異來推測，因為這些基因表現量太低、資料太稀疏，等於亂猜。更多外部資料也試過，沒有幫助。這些負面結果也很重要：它們說明目前的限制來自資料，而不是模型技巧。")

# 15 limits ---------------------------------------------------------------------------------------------
s = new_slide("限制與下一步", n=15)
text(s, 0.6, 1.45, cw, 0.5, ["限制"], size=22, color=ORANGE, bold=True)
text(s, 0.6 + cw + 0.3, 1.45, cw, 0.5, ["下一步"], size=22, color=BLUE, bold=True)
for k, (h, b) in enumerate([("期末考細胞和訓練資料差很多", "從標記基因看：D 像骨髓性白血病細胞、E 像纖維母細胞、F 像上皮／間質癌細胞；訓練資料中沒有相似細胞"),
                            ("公開資料已經用盡", "能涵蓋這 300 個基因的大型公開實驗都已加入")]):
    card(s, 0.6, 2.05 + k * 2.3, cw, 2.1, h, b, head_color=NAVY, head_size=18, body_size=15)
for k, (h, b) in enumerate([("持續小幅改進", "每項都在三種細胞的模擬考上確認不退步才採用"),
                            ("11/5 前定案", "選定最後版本上傳（只算最後一次）"),
                            ("整理成果", "整理程式碼、實驗紀錄與書面報告")]):
    card(s, 0.6 + cw + 0.3, 2.05 + k * 1.5, cw, 1.35, h, b, head_color=NAVY, head_size=17, body_size=14, fill=TINT2)
notes(s, "限制有兩個。第一，期末考的三種細胞和我們的訓練資料差很多：從它們的標記基因看，D 像骨髓性白血病細胞，E 像纖維母細胞，F 像上皮或間質的癌細胞，而我們的訓練資料裡沒有相似的細胞，所以預測本來就困難。第二，能涵蓋這 300 個目標基因的大型公開實驗，我們都已經加進來了。下一步：在截止前持續做小幅改進，每一項都要在三種細胞的模擬考上確認不退步才採用；11 月 5 日前選定最後版本；同時整理程式碼、實驗紀錄和書面報告。")

# 16 compute --------------------------------------------------------------------------------------------
s = new_slide("運算資源與工作量", dark=True, n=16)
for k, (h, b) in enumerate([("訓練：本機 GPU", "RTX 4060 Ti 16 GB；模型小，每個模型訓練約 1–3 分鐘，可以大量反覆驗證"),
                            ("雲端：Brev", "本機記憶體不夠時，用於大檔案打包與批次評分；共約 US$6（US$500 額度）"),
                            ("驗證量", "約 100 次官方評分模擬考、60 多組交叉驗證實驗")]):
    card(s, 0.6 + k * (cw3 + 0.3), 1.55, cw3, 3.3, h, b, head_color=ICE, fill=NAVY2, head_size=20, body_size=16, body_color=LIGHT2, num=k + 1)
text(s, 0.6, 5.4, W, 0.8, ["謝謝教授！歡迎提問。"], size=26, color=LIGHT, bold=True)
notes(s, "最後說明運算資源。模型的訓練和大部分驗證都在本機的 GPU（RTX 4060 Ti，16 GB）上完成；因為模型很小，每個模型訓練大約一到三分鐘，所以可以做大量的反覆驗證。教授提供的 Brev 雲端額度，我們用在本機記憶體不夠的時候，例如打包幾 GB 的提交檔和批次評分，目前總共大約用了 6 美元。驗證量方面，我們跑了大約 100 次官方評分的模擬考，以及 60 多組交叉驗證實驗。謝謝教授，歡迎提問。")

prs.core_properties.title = "虛擬細胞挑戰賽 進度報告"
prs.save(OUT)
print("wrote", OUT, len(prs.slides), "slides")
