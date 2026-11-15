"""Generate docs/featransform_architecture.drawio and an SVG preview from one layout.

Every primitive is emitted twice: as an mxCell (draw.io) and as SVG (preview), so
the preview shows exactly the geometry draw.io will render.
Values in the example tables come from a real Featransform 2.4.0 run
(standard preset, random_state=42) - see example_run.py.
"""
import html
import sys
from xml.sax.saxutils import escape

OUT_DRAWIO, OUT_SVG = sys.argv[1], sys.argv[2]
W, H = 1600, 2330

INTER = "Inter"
MONO = "JetBrains Mono"
FS_INTER = "fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DInter;"
FS_MONO = "fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DJetBrains%2BMono;"
SVG_INTER = "Inter, Helvetica Neue, Arial, sans-serif"
SVG_MONO = "JetBrains Mono, Menlo, monospace"

INK, INK2, MUTED, LINE = "#0f172a", "#475569", "#64748b", "#e1e4e8"

STAGE = {
    "pre": dict(bg="#eff6ff", border="#bfdbfe", accent="#3b82f6", title="#1e40af", chip="#dbeafe", box="#93c5fd"),
    "fe": dict(bg="#f0fdf4", border="#bbf7d0", accent="#16a34a", title="#166534", chip="#dcfce7", box="#86efac"),
    "sel": dict(bg="#faf5ff", border="#e9d5ff", accent="#9333ea", title="#6b21a8", chip="#f3e8ff", box="#d8b4fe"),
    "out": dict(bg="#fff7ed", border="#fed7aa", accent="#ea580c", title="#9a3412", chip="#ffedd5", box="#fdba74"),
}

cells, svg = [], []
_n = [0]


def nid(prefix):
    _n[0] += 1
    return f"{prefix}-{_n[0]}"


def q(s):
    return escape(s, {'"': "&quot;"})


# --------------------------------------------------------------------- labels
# A label is a list of lines; each line is (text, size, color, bold, mono).
def L(text, size=10, color=INK2, bold=False, mono=True):
    return (text, size, color, bold, mono)


def label_html(lines):
    parts = []
    for text, size, color, bold, mono in lines:
        fam = f"font-family:{MONO};" if mono else ""
        weight = "font-weight:bold;" if bold else ""
        parts.append(
            f'<span style="font-size:{size}px;color:{color};{fam}{weight}">{html.escape(text)}</span>'
        )
    return "<br>".join(parts)


def svg_lines(x, y, w, h, lines, align="center", valign="middle", pad=6):
    lh = [ln[1] * 1.32 for ln in lines]
    total = sum(lh)
    if valign == "middle":
        cy = y + (h - total) / 2
    else:
        cy = y + pad
    anchor = {"center": "middle", "left": "start", "right": "end"}[align]
    tx = {"center": x + w / 2, "left": x + pad, "right": x + w - pad}[align]
    for (text, size, color, bold, mono), step in zip(lines, lh):
        base = cy + step * 0.78
        weight = ' font-weight="bold"' if bold else ""
        svg.append(
            f'<text x="{tx:.1f}" y="{base:.1f}" font-family="{SVG_MONO if mono else SVG_INTER}" '
            f'font-size="{size}" fill="{color}" text-anchor="{anchor}"'
            f'{weight} xml:space="preserve">{escape(text)}</text>'
        )
        cy += step


# ----------------------------------------------------------------- primitives
def vertex(x, y, w, h, style, value="", cid=None):
    cid = cid or nid("c")
    cells.append(
        f'<mxCell id="{cid}" value="{q(value)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
    )
    return cid


def rect(x, y, w, h, fill, stroke="none", sw=1, arc=None, dashed=False, cid=None):
    style = f"whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};"
    style += f"rounded=1;arcSize={arc};" if arc is not None else "rounded=0;"
    if dashed:
        style += "dashed=1;dashPattern=6 4;"
    vertex(x, y, w, h, style, "", cid)
    r = 0 if arc is None else min(w, h) * arc / 100
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    svg.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke if stroke != "none" else "none"}" stroke-width="{sw}"{dash}/>'
    )


def text(x, y, w, h, lines, align="left", valign="middle"):
    style = (
        f"text;html=1;whiteSpace=wrap;align={align};verticalAlign={valign};spacing=0;"
        f"spacingLeft={6 if align == 'left' else 0};spacingTop={6 if valign == 'top' else 0};"
        f"fontFamily={INTER};{FS_INTER}fillColor=none;strokeColor=none;"
    )
    vertex(x, y, w, h, style, label_html(lines))
    svg_lines(x, y, w, h, lines, align, valign)


def box(x, y, w, h, title, details, st, strong=False, title_size=12):
    s = STAGE[st]
    sw = 2.5 if strong else 1.5
    stroke = s["accent"] if strong else s["box"]
    lines = [L(title, title_size, INK, True, False)] + [L(d) for d in details]
    style = (
        f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={stroke};"
        f"strokeWidth={sw};align=center;verticalAlign=middle;spacing=4;fontFamily={INTER};{FS_INTER}"
        + ("shadow=1;" if strong else "")
    )
    vertex(x, y, w, h, style, label_html(lines))
    r = min(w, h) * 0.08
    svg.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r:.1f}" fill="#ffffff" '
        f'stroke="{stroke}" stroke-width="{sw}"/>'
    )
    svg_lines(x, y, w, h, lines)


def chip(x, y, w, h, label, fill, color, stroke="none", bold=True, size=10, mono=True):
    style = (
        f"rounded=1;arcSize=40;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
        f"fontColor={color};fontSize={size};fontStyle={1 if bold else 0};"
        f"fontFamily={MONO if mono else INTER};{FS_MONO if mono else FS_INTER}"
    )
    vertex(x, y, w, h, style, html.escape(label))
    svg.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1"/>'
    )
    svg_lines(x, y, w, h, [L(label, size, color, bold, mono)])


def chip_w(label, size=10):
    return round(len(label) * size * 0.62 + 18)


def circle(x, y, d, label, fill):
    style = (
        f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=none;fontColor=#FFFFFF;"
        f"fontSize=12;fontStyle=1;fontFamily={INTER};{FS_INTER}"
    )
    vertex(x, y, d, d, style, label)
    svg.append(f'<circle cx="{x + d / 2}" cy="{y + d / 2}" r="{d / 2}" fill="{fill}"/>')
    svg_lines(x, y, d, d, [L(label, 12, "#ffffff", True, False)])


def diamond(x, y, w, h, label_lines, fill, stroke):
    lines = [L(t, 11, "#ffffff", True, False) for t in label_lines]
    style = (
        f"rhombus;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
        f"fontFamily={INTER};{FS_INTER}"
    )
    vertex(x, y, w, h, style, label_html(lines))
    pts = f"{x + w / 2},{y} {x + w},{y + h / 2} {x + w / 2},{y + h} {x},{y + h / 2}"
    svg.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    svg_lines(x, y, w, h, lines)


def cylinder(x, y, w, h, fill, stroke):
    style = f"shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;size=9;fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
    vertex(x, y, w, h, style)
    ry = 9
    svg.append(
        f'<path d="M{x},{y + ry} L{x},{y + h - ry} A{w / 2},{ry} 0 0 0 {x + w},{y + h - ry} '
        f'L{x + w},{y + ry} A{w / 2},{ry} 0 0 0 {x},{y + ry} Z" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )
    svg.append(
        f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )


def edge(points, color="#94a3b8", dashed=False, arrow=True, width=1.5):
    (sx, sy), (tx, ty) = points[0], points[-1]
    mids = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points[1:-1])
    style = (
        f"edgeStyle=none;html=1;rounded=1;strokeColor={color};strokeWidth={width};"
        f"endArrow={'block' if arrow else 'none'};endFill=1;endSize=6;"
        + ("dashed=1;dashPattern=6 4;" if dashed else "")
    )
    cells.append(
        f'<mxCell id="{nid("e")}" value="" style="{style}" edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry"><mxPoint x="{sx}" y="{sy}" as="sourcePoint"/>'
        f'<mxPoint x="{tx}" y="{ty}" as="targetPoint"/>'
        + (f'<Array as="points">{mids}</Array>' if mids else "")
        + "</mxGeometry></mxCell>"
    )
    d = "M" + " L".join(f"{px},{py}" for px, py in points)
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    marker = f' marker-end="url(#arr-{color[1:]})"' if arrow else ""
    svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash}{marker}/>')
    markers.add(color)


markers = set()


def edge_label(x, y, w, label, color, bold=True, size=11, align="center"):
    text(x, y, w, 16, [L(label, size, color, bold, False)], align=align)


# Cell styles for tables: (fill, color, bold)
CELL = {
    "head": ("#f8fafc", INK2, True),
    "cell": ("#ffffff", "#334155", False),
    "nan": ("#fef2f2", "#dc2626", True),
    "imp": ("#fef3c7", "#92400e", True),
    "pre": ("#dbeafe", "#1e40af", True),
    "fe": ("#dcfce7", "#166534", True),
    "flag": ("#fee2e2", "#b91c1c", True),
    "sel": ("#f3e8ff", "#6b21a8", True),
    "tie": ("#faf5ff", "#7e22ce", False),
    "dim": ("#ffffff", "#94a3b8", False),
    "out": ("#ffedd5", "#9a3412", True),
    "idx": ("#f8fafc", "#94a3b8", False),
}


def table(x, y, cols, rows, head_styles=None, cell_styles=None, row_h=20, head_h=22, size=10):
    """cols: [(header, width)]; rows: list of lists. Returns the table bottom y."""
    head_styles = head_styles or {}
    cell_styles = cell_styles or {}

    def cell(cx, cy, w, h, val, key):
        fill, color, bold = CELL[key]
        style = (
            f"rounded=0;whiteSpace=wrap;html=1;strokeColor={LINE};fillColor={fill};fontColor={color};"
            f"fontSize={size};fontFamily={MONO};{FS_MONO}align=center;verticalAlign=middle;"
            f"fontStyle={1 if bold else 0};spacing=2;"
        )
        vertex(cx, cy, w, h, style, html.escape(str(val)))
        svg.append(f'<rect x="{cx}" y="{cy}" width="{w}" height="{h}" fill="{fill}" stroke="{LINE}" stroke-width="1"/>')
        svg_lines(cx, cy, w, h, [L(str(val), size, color, bold, True)])

    cx = x
    for c, (hdr, w) in enumerate(cols):
        cell(cx, y, w, head_h, hdr, head_styles.get(c, "head"))
        cx += w
    cy = y + head_h
    for r, row in enumerate(rows):
        cx = x
        for c, ((_, w), val) in enumerate(zip(cols, row)):
            key = cell_styles.get((r, c), cell_styles.get(("col", c), "cell"))
            if val == "⋮":
                key = "dim"
            cell(cx, cy, w, row_h, val, key)
            cx += w
        cy += row_h
    return cy


def panel(y, h, st, num, title, badges, chip_x=None):
    s = STAGE[st]
    rect(40, y, 1520, h, s["bg"], s["border"], 1.5, arc=1)
    circle(60, y + 15, 28, str(num), s["accent"])
    text(98, y + 15, 300, 28, [L(title, 13, s["title"], True, False)])
    bx = chip_x or 98 + len(title) * 10 + 40
    for b in badges:
        w = chip_w(b)
        chip(bx, y + 18, w, 22, b, s["chip"], s["title"])
        bx += w + 10


def caption(x, y, w, title, color):
    text(x, y, w, 16, [L(title, 10, color, True, False)])


def swatch(x, y, key, label, w=380):
    fill, color, _ = CELL[key]
    rect(x, y + 3, 14, 12, fill, color, 1)
    text(x + 20, y, w, 18, [L(label, 11, "#334155", False, False)])


# ======================================================================= LAYOUT
# ---- title bar
rect(40, 20, 1520, 60, "#1a1f36", arc=6)
text(70, 28, 800, 44, [L("Featransform — Automated Feature Engineering Pipeline", 20, "#ffffff", True, False)])
chip(1170, 36, 360, 28, "fit(X, y)  →  transform(X)  →  get_summary()", "#2d3561", "#E2E8F0", "#4a5280", size=11)

# ---- data source
cylinder(215, 108, 50, 64, "#3b82f6", "#1e40af")
text(282, 104, 320, 96, [
    L("Input Data", 14, INK, True, False),
    L("pandas DataFrame X · one row per record", 11, MUTED, False, False),
    L("numeric · categorical · datetime · id columns", 11, MUTED, False, False),
    L("target y: optional, required for feature selection", 11, MUTED, False, False),
], valign="top")
src_cols = [("row", 40), ("customer_id", 96), ("age", 56), ("income", 66), ("city", 72), ("signup", 96), ("y", 44)]
src_rows = [
    ["0", "1", "70.0", "41.2", "Lisbon", "2024-01-01", "1"],
    ["1", "2", "57.0", "NaN", "Porto", "2024-01-02", "0"],
    ["2", "3", "49.0", "58.8", "NaN", "2024-01-03", "1"],
    ["⋮", "⋮", "⋮", "⋮", "⋮", "⋮", "⋮"],
    ["41", "42", "19.0", "3.4", "Lisbon", "2024-02-11", "0"],
]
table(610, 102, src_cols, src_rows, cell_styles={(1, 3): "nan", (2, 4): "nan", ("col", 0): "idx"})
text(1110, 100, 440, 126, [
    L("Worked example used in every table below", 11, INK, True, False),
    L("400 rows · 7 missing income · 3 missing city", 11, MUTED, False, False),
    L("binary target y (259 × 0 / 141 × 1)", 11, MUTED, False, False),
    L("FTconfig standard preset · random_state = 42", 11, MUTED, False, False),
    L("All values are real output of Featransform 2.4.0.", 11, MUTED, False, False),
], valign="top")

# ---- 1 PREPROCESSING
P1 = 262
panel(P1, 352, "pre", 1, "PREPROCESSING", ["fit(X, y)", "transform(X)"])
bx = [60, 312, 564, 816, 1068, 1320]
BW, BY, BH = 210, P1 + 60, 80
pre_boxes = [
    ("Validate & Align", ["unique string column names", "y: same length, no NaN", "transform: fit column order"]),
    ("Expand Datetimes", ["year · month · day · week", "dayofweek · quarter · weekend", "month & day as sin / cos"]),
    ("Impute Missing", ["numeric: iterative (default)", "only NaN cells are written", "categorical: fit-time mode"]),
    ("Encode Categories", ["label code per category", "mapping learned at fit", "unseen / NaN → -1"]),
    ("Clean Columns", ["drop constant columns", "drop duplicate columns", "decided at fit, replayed"]),
    ("Processed Frame", ["16 numeric columns", "id column still present", "original scale"]),
]
for i, (t, d) in enumerate(pre_boxes):
    box(bx[i], BY, BW, BH, t, d, "pre", strong=(i == 5))
    if i:
        edge([(bx[i - 1] + BW, BY + BH / 2), (bx[i], BY + BH / 2)], STAGE["pre"]["box"])
edge([(240, 172), (240, BY)], "#94a3b8")

caption(60, P1 + 160, 700, "PROCESSED FRAME · rows 0, 1, 2, 41 of 400 · 16 columns", STAGE["pre"]["title"])
pre_cols = [("row", 36), ("customer_id", 84), ("age", 50), ("income", 62), ("city", 46),
            ("signup_year", 84), ("signup_month", 90), ("signup_day", 78), ("signup_dayofweek", 114),
            ("…", 34), ("signup_month_cos", 114), ("signup_day_sin", 102), ("signup_day_cos", 102)]
pre_rows = [
    ["0", "1", "70.0", "41.2", "1", "2024", "1", "1", "0", "…", "1.00", "0.00", "1.00"],
    ["1", "2", "57.0", "49.32", "2", "2024", "1", "2", "1", "…", "1.00", "0.20", "0.98"],
    ["2", "3", "49.0", "58.8", "0", "2024", "1", "3", "2", "…", "1.00", "0.39", "0.92"],
    ["⋮"] * 13,
    ["41", "42", "19.0", "3.4", "1", "2024", "2", "11", "6", "…", "0.87", "0.83", "-0.56"],
]
table(60, P1 + 180, pre_cols, pre_rows,
      head_styles={c: "pre" for c in range(5, 13)},
      cell_styles={(1, 3): "imp", (2, 4): "imp", ("col", 0): "idx"})
sx = 1150
swatch(sx, P1 + 176, "imp", "imputed: income by IterativeImputer, city by mode")
swatch(sx, P1 + 198, "pre", "new column: signup → 12 calendar features")
text(sx, P1 + 220, 390, 18, [L("city codes:  Faro 0 · Lisbon 1 · Porto 2", 11, "#334155", False, False)])
text(sx, P1 + 246, 390, 60, [
    L("ValidationError on duplicate column names,", 10, MUTED, False, False),
    L("X / y length mismatch, NaN in y (when selecting)", 10, MUTED, False, False),
    L("or a fit-time column missing at transform.", 10, MUTED, False, False),
], valign="top")

# ---- 2 FEATURE ENGINEERING
P2 = 652
panel(P2, 590, "fe", 2, "FEATURE ENGINEERING", ["3 ensembles", "fit on all rows"])
edge([(1425, BY + BH), (1425, 632), (56, 632), (56, P2 + 190), (76, P2 + 190)], "#94a3b8")

box(76, P2 + 140, 220, 100, "Model Input", ["drop id columns", "(customer_id)", "StandardScaler, fit once", "15 scaled columns"], "fe")

EX, EW, EH = 335, 500, 100
ens = [
    ("Anomaly Ensemble", "→ <model>_anomaly   0 = inlier · 1 = outlier",
     [("IsolationForest", True), ("LOF", True), ("OneClassSVM", False), ("EllipticEnvelope", False)]),
    ("Clustering Ensemble", "→ <model>_cluster   integer group label",
     [("KMeans k=5", True), ("MiniBatchKMeans", False), ("Birch", False), ("GaussianMixture", False), ("DBSCAN eps=auto", False)]),
    ("Dimensionality Ensemble", "→ pca_0 … pca_k   k from n_components",
     [("PCA 0.95", True), ("TruncatedSVD", False), ("FastICA", False)]),
]
fe = STAGE["fe"]
ey_list = []
for i, (title, out, models) in enumerate(ens):
    ey = P2 + 62 + i * (EH + 14)
    ey_list.append(ey)
    rect(EX, ey, EW, EH, "#ffffff", fe["box"], 1.5, arc=6)
    text(EX + 8, ey + 6, EW - 16, 20, [L(title, 12, INK, True, False)])
    text(EX + 8, ey + 28, EW - 16, 18, [L(out, 10, fe["title"], True, True)])
    cx = EX + 14
    for m, used in models:
        w = chip_w(m, 10)
        if used:
            chip(cx, ey + 62, w, 24, m, fe["chip"], fe["title"], fe["accent"], True)
        else:
            chip(cx, ey + 62, w, 24, m, "#ffffff", INK2, "#cbd5e1", False)
        cx += w + 8
    edge([(296, P2 + 190), (314, P2 + 190), (314, ey + EH / 2), (EX, ey + EH / 2)], fe["box"])
    edge([(EX + EW, ey + EH / 2), (858, ey + EH / 2), (858, P2 + 110), (880, P2 + 110)], fe["box"])

box(880, P2 + 62, 240, 96, "Name & Prune", ["prefix = model name [:3]", "name taken → suffix _ft", "constant on train → dropped"], "fe")
edge([(1000, P2 + 158), (1000, P2 + 210)], fe["box"])
box(880, P2 + 210, 240, 96, "Concatenate", ["processed frame, unscaled,", "id kept  +  engineered", "16 + 13 = 29 columns"], "fe", strong=True)

caption(1150, P2 + 62, 390, "WHAT THE STANDARD PRESET ADDS", fe["title"])
name_cols = [("model", 136), ("setting", 118), ("columns out", 126)]
name_rows = [
    ["IsolationForest", "contam. 0.05", "iso_anomaly"],
    ["LOF", "novelty mode", "lof_anomaly"],
    ["KMeans", "n_clusters=5", "kme_cluster"],
    ["PCA", "0.95 variance", "pca_0 … pca_9"],
    ["name collision", "pca_0 exists", "pca_0_ft"],
    ["constant output", "never fires", "dropped"],
]
table(1150, P2 + 82, name_cols, name_rows, row_h=22,
      cell_styles={**{(r, 2): "fe" for r in range(4)}, (4, 0): "dim", (5, 0): "dim", (4, 1): "dim", (5, 1): "dim"})
text(1150, P2 + 240, 390, 60, [
    L("15 scaled inputs → 2 + 1 + 10 = 13 new columns.", 11, "#334155", False, False),
    L("Filled chips = models in the standard preset;", 10, MUTED, False, False),
    L("others are available in optimized / complete.", 10, MUTED, False, False),
], valign="top")

caption(60, P2 + 410, 900, "ENGINEERED FRAME · 16 processed + 13 engineered = 29 columns", fe["title"])
fe_cols = [("row", 36), ("customer_id", 84), ("age", 50), ("income", 62), ("city", 46), ("signup_* ×12", 104),
           ("iso_anomaly", 92), ("lof_anomaly", 92), ("kme_cluster", 92), ("pca_0", 62), ("pca_1", 62), ("…", 34), ("pca_9", 62)]
fe_rows = [
    ["0", "1", "70.0", "41.2", "1", "…", "0", "0", "2", "-2.75", "-1.21", "…", "-1.15"],
    ["1", "2", "57.0", "49.32", "2", "…", "0", "0", "2", "-2.75", "-1.07", "…", "-1.15"],
    ["2", "3", "49.0", "58.8", "0", "…", "0", "0", "2", "-2.71", "-0.54", "…", "-0.94"],
    ["⋮"] * 13,
    ["41", "42", "19.0", "3.4", "1", "…", "1", "1", "2", "-2.56", "2.57", "…", "-0.30"],
]
table(60, P2 + 430, fe_cols, fe_rows,
      head_styles={c: "fe" for c in range(6, 13)},
      cell_styles={(4, 6): "flag", (4, 7): "flag", ("col", 0): "idx"})
swatch(1160, P2 + 426, "flag", "row 41 (age 19, income 3.4): both detectors fire")
text(1160, P2 + 452, 380, 80, [
    L("Across all 400 rows: iso_anomaly = 1 on 20 rows,", 11, "#334155", False, False),
    L("lof_anomaly = 1 on 17 rows; kme_cluster group sizes", 11, "#334155", False, False),
    L("116 · 78 · 93 · 79 · 34.", 11, "#334155", False, False),
    L("transform(X) replays the fitted models; nothing is refit.", 10, MUTED, False, False),
], valign="top")

# ---- 3 FEATURE SELECTION
P3 = 1282
sel = STAGE["sel"]
panel(P3, 712, "sel", 3, "FEATURE SELECTION", ["optimization", "uses y"], chip_x=345)
edge([(1120, P2 + 258), (1138, P2 + 258), (1138, 1262), (305, 1262), (305, P3 + 58)], "#94a3b8")

diamond(245, P3 + 58, 120, 72, ["optimization", "set?"], sel["accent"], "#7e22ce")
LX, LW = 140, 330
steps = [
    ("Holdout Split", ["75 % train / 25 % validation", "stratified on y", "single-row classes → train"]),
    ("Rank Features", ["CatBoost + XGBoost importance", "averaged, renormalized to 1", "fit on the train slice only"]),
    ("Next Threshold t", ["keep cumulative importance ≤ t", "t = 0.50 … 1.00 in 10 steps", "floor: min_features = 10"]),
    ("Score Subset", ["fresh XGBoost on train slice", "F1 on the validation slice", "tie → fewer columns wins"]),
]
sy0 = P3 + 160
for i, (t, d) in enumerate(steps):
    y = sy0 + i * 92
    box(LX, y, LW, 66, t, d, "sel")
    if i:
        edge([(305, y - 26), (305, y)], sel["box"])
edge([(305, P3 + 130), (305, sy0)], sel["accent"])
edge_label(312, P3 + 136, 40, "Yes", "#15803d", align="left")

dy = sy0 + 4 * 92 - 4
diamond(230, dy, 150, 76, ["more", "thresholds?"], sel["accent"], "#7e22ce")
edge([(305, sy0 + 3 * 92 + 66), (305, dy)], sel["box"])
t3y = sy0 + 2 * 92 + 33
edge([(230, dy + 38), (110, dy + 38), (110, t3y), (LX, t3y)], "#16a34a")
edge_label(118, dy + 18, 100, "Yes · next t", "#15803d", align="left")
edge([(305, dy + 76), (305, dy + 98)], sel["accent"])
edge_label(312, dy + 78, 40, "No", "#b91c1c", align="left")
box(LX, dy + 98, LW, 56, "Keep Best Subset", ["11 of 29 columns · F1 0.800"], "sel", strong=True)

# "No" branch: optimization=None keeps every engineered column
edge([(245, P3 + 94), (60, P3 + 94), (60, 2056), (150, 2056), (150, 2092)], "#b91c1c", dashed=True)
edge_label(150, P3 + 76, 60, "No", "#b91c1c")
text(66, P3 + 100, 160, 34, [L("all 29 columns", 10, "#b91c1c", True, False), L("are kept", 10, "#b91c1c", True, False)], valign="top")

# importance ranking table
IX = 520
caption(IX, P3 + 58, 520, "IMPORTANCE RANKING · train slice · top 16 of 29", sel["title"])
imp = [
    ("age", 0.354, 0.354), ("pca_4", 0.108, 0.462), ("signup_day_cos", 0.041, 0.503), ("signup_month_cos", 0.040, 0.542),
    ("income", 0.039, 0.582), ("pca_5", 0.035, 0.616), ("signup_month", 0.034, 0.650), ("pca_7", 0.030, 0.680),
    ("pca_9", 0.028, 0.708), ("pca_1", 0.028, 0.736), ("pca_8", 0.026, 0.762), ("pca_0", 0.025, 0.788),
    ("signup_dayofweek", 0.022, 0.810), ("signup_month_sin", 0.022, 0.831), ("pca_2", 0.021, 0.852), ("pca_6", 0.021, 0.873),
]
imp_cols = [("rank", 44), ("feature", 150), ("importance", 92), ("cumulative", 92), ("share", 150)]
imp_rows = [[str(i + 1), f, f"{v:.3f}", f"{c:.3f}", ""] for i, (f, v, c) in enumerate(imp)]
imp_rows.append(["⋮", "13 more ≤ 0.021", "⋮", "⋮", "⋮"])
istyles = {("col", 0): "idx"}
for r in range(11):
    istyles[(r, 1)] = "sel"
top = P3 + 78
table(IX, top, imp_cols, imp_rows, cell_styles=istyles)
bar_x = IX + 44 + 150 + 92 + 92 + 8
for i, (_, v, _) in enumerate(imp):
    rect(bar_x, top + 22 + i * 20 + 5, max(2, round(v / 0.354 * 134)), 10, sel["accent"] if i < 11 else "#d8b4fe", arc=30)
iw = 44 + 150 + 92 + 92 + 150
for after, lbl, col in [(11, "t = 0.78 → 11 cols", sel["accent"]), (14, "t = 0.83 → 14 cols", "#a855f7")]:
    ly = top + 22 + after * 20
    edge([(IX - 6, ly), (IX + iw + 6, ly)], col, dashed=True, arrow=False, width=2)
    text(IX + iw - 132, ly - 17, 132, 15, [L(lbl, 10, col, True, True)], align="right")

# kept / dropped columns, coloured by origin
KY = P3 + 462
caption(IX, KY, 520, "RESULT · 29 columns in → 11 kept", sel["title"])
groups = [
    ("kept", [("age", "orig"), ("income", "orig"), ("signup_month", "dt"), ("signup_month_cos", "dt"),
              ("signup_day_cos", "dt"), ("pca_1", "eng"), ("pca_4", "eng"), ("pca_5", "eng"),
              ("pca_7", "eng"), ("pca_8", "eng"), ("pca_9", "eng")]),
    ("dropped", [("customer_id", "orig"), ("city", "orig"), ("signup_year", "dt"), ("9 more signup_*", "dt"),
                 ("iso_anomaly", "eng"), ("lof_anomaly", "eng"), ("kme_cluster", "eng"), ("pca_0 · 2 · 3 · 6", "eng")]),
]
pal = {"orig": ("#f1f5f9", "#334155", "#cbd5e1"), "dt": (STAGE["pre"]["chip"], STAGE["pre"]["title"], STAGE["pre"]["box"]),
       "eng": (STAGE["fe"]["chip"], STAGE["fe"]["title"], STAGE["fe"]["box"])}
gy = KY + 22
for gname, items in groups:
    text(IX, gy, 70, 24, [L(gname, 10, sel["title"] if gname == "kept" else MUTED, True, False)])
    cx, rowy = IX + 70, gy
    for name, kind in items:
        w = chip_w(name, 9)
        if cx + w > IX + 528:
            cx, rowy = IX + 70, rowy + 30
        f, c, b = pal[kind]
        if gname == "dropped":
            f, c, b = "#ffffff", "#94a3b8", "#e2e8f0"
        chip(cx, rowy + 1, w, 22, name, f, c, b, gname == "kept", size=9)
        cx += w + 6
    gy = rowy + 36
for i, (kind, lbl) in enumerate([("orig", "original"), ("dt", "datetime-derived"), ("eng", "engineered")]):
    f, c, b = pal[kind]
    rect(IX + 70 + i * 150, gy + 4, 14, 12, f, b, 1)
    text(IX + 90 + i * 150, gy, 130, 20, [L(lbl, 10, MUTED, False, False)])

# threshold sweep table
SX = 1080
caption(SX, P3 + 58, 470, "THRESHOLD SWEEP · F1 on the validation slice", sel["title"])
sw_cols = [("t", 96), ("cols", 54), ("F1", 64), ("outcome", 246)]
sw_rows = [
    ["all (base)", "29", "0.784", "reference score"],
    ["0.50 – 0.72", "10", "0.789", "10-column floor, repeats skipped"],
    ["0.78", "11", "0.800", "✓ selected"],
    ["0.83", "14", "0.800", "tie: 11 columns win"],
    ["0.89", "16", "0.767", "worse"],
    ["0.94", "19", "0.784", "= baseline"],
    ["1.00", "29", "0.784", "full set, score reused"],
]
sws = {(2, c): "sel" for c in range(4)}
sws.update({(3, c): "tie" for c in range(4)})
sws.update({(4, 2): "dim", (4, 3): "dim"})
table(SX, P3 + 78, sw_cols, sw_rows, row_h=24, cell_styles=sws)
text(SX, P3 + 278, 470, 200, [
    L("How to read it", 11, INK, True, False),
    L("Ten thresholds yield six distinct subsets: t = 0.50 – 0.72", 11, "#334155", False, False),
    L("all fall under the 10-column floor and are scored once.", 11, "#334155", False, False),
    L("11 and 14 columns tie at F1 0.800, so the smaller wins.", 11, "#334155", False, False),
    L("6 of the 11 kept columns are engineered (pca_*);", 11, "#334155", False, False),
    L("customer_id is ranked like any column and falls below the cut.", 11, "#334155", False, False),
], valign="top")
rect(SX, P3 + 440, 460, 74, "#ffffff", "#e9d5ff", 1, arc=8)
text(SX + 4, P3 + 446, 452, 64, [
    L("Caveat", 10, sel["title"], True, False),
    L("The unsupervised models were fit on every row, validation", 10, MUTED, False, False),
    L("included, so read scores as a ranking between subsets,", 10, MUTED, False, False),
    L("not as an unbiased estimate of generalization.", 10, MUTED, False, False),
], valign="top")

# ---- 4 OUTPUT
P4 = 2034
out = STAGE["out"]
panel(P4, 276, "out", 4, "OUTPUT", ["fitted pipeline"])
edge([(305, dy + 154), (305, 2092)], sel["accent"])

rect(60, 2092, 790, 198, "#ffffff", out["box"], 1.5, arc=4)
text(72, 2100, 600, 22, [L("fit_transform(X, y)  →  DataFrame", 13, out["title"], True, True)])
o_cols = [("row", 36), ("age", 48), ("income", 58), ("signup_month", 90), ("signup_month_cos", 112), ("signup_day_cos", 100),
          ("pca_1", 52), ("pca_4", 52), ("pca_5", 52), ("pca_7", 52), ("pca_8", 52), ("pca_9", 52)]
o_rows = [
    ["0", "70.0", "41.2", "1", "1.00", "1.00", "-1.21", "0.22", "1.51", "-1.18", "1.06", "-1.15"],
    ["1", "57.0", "49.32", "1", "1.00", "0.98", "-1.07", "0.45", "0.38", "-0.99", "1.07", "-1.15"],
    ["2", "49.0", "58.8", "1", "1.00", "0.92", "-0.54", "0.04", "1.47", "0.90", "1.17", "-0.94"],
    ["⋮"] * 12,
    ["41", "19.0", "3.4", "2", "0.87", "-0.56", "2.57", "-2.98", "-1.33", "-1.03", "0.76", "-0.30"],
]
table(76, 2130, o_cols, o_rows, head_styles={c: "out" for c in range(1, 12)}, cell_styles={("col", 0): "idx"})
text(72, 2258, 760, 20, [L("400 rows × 11 columns · 5 original + 6 engineered · transform(X_new) returns the same columns", 10, MUTED, False, False)])

rect(870, 2092, 330, 198, "#ffffff", out["box"], 1.5, arc=4)
text(882, 2100, 310, 22, [L("transform(X_new)  →  DataFrame", 13, out["title"], True, True)])
text(882, 2128, 310, 160, [
    L("1  align to the fit column order", 10, "#334155"),
    L("   extras ignored · missing → error", 10, MUTED),
    L("2  replay datetime · imputer ·", 10, "#334155"),
    L("   encoder · scaler · ensembles", 10, "#334155"),
    L("3  keep the 11 selected columns", 10, "#334155"),
    L("4  check against the fit schema", 10, "#334155"),
    L("   nothing is refit", 10, MUTED),
], valign="top")

rect(1220, 2092, 320, 198, "#ffffff", out["box"], 1.5, arc=4)
text(1232, 2100, 300, 22, [L("Fitted pipeline", 13, out["title"], True, False)])
text(1232, 2128, 300, 160, [
    L("get_summary()", 10, "#334155"),
    L("get_feature_importance()", 10, "#334155"),
    L("get_optimization_history()", 10, "#334155"),
    L("get_pipeline_schema()", 10, "#334155"),
    L("PipelineSerializer.save / load", 10, "#334155"),
    L("   type and version checked", 10, MUTED),
    L("training rows are not stored", 10, MUTED),
], valign="top")

# ================================================================== EMIT FILES
drawio = (
    '<mxfile host="app.diagrams.net" agent="Featransform" version="24.0">\n'
    '  <diagram id="featransform-arch" name="Featransform 2.4.0 Pipeline">\n'
    f'    <mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
    f'fold="1" page="0" pageScale="1" pageWidth="{W}" pageHeight="{H}" background="#FFFFFF" math="0" shadow="0">\n'
    '      <root>\n        <mxCell id="0"/>\n        <mxCell id="1" parent="0"/>\n'
    + "".join(f"        {c}\n" for c in cells)
    + "      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n"
)
with open(OUT_DRAWIO, "w", encoding="utf-8") as f:
    f.write(drawio)

defs = "".join(
    f'<marker id="arr-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
    for c in sorted(markers)
)
with open(OUT_SVG, "w", encoding="utf-8") as f:
    f.write(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
        f"<defs>{defs}</defs><rect width='{W}' height='{H}' fill='#ffffff'/>" + "".join(svg) + "</svg>"
    )
print(f"{len(cells)} cells")
