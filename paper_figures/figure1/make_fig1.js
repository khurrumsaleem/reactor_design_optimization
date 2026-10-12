// Figure 1 for the NMI resubmission (October 2026), editable PPTX.
//
// Authored at 2x print scale: the slide is 14.4 x 12.7 in and prints at 50 %
// as 183 mm x 161 mm. Diagram text is Arial 7 pt at print size (14 pt in the
// file); panel letters 8 pt bold; panel titles, policy names and the three
// attribution terms bold. Data charts (chart_*.png, from make_charts.py) are
// placed at the print size they were drawn at, so their 7 pt labels and 6 pt
// tick numbers match the diagram text and stay aligned in every viewer.
// 3D assemblies come from render_blender.py (Cycles); the PyVista renders in
// assets/assembly_*.png are the fallback when the Blender files are absent.
// Colour appears only on physical quantities (OpenMC thermal flux and pin
// fission rate, measured reaction yield) and on the Gd-bearing rods.
//
// Run: node make_fig1.js --output /path/to/fig1_concept.pptx
// Requires pptxgenjs (or set PPTXGENJS to its module path); see README.md.

const fs = require("fs");
const path = require("path");
const PPTX = process.env.PPTXGENJS || "pptxgenjs";
const pptxgen = require(PPTX);

const HERE = __dirname;
const A = path.join(HERE, "assets");
const D = JSON.parse(fs.readFileSync(path.join(HERE, "data/fig1_assets.json"), "utf8"));
const has = (n) => fs.existsSync(path.join(A, n));
const IMG_IN = has("blender_input.png") ? "blender_input.png" : "assembly_input.png";
const IMG_OUT = has("blender_output.png") ? "blender_output.png" : "assembly_output.png";

function pngSize(name) { // [width, height] in px from the PNG header
  const b = fs.readFileSync(path.join(A, name));
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
const pngAspect = (name) => { const [w, h] = pngSize(name); return w / h; };

const S = 2;
const P = (v) => v * S, pt = (v) => v * S;
const FW = 7.2, FH = 6.35;

const pres = new pptxgen();
pres.defineLayout({ name: "FIG1", width: P(FW), height: P(FH) });
pres.layout = "FIG1";
const sl = pres.addSlide();
sl.background = { color: "FFFFFF" };

const C = { ink: "1F2328", line: "3F4650", gd: "0072B2", gdE: "00527F", bar: "C9CED3", dark: "4A5057", white: "FFFFFF" };
const F = "Arial";
const FS = { body: 7, letter: 8, small: 6 };

// ------------------------------------------------------------------ helpers
function T(text, x, y, w, h, o = {}) {
  sl.addText(text, {
    x: P(x), y: P(y), w: P(w), h: P(h), isTextBox: true, margin: 0, fontFace: F,
    fontSize: pt(o.size || FS.body), color: C.ink, align: o.align || "left",
    valign: o.valign || "top", bold: !!o.bold, italic: !!o.italic, fit: "none",
    lineSpacing: pt(o.ls || 8.4),
  });
}
const R = (text, o = {}) => ({ text, options: o });
const sub = (t, o = {}) => R(t, Object.assign({ subscript: true }, o));
const it = (t, o = {}) => R(t, Object.assign({ italic: true }, o));
const Fq = () => [it("F"), sub("q", { italic: true })];
const FdH = () => [it("F"), sub("ΔH", { italic: true })];

function rect(x, y, w, h, o = {}) {
  const opts = { x: P(x), y: P(y), w: P(w), h: P(h) };
  opts.fill = o.fill ? { color: o.fill } : { color: "FFFFFF", transparency: 100 };
  opts.line = o.line ? { color: o.line, width: pt(o.lw || 0.5) } : { color: "FFFFFF", width: 0, transparency: 100 };
  if (o.r) { opts.rectRadius = P(o.r); sl.addShape(pres.ShapeType.roundRect, opts); }
  else sl.addShape(pres.ShapeType.rect, opts);
}
function line(x1, y1, x2, y2, o = {}) {
  const opts = {
    x: P(Math.min(x1, x2)), y: P(Math.min(y1, y2)), w: P(Math.abs(x2 - x1)), h: P(Math.abs(y2 - y1)),
    flipH: x2 < x1, flipV: y2 < y1, line: { color: o.color || C.line, width: pt(o.lw || 0.6) },
  };
  if (o.head) opts.line.endArrowType = "triangle";
  if (o.dash) opts.line.dashType = o.dash;
  sl.addShape(pres.ShapeType.line, opts);
}
const arrow = (a, b, c, d, o = {}) => line(a, b, c, d, Object.assign({ head: true }, o));
function img(name, x, y, w, h) { sl.addImage({ path: path.join(A, name), x: P(x), y: P(y), w: P(w), h: P(h) }); }
function chart(name, x, y) {   // place a chart PNG at its drawn print size (600 dpi)
  const [w, h] = pngSize(name);
  img(name, x, y, w / 600, h / 600);
  return [w / 600, h / 600];
}
function panel(letter, x, y, title) {
  T(letter, x, y, 0.2, 0.16, { size: FS.letter, bold: true });
  T(title, x + 0.22, y + 0.012, 4.0, 0.14, { bold: true });
}
// token squares: g carries the material colour of the Gd-bearing rods
function tokens(x, y, syms, s = 0.12, gap = 0.012) {
  syms.forEach((t, i) => {
    const xi = x + i * (s + gap);
    if (t === "…") { T("…", xi, y - 0.005, s, s, { align: "center", valign: "middle" }); return; }
    const isG = t === "g";
    rect(xi, y, s, s, { fill: isG ? C.gd : C.white, line: isG ? C.gdE : C.line, lw: 0.5 });
    sl.addText(t, { x: P(xi), y: P(y), w: P(s), h: P(s), margin: 0, fontFace: F, fontSize: pt(FS.body),
      color: isG ? "FFFFFF" : C.ink, align: "center", valign: "middle", fit: "none" });
  });
  return x + syms.length * (s + gap) - gap;
}
function box(x, y, w, h, lines) {   // white card with a thin outline, text centred
  rect(x, y, w, h, { fill: C.white, line: C.ink, lw: 0.5, r: 0.045 });
  const runs = lines.map((l, i) => ({ text: l.text, options: Object.assign({}, l.options || {}, { breakLine: i < lines.length - 1 }) }));
  sl.addText(runs, { x: P(x), y: P(y), w: P(w), h: P(h), margin: 0, fontFace: F, fontSize: pt(FS.body), color: C.ink,
    align: "center", valign: "middle", fit: "none", lineSpacing: pt(8.6) });
}
function colorbar(name, x, y, w, h, lo, hi, label) {
  img(name, x, y, w, h);
  rect(x, y, w, h, { line: C.line, lw: 0.3 });
  T(lo, x - 0.30, y - 0.02, 0.27, h + 0.04, { align: "right", valign: "middle", size: FS.small });
  T(hi, x + w + 0.03, y - 0.02, 0.45, h + 0.04, { valign: "middle", size: FS.small });
  T(label, x - 0.40, y + h + 0.025, w + 0.80, 0.10, { align: "center", size: FS.small });
}

// shared column grid of panels a and b (print inches)
const COL = { corpus: 0.85, box: [2.10, 1.20], cand: 4.36, evalL: 5.07 };
const BOXC = COL.box[0] + COL.box[1] / 2;

// ====================================================================== header row
{
  const y = 0.24, yl = y + 0.055;
  T("training corpus", COL.corpus - 0.6, y, 1.2, 0.11, { align: "center" });
  T("policy", BOXC - 0.4, y, 0.8, 0.11, { align: "center" });
  T("candidate", COL.cand - 0.5, y, 1.0, 0.11, { align: "center" });
  T("scientific evaluator", 6.12 - 0.7, y, 1.4, 0.11, { align: "center" });
  arrow(COL.corpus + 0.42, yl, BOXC - 0.24, yl);
  arrow(BOXC + 0.24, yl, COL.cand - 0.30, yl);
  arrow(COL.cand + 0.30, yl, 6.12 - 0.56, yl);
}

// ====================================================================== PANEL a
{
  panel("a", 0.05, 0.05, "Reactor lattices");
  const yI = 0.42, hI = 1.36, yLab = yI + hI + 0.05;     // image row and first label line

  // training assembly (3D)
  const wIn = hI * pngAspect(IMG_IN), xIn = 0.12;
  img(IMG_IN, xIn, yI, wIn, hI);
  T("16 absorbers", 0.10, yLab, 0.95, 0.11);

  // corpus k_eff distribution against the requested target (chart)
  chart("chart_kstrip.png", 0.08, 1.98);

  // cross-section and serialization
  const xL = 1.10, yL = 0.60, sL = 0.86;
  T("17 × 17", xL + 0.25, 0.44, 0.5, 0.11);
  img("lattice_input.png", xL, yL, sL, sL);
  line(xIn + wIn * 0.62, yI + hI * 0.44, xL, yL, { dash: "dash", lw: 0.45 });
  line(xIn + wIn * 0.62, yI + hI * 0.44, xL, yL + sL, { dash: "dash", lw: 0.45 });
  arrow(xL + sL / 2, yL + sL + 0.02, xL + sL / 2, yL + sL + 0.12);
  const strip = ["f", "f", "c", "f", "g", "…", "g", "f", "f"], sw = strip.length * 0.132 - 0.012;
  const xs = xL + sL / 2 - sw / 2;
  tokens(xs, 1.60, strip);
  T("289 symbols", xs, 1.75, 1.2, 0.11);
  T("f fuel, g absorber, c guide", xs, 1.86, 1.3, 0.11);

  // sequence policy
  const [xB, wB] = COL.box, yB = 0.78, hB = 0.46, xc = BOXC;
  arrow(xL + sL + 0.04, 1.01, xB - 0.03, 1.01);
  box(xB, yB, wB, hB, [{ text: "Gemma 3, 270M", options: { bold: true } }, { text: "sequence policy" }]);
  T([R("target "), it("k"), R(" = 1.05")], xc - 0.6, 0.50, 1.2, 0.11, { align: "center" });
  arrow(xc, 0.63, xc, yB - 0.02);

  // generated text, corrected
  const xT = 3.50;
  arrow(xB + wB + 0.03, 1.01, xT - 0.03, 1.01);
  const xTe = tokens(xT, 0.95, ["f", "g", "…", "g"]);
  T("corrected", xT - 0.2, 1.10, 0.92, 0.11, { align: "center" });

  // generated assembly (3D), cut at mid-height
  const wOut = hI * pngAspect(IMG_OUT), xOut = 4.14;
  arrow(xTe + 0.03, 1.01, xOut - 0.02, 1.01);
  img(IMG_OUT, xOut, yI, wOut, hI);
  T("Generated lattice", xOut - 0.02, yLab, 1.0, 0.11);
  T(`${D.dpo_best.gd} absorbers`, xOut - 0.02, yLab + 0.11, 1.0, 0.11);

  // OpenMC fields: thermal flux and pin fission rate (chart)
  const xF = COL.evalL;
  arrow(xOut + wOut + 0.02, 1.01, xF - 0.02, 1.01);
  T("OpenMC", xF + 0.02, 0.40, 1.0, 0.11);
  const [wF] = chart("chart_fields.png", xF, 0.50);

  // feedback path: evaluator -> objective -> policy
  const xR = 7.17, yRet = 2.62;
  line(xF + wF - 0.01, 1.01, xR, 1.01);
  line(xR, 1.01, xR, yRet);
  line(xR, yRet, xc, yRet);
  arrow(xc, yRet, xc, yB + hB + 0.02);
  T([R("0.6 "), ...Fq(), R(" + 0.4 "), ...FdH(), R(" + 100 |"), it("k"), R(" − 1.05|")],
    3.30, yRet - 0.24, 2.30, 0.11, { align: "center" });
  T("DPO / GRPO update", 3.30, yRet - 0.13, 2.30, 0.11, { align: "center" });
}

// ====================================================================== PANEL b
{
  panel("b", 0.05, 2.76, "Reaction conditions");
  const yc = 3.40;                                   // baseline of the flow

  // supervised corpus: one anchor ligand per task (data)
  {
    const g = D.anchor_grid, cs = 0.055, gap = 0.008;
    const gw = g.n_tasks * (cs + gap) - gap, gh = g.n_ligands * (cs + gap) - gap;
    const x0 = COL.corpus - gw / 2, y0 = yc - gh / 2;
    for (let t = 0; t < g.n_tasks; t++) for (let l = 0; l < g.n_ligands; l++) {
      const on = g.anchor_index[t] === l;
      rect(x0 + t * (cs + gap), y0 + l * (cs + gap), cs, cs, { fill: on ? C.dark : C.white, line: on ? C.dark : C.line, lw: 0.3 });
    }
    T("Training corpus", x0, y0 - 0.14, 1.2, 0.11);
    T("15 tasks × 4 ligands", x0, y0 + gh + 0.04, 1.2, 0.11);
    T("anchor ligand per task", x0, y0 + gh + 0.15, 1.2, 0.11);
    arrow(x0 + gw + 0.06, yc, COL.box[0] - 0.03, yc);
  }

  const [xB, wB] = COL.box, yB = yc - 0.23, hB = 0.46, xc = BOXC;
  box(xB, yB, wB, hB, [{ text: "categorical policy", options: { bold: true } }, { text: "ligand → additive → base" }]);
  T([R("target yield "), it("u")], xc - 0.6, yB - 0.27, 1.2, 0.11, { align: "center" });
  arrow(xc, yB - 0.14, xc, yB - 0.02);

  // Buchwald-Hartwig coupling
  arrow(xB + wB + 0.03, yc, 3.47, yc);
  T("Ar–X", 3.50, yc - 0.06, 0.30, 0.12);
  T("+", 3.78, yc - 0.06, 0.10, 0.12, { align: "center" });
  const wTol = 0.60, hTol = wTol / pngAspect("mol_toluidine.png");
  img("mol_toluidine.png", 3.90, yc - hTol / 2, wTol, hTol);
  arrow(4.56, yc, 5.00, yc, { lw: 0.7 });
  T("Pd, ligand", 4.42, yc - 0.31, 0.72, 0.11, { align: "center" });
  T("base, additive", 4.42, yc - 0.20, 0.72, 0.11, { align: "center" });
  const wPr = 0.78, hPr = wPr / pngAspect("mol_product.png");
  img("mol_product.png", 5.10, yc - hPr / 2, wPr, hPr);

  // measured yield lookup (colour = measured yield)
  const xH = 5.98, wH = 22 * 0.053, hH = 4 * 0.053, yH = yc - hH / 2;
  arrow(5.90, yc, xH - 0.02, yc);
  T("4 ligands × 22 additives", xH - 0.02, yH - 0.15, 1.2, 0.11);
  img("yield_heatmap.png", xH, yH, wH, hH);
  rect(xH, yH, wH, hH, { line: C.line, lw: 0.3 });
  colorbar("cb_cividis.png", xH + 0.30, yH + hH + 0.07, 0.56, 0.055, "0", "100 %", "measured yield");
  T([R("reward  −|yield − "), it("u"), R("|")], xH - 0.05, 3.80, 1.25, 0.11, { align: "center" });

  // update path
  const yRet = 4.06, xRet = xH + wH / 2;
  line(xRet, 3.95, xRet, yRet);
  line(xRet, yRet, xc, yRet);
  arrow(xc, yRet, xc, yB + hB + 0.02);
  T("DPO / GRPO update", 3.3, yRet - 0.13, 2.3, 0.11, { align: "center" });
}

// ====================================================================== PANEL c
{
  panel("c", 0.05, 4.20, "Attribution on the inventory axis");
  chart("chart_attribution.png", 0.05, 4.28);
  chart("chart_right.png", 4.95, 4.28);
}

const args = process.argv.slice(2);
if (args.length !== 0 && !(args.length === 2 && args[0] === "--output")) {
  throw new Error("Usage: node make_fig1.js [--output /path/to/fig1_concept.pptx]");
}
const out = args.length ? path.resolve(args[1]) : path.join(HERE, "Figure1_NMI_v2.pptx");
fs.mkdirSync(path.dirname(out), { recursive: true });
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));
