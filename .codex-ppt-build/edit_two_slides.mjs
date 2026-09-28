import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { pathToFileURL } from "node:url";

const { SKILL_DIR, TMP_DIR, FINAL_PPTX, RUNTIME_PYTHON } = process.env;
if (![SKILL_DIR, TMP_DIR, FINAL_PPTX, RUNTIME_PYTHON].every((v) => v && path.isAbsolute(v))) {
  throw new Error("SKILL_DIR, TMP_DIR, FINAL_PPTX and RUNTIME_PYTHON must be absolute paths");
}

const workspaceDir = "C:\\Users\\ZihanWANG\\Desktop\\FFY\\FFY";
const sourcePath = "C:\\Users\\ZihanWANG\\Desktop\\FFY\\机理约束生成式设计_两页.pptx";
const { importRuntimeModule } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools", "runtime_helpers.mjs")).href
);
const { FileBlob, PresentationFile } = await importRuntimeModule("@oai/artifact-tool");
const { finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href
);

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const slide1 = presentation.resolve("sl/y90nupkv");
const slide2 = presentation.resolve("sl/hwbqtkby");

// Slide 1: keep the user's composition and tighten the two captions most relevant to FFY and inverse design.
presentation.resolve("sh/xsfy5grm").text = "生物观测、仿真与试验融合数据集";
presentation.resolve("sh/4ne5g3yh").text = "目标性能条件生成与约束筛选";
slide1.speakerNotes.textFrame.setText(
  "本页说明研究问题与总体技术路线。左侧概括真实试验昂贵、仿真预算有限、高维耦合、强物理约束和多目标权衡五项共性难题。右侧四项研究形成闭环：a 融合生物观测、有限元仿真与力学试验数据；b 以机理模型为基础、由数据学习残差，在有限预算下预测关键性能；c 给定轻量化、强度与寿命目标生成候选设计，并用知识图谱、物理约束和制造规则筛选；d 制造缩比原型并开展力学实验，将实测偏差回灌数据集与模型。下一页进一步说明观察数据和受控干预分别在闭环中承担什么作用。"
);

// Slide 2: rebuild the content below the retained title to make the causal role operational and FFY-specific.
const slide2Ids = [
  "sh/ozy1ofad","sh/b29kza94","sh/a10jqpsj","sh/x4r21kru","sh/w3i1sfa9","sh/r65knqtk",
  "sh/q5wjelsz","sh/ove9o7yd","sh/9wnqhczy","sh/mtwrmxg7","sh/nu58f2hs","sh/wjy9sry9",
  "sh/xk7qlczu","sh/ahwrqhgj","sh/bip8jmho","sh/cnupgny5","sh/do3q9szq","sh/t8byxkn2",
  "sh/s72xofmh","sh/76twva5c","sh/65kfmpor","sh/hcvy14ne","sh/gbmxszmt","sh/fadgzu58",
  "sh/u94fqp4n","sh/p0vy54na","sh/4zmxwzmp","sh/mlgvq147","sh/nmpcj6ls","sh/onydsbmd",
  "sh/9o7ulg3y","sh/a9gvulkj","sh/ba9cn6l4","sh/wbidwb29","sh/xcrupg3u","sh/id0vylkf",
  "sh/je9crql0","sh/ru5cbqd4","sh/69wbilcj","sh/tw7ud0va","sh/svedkvup","sh/fypcfqdg",
  "sh/exgvmlcv","sh/107uh0v6"
];
for (const id of slide2Ids) presentation.resolve(id).delete();
presentation.resolve("sh/9072xkry").text = "因果干预驱动的有限数据设计闭环";

const FONT = "微软雅黑";
const C = {
  burgundy: "#8E3034",
  burgundyDark: "#762529",
  burgundyLight: "#F5E9EA",
  navy: "#42546E",
  navyLight: "#EAF0F6",
  ink: "#282828",
  gray: "#6C6C6C",
  line: "#D7D7D7",
  pale: "#F4F4F4",
  white: "#FFFFFF",
};

function addText(position, text, style = {}) {
  const shape = slide2.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  if (Array.isArray(text)) shape.text.set(text);
  else shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: style.fontSize ?? 22,
    bold: style.bold ?? false,
    color: style.color ?? C.ink,
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "top",
    autoFit: style.autoFit ?? "shrinkText",
    wrap: "square",
    insets: style.insets ?? { top: 3, right: 4, bottom: 3, left: 4 },
  };
  return shape;
}

function addCard(x, letter, title, paragraphs, tone) {
  const fill = tone === "blue" ? C.navyLight : tone === "red" ? C.burgundyLight : C.pale;
  const accent = tone === "blue" ? C.navy : C.burgundy;
  const card = slide2.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: 154, width: 280, height: 260 },
    fill,
    line: { style: "solid", fill: C.line, width: 1.2 },
    borderRadius: 14,
  });
  const badge = slide2.shapes.add({
    geometry: "ellipse",
    position: { left: x + 18, top: 171, width: 40, height: 40 },
    fill: accent,
    line: { fill: "none", width: 0 },
  });
  badge.text = letter;
  badge.text.style = {
    typeface: FONT, fontSize: 24, bold: true, color: C.white,
    alignment: "center", verticalAlignment: "middle", autoFit: "none",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
  addText(
    { left: x + 68, top: 171, width: 194, height: 42 },
    title,
    { fontSize: 25, bold: true, color: C.ink, verticalAlignment: "middle" },
  );
  const body = addText(
    { left: x + 18, top: 222, width: 244, height: 170 },
    paragraphs,
    { fontSize: 19, color: C.ink, insets: { top: 1, right: 2, bottom: 1, left: 2 } },
  );
  return { card, badge, body };
}

addText(
  { left: 48, top: 92, width: 1184, height: 43 },
  [
    [
      { run: "观察数据", textStyle: { bold: true, color: C.navy } },
      "提供设计先验，",
      { run: "受控干预", textStyle: { bold: true, color: C.burgundy } },
      "识别可控效应，原型试验校准真实偏差",
    ],
  ],
  { fontSize: 25, color: C.ink, verticalAlignment: "middle" },
);

const cardA = addCard(50, "a", "数据融合", [
  [{ run: "观察数据  ", textStyle: { bold: true, color: C.navy } }],
  ["水鸟形态、着陆视频与历史记录"],
  [{ run: "得到  ", textStyle: { bold: true } }, "相关关系  P(Y|X)"],
  [{ run: "干预数据  ", textStyle: { bold: true, color: C.burgundy } }],
  ["有限元扫描与受控缩比试验"],
  [{ run: "得到  ", textStyle: { bold: true } }, "因果效应  P(Y|do(X))"],
], "blue");

const cardB = addCard(350, "b", "因果代理模型", [
  [{ run: "结构  ", textStyle: { bold: true } }, "设计变量 X、机理状态 M、性能响应 Y 分层建模"],
  [{ run: "工况  ", textStyle: { bold: true } }, "冲击、静载与载荷谱作为条件输入"],
  [{ run: "采样  ", textStyle: { bold: true, color: C.burgundy } }, "按预测不确定性与干预价值选择下一次高保真仿真"],
], "neutral");

const cardC = addCard(650, "c", "条件逆向生成", [
  [{ run: "输入  ", textStyle: { bold: true } }, "目标重量、强度与疲劳寿命"],
  [{ run: "输出  ", textStyle: { bold: true } }, "几何、壁厚、材料、刚度与阻尼候选"],
  [{ run: "筛选  ", textStyle: { bold: true, color: C.burgundy } }, "知识图谱规则与科学神经网络嵌入物理、制造约束"],
], "red");

const cardD = addCard(950, "d", "原型验证与回灌", [
  [{ run: "控制变量  ", textStyle: { bold: true } }, "制造关键参数不同的缩比原型"],
  [{ run: "检验  ", textStyle: { bold: true } }, "代理模型预测的性能增量"],
  [{ run: "回灌  ", textStyle: { bold: true, color: C.burgundy } }, "校准仿真偏差与模型不确定性"],
  [{ run: "迁移  ", textStyle: { bold: true } }, "依据相似律验证全尺寸设计"],
], "neutral");

for (const x of [326, 626, 926]) {
  slide2.shapes.add({
    geometry: "rightArrow",
    position: { left: x, top: 274, width: 28, height: 26 },
    fill: C.burgundy,
    line: { fill: "none", width: 0 },
  });
}

const feedback = slide2.shapes.add({
  geometry: "leftArrow",
  position: { left: 116, top: 423, width: 1048, height: 24 },
  fill: C.burgundyLight,
  line: { fill: "none", width: 0 },
});
feedback.text = "验证偏差回灌数据集与代理模型";
feedback.text.style = {
  typeface: FONT, fontSize: 17, bold: true, color: C.burgundy,
  alignment: "center", verticalAlignment: "middle", autoFit: "none",
  insets: { top: 0, right: 0, bottom: 0, left: 0 },
};

addText(
  { left: 50, top: 465, width: 1180, height: 35 },
  "FFY 中的具体落点",
  { fontSize: 26, bold: true, color: C.ink, verticalAlignment: "middle" },
);

const ffys = [
  {
    x: 50, n: "1", title: "观察规律",
    body: "水鸟腿长、比例与体重和着陆姿态的关联\n用于限定仿生先验与搜索范围",
    color: C.navy,
  },
  {
    x: 440, n: "2", title: "受控干预",
    body: "固定质量、速度与地面条件\n主动改变腿长比、刚度和阻尼",
    color: C.burgundy,
  },
  {
    x: 830, n: "3", title: "物理判官",
    body: "Exudyn、有限元与缩比原型比较\n峰值过载、行程、质量与寿命",
    color: C.burgundyDark,
  },
];

for (const item of ffys) {
  const num = slide2.shapes.add({
    geometry: "ellipse",
    position: { left: item.x, top: 509, width: 32, height: 32 },
    fill: item.color,
    line: { fill: "none", width: 0 },
  });
  num.text = item.n;
  num.text.style = {
    typeface: FONT, fontSize: 19, bold: true, color: C.white,
    alignment: "center", verticalAlignment: "middle", autoFit: "none",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
  addText({ left: item.x + 42, top: 505, width: 300, height: 39 }, item.title,
    { fontSize: 23, bold: true, color: item.color, verticalAlignment: "middle" });
  addText({ left: item.x, top: 547, width: 350, height: 68 }, item.body,
    { fontSize: 19, color: C.ink, verticalAlignment: "top" });
}

for (const x of [420, 810]) {
  slide2.shapes.add({
    geometry: "line",
    position: { left: x, top: 508, width: 0, height: 98 },
    fill: "none",
    line: { style: "solid", fill: C.line, width: 1.2 },
  });
}

const takeaway = slide2.shapes.add({
  geometry: "roundRect",
  position: { left: 45, top: 630, width: 1190, height: 52 },
  fill: C.burgundy,
  line: { fill: "none", width: 0 },
  borderRadius: 8,
});
takeaway.text.set([[
  { run: "生物数据", textStyle: { bold: true } }, "缩小搜索空间，",
  { run: "受控仿真", textStyle: { bold: true } }, "识别工程因果，",
  { run: "物理试验", textStyle: { bold: true } }, "完成真实验证",
]]);
takeaway.text.style = {
  typeface: FONT, fontSize: 23, color: C.white,
  alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText",
  insets: { top: 2, right: 10, bottom: 2, left: 10 },
};

slide2.speakerNotes.textFrame.setText(
  "本页回答数据和因果干预如何进入设计闭环。观察数据包括水鸟形态、着陆视频以及历史运行和试验记录，它们用于学习设计先验和相关规律，但不能单独证明改变某个参数会改善性能。有限元参数扫描和受控缩比实验主动设定设计变量，在工况保持一致时比较性能变化，因此提供模型内或实验条件下的干预证据。代理模型按照设计变量、机理状态和性能响应分层建模，并综合预测不确定性与干预价值安排下一次高保真仿真。生成模型给定目标性能提出候选，再由知识图谱、物理约束和制造规则筛选。FFY 中的具体例子是：水鸟数据提示腿长和比例的先验范围；在固定质量、着陆速度和地面条件时，主动改变腿长比、刚度和阻尼；最后由 Exudyn、有限元和缩比原型比较峰值过载、行程、质量与寿命。需要强调，仿真干预识别的是模型内因果，原型试验负责检验它能否迁移到真实结构。"
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
const stagingDir = path.join(workspaceDir, ".codex-finalizer-two-slides");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const sourceBytes = await fs.readFile(sourcePath);
const sourceSha256 = crypto.createHash("sha256").update(sourceBytes).digest("hex");
const result = await finalizePresentation({
  explicitTotalSlideCount: 2,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  fontPolicy: {
    basis: "reference",
    families: ["微软雅黑", "Arial"],
    referencePath: sourcePath,
    referenceSha256: sourceSha256,
  },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${path.basename(FINAL_PPTX)}.validation.json`),
});

console.log(JSON.stringify({ finalPath: FINAL_PPTX, result }, null, 2));
