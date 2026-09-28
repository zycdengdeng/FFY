import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const skillDir = process.env.SKILL_DIR;
if (!skillDir) throw new Error("SKILL_DIR is required");
const { importRuntimeModule } = await import(
  pathToFileURL(path.join(skillDir, "container_tools", "runtime_helpers.mjs")).href
);
const { FileBlob, PresentationFile } = await importRuntimeModule("@oai/artifact-tool");

const source = process.argv[2];
const output = process.argv[3];
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const snapshot = await presentation.inspect({
  kind: "deck,slide,textbox,shape,image,table,chart,notes,layout",
  include: "id,slide,name,title,text,textPreview,textChars,textLines,bbox,bboxUnit,alt,isPlaceholder,placeholders",
  maxChars: 60000,
});
await fs.writeFile(output, snapshot.ndjson, "utf8");

const info = {
  slideCount: presentation.slides.count,
  masters: presentation.masters?.items?.length ?? null,
};
console.log(JSON.stringify(info, null, 2));
