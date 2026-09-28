import path from "node:path";
import { pathToFileURL } from "node:url";
const skillDir = process.env.SKILL_DIR;
const { importRuntimeModule } = await import(pathToFileURL(path.join(skillDir, "container_tools", "runtime_helpers.mjs")).href);
const { FileBlob, PresentationFile } = await importRuntimeModule("@oai/artifact-tool");
const p = await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
for (const id of process.argv.slice(3)) {
  const o = p.resolve(id);
  const val = {id, frame:o.frame, geometry:o.geometry, fill:o.fill, line:o.line, textStyle:o.text?.style, paragraphStyle:o.text?.paragraphStyle, margin:o.text?.margin, verticalAlignment:o.text?.verticalAlignment};
  console.log(JSON.stringify(val, null, 2));
}
