import path from "node:path";
import { pathToFileURL } from "node:url";
const skillDir = process.env.SKILL_DIR;
const { importRuntimeModule } = await import(pathToFileURL(path.join(skillDir, "container_tools", "runtime_helpers.mjs")).href);
const { Presentation } = await importRuntimeModule("@oai/artifact-tool");
const p = Presentation.create({slideSize:{width:1280,height:720}});
for (const q of process.argv.slice(2)) {
  const res = p.help("*", {search:q, include:["index","examples","notes"], maxChars:12000});
  console.log(`QUERY ${q}\n${res.ndjson}\n`);
}
