// electron-builder afterPack hook: put the right native WebRTC module (node-datachannel) into
// cross-platform builds. npm installs the binary for the machine you build on (a Mac), so a
// Windows build made on a Mac needs the Windows binary swapped in. Fetch it once with:
//   cd <copy of node_modules/node-datachannel> && prebuild-install -r napi --platform win32 --arch x64
// and save it as build/natives/win32-x64/node_datachannel.node
"use strict";
const fs = require("fs");
const path = require("path");

const ARCH = { 0: "ia32", 1: "x64", 2: "armv7l", 3: "arm64", 4: "universal" };

exports.default = async function afterPack(ctx) {
  const plat = ctx.electronPlatformName;
  if (plat === process.platform) return; // native build: npm already installed the right binary
  const arch = ARCH[ctx.arch] || String(ctx.arch);
  const src = path.join(__dirname, "natives", `${plat}-${arch}`, "node_datachannel.node");
  if (!fs.existsSync(src)) throw new Error(`missing ${src}: download the ${plat}-${arch} node-datachannel binary first`);
  const res = plat === "darwin" ? path.join(ctx.appOutDir, `${ctx.packager.appInfo.productFilename}.app`, "Contents", "Resources")
                                : path.join(ctx.appOutDir, "resources");
  const dst = path.join(res, "app.asar.unpacked", "node_modules", "node-datachannel", "build", "Release", "node_datachannel.node");
  fs.copyFileSync(src, dst);
  console.log(`  • afterPack: ${plat}-${arch} node-datachannel binary -> ${path.relative(ctx.appOutDir, dst)}`);
};
