// node tests/test_index_core.mjs  — checks the page's core on a synthetic pan (no browser needed)
import fs from "fs";
import assert from "assert";
const html = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");
const core = html.split("// CORE-BEGIN")[1].split("\n").slice(1).join("\n").split("// CORE-END")[0];
const module = { exports: {} };
new Function("module", core)(module);
const LP = module.exports, { W, H, N } = LP;

// a smooth-but-textured RGB world, larger than the view
let seed = 3; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
function makeWorld(WW, HH) {
  const w = new Float32Array(WW * HH * 3);
  const blobs = Array.from({ length: 60 }, () => [rnd() * WW, rnd() * HH, 4 + rnd() * 18, rnd(), rnd(), rnd()]);
  for (let y = 0; y < HH; y++) for (let x = 0; x < WW; x++) {
    let v = [0.5 + 0.2 * Math.sin(x / 9), 0.5 + 0.2 * Math.cos(y / 7), 0.5];
    for (const [bx, by, r, a, b, c] of blobs) { const d = ((x - bx) ** 2 + (y - by) ** 2) / (r * r); if (d < 1) v = [a, b, c]; }
    w.set(v, 3 * (y * WW + x));
  }
  return w;
}
const WW = 260, HH = 180;
function view(world, x0, y0, noise = 0.02) {
  const o = new Float32Array(3 * N);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) for (let k = 0; k < 3; k++)
    o[3 * (r * W + c) + k] = world[3 * ((r + y0) * WW + c + x0) + k] + noise * (rnd() - .5) * 2;
  return o;
}
const mask = new Uint8Array(N).fill(1), hidden = [];
for (let r = 30; r < 64; r++) for (let c = 47; c < 81; c++) { mask[r * W + c] = 0; hidden.push(r * W + c); }
const prm = { rho: .985, sigma: .035, kappa: 12, check: true, slack: 3, R: 6 };

function run(world, path, P = LP.makePaper(), checkShift = true) {
  const xP = new Float32Array(3 * N).fill(.5), xA = new Float32Array(3 * N).fill(.5), w = new Float32Array(N), y = new Float32Array(N);
  let last;
  for (let t = 0; t < path.length; t++) {
    const [x0, y0] = path[t]; const o = view(world, x0, y0); LP.gray(o, y);
    LP.predict(P, o, y, mask, prm);
    if (checkShift && t > 0) assert.deepStrictEqual(P.lastShift, [path[t][1] - path[t - 1][1], path[t][0] - path[t - 1][0]]);
    for (let i = 0; i < N; i++) w[i] = 3 * P.piT[i] * P.trust[i];
    for (const i of hidden) if (w[i] > .5) for (let k = 0; k < 3; k++) xP[3 * i + k] = P.mT[3 * i + k];
    LP.fill(o, mask, xP, P.mT, w, 60, hidden); LP.fill(o, mask, xA, P.mT, null, 60, hidden);
    LP.update(P, o, y, mask, xP, 40);
    last = { o, xP: Float32Array.from(xP), xA: Float32Array.from(xA), x0, y0, P };
  }
  return last;
}
const mseHidden = (x, truth) => { let s = 0; for (const i of hidden) for (let k = 0; k < 3; k++) s += (x[3 * i + k] - truth[3 * i + k]) ** 2; return s / (3 * hidden.length); };

const world = makeWorld(WW, HH);
const path = Array.from({ length: 30 }, (_, t) => [20 + 3 * t, 30 + (t % 10 < 5 ? t % 5 : 5 - t % 5)]);
const r = run(world, path);
const truth = view(world, r.x0, r.y0, 0);
const ePaper = mseHidden(r.xP, truth), eAlone = mseHidden(r.xA, truth);
console.log("hidden MSE  paper", ePaper.toFixed(5), " this-frame-only", eAlone.toFixed(5));
assert(ePaper < eAlone / 5, "memory should beat the plain fill by a wide margin");

// foreign memory: carry a paper from another place, then show this place
const other = makeWorld(WW, HH);
const P2 = run(other, path.slice(0, 15)).P;
const r2 = run(world, [path[20]], P2, false);
let tH = 0; for (const i of hidden) tH += P2.trust[i];
const eF = mseHidden(r2.xP, view(world, path[20][0], path[20][1], 0)), eFA = mseHidden(r2.xA, view(world, path[20][0], path[20][1], 0));
console.log("foreign memory: mean trust in hole", (tH / hidden.length).toFixed(4), " paper", eF.toFixed(5), " plain", eFA.toFixed(5));
assert(tH / hidden.length < 0.05, "foreign memory must be rejected");
assert(Math.abs(eF - eFA) < 0.1 * eFA + 1e-4, "with memory rejected, the paper equals the plain fill");
console.log("ok");
