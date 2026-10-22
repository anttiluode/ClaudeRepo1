// node tests/test_world_core.mjs — checks world.html's learning core (no browser needed, about a minute)
import fs from "fs";
import assert from "assert";
const html = fs.readFileSync(new URL("../world.html", import.meta.url), "utf8");
const core = html.split("// CORE-BEGIN")[1].split("\n").slice(1).join("\n").split("// CORE-END")[0];
const module = { exports: {} };
new Function("module", core)(module);
const WL = module.exports;

const live = (B, world, n, seed) => { const u = WL.rng(seed); for (let t = 0; t < n; t++) WL.learn(B, WL.render(WL.sampleStates(world, u), u)); return B; };
const kept = B => B.branches.filter(b => b.stable).map(b => b.tiles.join(","));

// 1. the worlds: every tile and every pair is identical in A and B, only the triple differs
{
  const u = WL.rng(1), m = { A: [], B: [] };
  for (const w of ["A", "B"]) for (let r = 0; r < 4000; r++) m[w].push(WL.sampleStates(w, u));
  const E = (w, f) => m[w].reduce((a, s) => a + f(s), 0) / m[w].length;
  for (let i = 0; i < 9; i++) for (let j = i + 1; j < 9; j++) assert(Math.abs(E("A", s => s[i] * s[j])) < 0.05 && Math.abs(E("B", s => s[i] * s[j])) < 0.05, `pair ${i},${j}`);
  assert(E("A", s => s[0] * s[2] * s[7]) === 1 && E("B", s => s[0] * s[2] * s[7]) === -1);
}

// 2. each learner finds the eye-eye-mouth wiring by itself and completes the covered mouth
for (const seed of [1, 2, 3]) {
  const A = live(WL.makeBrain({}, seed), "A", 2500, 100 + seed), B = live(WL.makeBrain({}, 50 + seed), "B", 2500, 200 + seed);
  assert(kept(A).includes("0,2,7") && kept(B).includes("0,2,7"), `seed ${seed}: wiring not found (${kept(A)} / ${kept(B)})`);
  const rA = WL.rule(A.branches.find(b => b.stable && b.tiles.join() === "0,2,7"));
  const rB = WL.rule(B.branches.find(b => b.stable && b.tiles.join() === "0,2,7"));
  assert(rA.triple > 0.9 && rB.triple < -0.9, "opposite three-way rules");
  for (const r of [rA, rB]) for (const p of r.pairs) assert(Math.abs(p[2]) < 0.1, "pairs say nothing");
  const accA = WL.evaluate(A, "A", 200, 9), accB = WL.evaluate(B, "B", 200, 9);
  const linA = WL.evaluate(A, "A", 200, 9, "lin"), linB = WL.evaluate(B, "B", 200, 9, "lin");
  console.log(`seed ${seed}: learner A ${accA}, learner B ${accB}, pairwise ${linA} / ${linB}`);
  assert(accA >= 0.95 && accB >= 0.95, "completion");
  assert(Math.abs(linA - 0.5) < 0.1 && Math.abs(linB - 0.5) < 0.1, "pairwise control at chance");

  // same partial input, different experience, different answer
  const x = WL.render([-1, 1, -1, 1, -1, 1, 1, 1, -1], WL.rng(5));
  assert(WL.infer(A, x, [WL.MOUTH]).tiles[WL.MOUTH].pPlus > 0.9 && WL.infer(B, x, [WL.MOUTH]).tiles[WL.MOUTH].pPlus < 0.1);

  // a peek that contradicts learner B takes its say away; one that agrees with A keeps it
  const rev = new Uint8Array(WL.N); const ur = WL.rng(9); for (const i of WL.PIX[WL.MOUTH]) if (ur() < 0.3) rev[i] = 1;
  assert(WL.infer(B, x, [WL.MOUTH], rev).tiles[WL.MOUTH].belief < 0.3, "contradiction lowers belief");
  assert(WL.infer(A, x, [WL.MOUTH], rev).tiles[WL.MOUTH].belief > 0.9, "agreement keeps belief");

  // an eye covered as well: undecided until the mouth is peeked, then the hidden eye follows the learner's rule
  const r0 = WL.infer(A, x, [2, 7]).tiles[2], r1 = WL.infer(A, x, [2, 7], rev).tiles[2], s1 = WL.infer(B, x, [2, 7], rev).tiles[2];
  assert(r0.belief < 0.5 && r1.belief > r0.belief, "peek raises belief");
  assert(r1.pPlus < 0.5 && s1.pPlus > 0.5, "round eye for A, flat eye for B, from the same peek");

  // a scene unlike experience: no say
  const y = Float32Array.from(x); const un = WL.rng(4); for (const i of WL.PIX[0]) y[i] = 0.5 + 0.4 * un.normal();
  assert(WL.infer(A, y, [WL.MOUTH]).tiles[WL.MOUTH].support < 0.05, "garbage gets no support");

  // causal checks: cut the links (pieces kept, re-paired) and erase
  WL.cutLinks(A, WL.rng(7)); const cut = WL.evaluate(A, "A", 200, 9);
  WL.erase(A); const er = WL.evaluate(A, "A", 200, 9);
  console.log(`        links cut ${cut}, erased ${er}`);
  assert(cut <= 0.75 && er === 0.5, "ability lives in the learned links");
}

// 3. a learner that never sees mouths cannot learn them; the pairwise control is not broken
{
  const H = WL.makeBrain({ hide: [WL.MOUTH] }, 7); live(H, "A", 2500, 77);
  assert(Math.abs(WL.evaluate(H, "A", 200, 9) - 0.5) < 0.05, "never-seen mouth");
  const C = live(WL.makeBrain({}, 8), "copy", 2000, 88);
  const lc = WL.evaluate(C, "copy", 200, 9, "lin"); console.log("pairwise control in a pairwise world (mouth copies the left eye):", lc);
  assert(lc > 0.9, "the control learns pairwise structure");
}
console.log("world core: all checks passed");
