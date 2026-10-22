// node world_receipt.mjs — the numbers quoted on world.html and in the README, from world.html's own core.
// 10 seeds per world; writes results/world/receipt.json (deterministic: rerunning regenerates it byte-for-byte).
import fs from "fs";
const html = fs.readFileSync(new URL("./world.html", import.meta.url), "utf8");
const core = html.split("// CORE-BEGIN")[1].split("\n").slice(1).join("\n").split("// CORE-END")[0];
const module = { exports: {} };
new Function("module", core)(module);
const WL = module.exports;

const N_TRAIN = 3000, SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rows = [];
for (const seed of SEEDS) {
  const row = { seed };
  for (const world of ["A", "B"]) {
    const B = WL.makeBrain({}, 1000 * seed + (world === "A" ? 1 : 2)), u = WL.rng(7000 * seed + (world === "A" ? 3 : 4));
    let found = null;
    for (let t = 1; t <= N_TRAIN; t++) {
      WL.learn(B, WL.render(WL.sampleStates(world, u), u));
      if (found === null && B.branches.some(b => b.stable && b.tiles.join() === "0,2,7")) found = t;
    }
    const kept = B.branches.find(b => b.stable && b.tiles.join() === "0,2,7");
    row[world] = {
      found_after: found, rewirings: B.rewired, other_kept: B.branches.filter(b => b.stable && b.tiles.join() !== "0,2,7").map(b => b.tiles.join("-")),
      triple: kept ? +WL.rule(kept).triple.toFixed(3) : null,
      learner: WL.evaluate(B, world, 400, 50 + seed), pairwise: WL.evaluate(B, world, 400, 50 + seed, "lin"),
    };
    WL.cutLinks(B, WL.rng(seed)); row[world].links_cut = WL.evaluate(B, world, 400, 50 + seed);
    WL.erase(B); row[world].erased = WL.evaluate(B, world, 400, 50 + seed);
  }
  const H = WL.makeBrain({ hide: [WL.MOUTH] }, 1000 * seed + 1), u = WL.rng(7000 * seed + 3);
  for (let t = 0; t < N_TRAIN; t++) WL.learn(H, WL.render(WL.sampleStates("A", u), u));
  row.mouth_never_seen = WL.evaluate(H, "A", 400, 50 + seed);
  rows.push(row);
  console.log(JSON.stringify(row));
}
const all = rows.flatMap(r => [r.A, r.B]), mean = a => +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(4);
const found = all.map(r => r.found_after).filter(v => v !== null).sort((a, b) => a - b);
const summary = {
  brains: all.length, found_wiring: found.length,
  found_after_median: found[Math.floor(found.length / 2)], found_after_range: [found[0], found[found.length - 1]],
  learner_accuracy_mean: mean(all.map(r => r.learner)), pairwise_mean: mean(all.map(r => r.pairwise)),
  links_cut_mean: mean(all.map(r => r.links_cut)), erased_mean: mean(all.map(r => r.erased)),
  mouth_never_seen_mean: mean(rows.map(r => r.mouth_never_seen)),
  triple_A: [...new Set(rows.map(r => r.A.triple))], triple_B: [...new Set(rows.map(r => r.B.triple))],
};
fs.mkdirSync(new URL("./results/world/", import.meta.url), { recursive: true });
fs.writeFileSync(new URL("./results/world/receipt.json", import.meta.url),
  JSON.stringify({ protocol: `${N_TRAIN} experiences per learner, 10 seeds per world, accuracy on 400 fresh scenes with the mouth covered (a grey answer counts as a coin flip)`, summary, rows }, null, 1) + "\n");
console.log("summary", JSON.stringify(summary));
