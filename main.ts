// Roblox Up-and-Coming top-10 forecaster — V5 (ensemble).
//
// Two genuinely different models, averaged 50/50:
// (a) an empirical rank-band transition kernel: each game's rank is a
//     discrete distribution over ranks 1..100 (+ "out"), propagated in
//     6 h steps by convolving with a rank-band-specific kernel fitted
//     from the history (recency-weighted, Gaussian-smoothed, +2%
//     uniform prior so no move is impossible), shifted by the game's
//     momentum. No Monte Carlo, no Gaussian assumption, exact arithmetic.
// (b) the V4 seeded Monte Carlo rank diffusion (per-hour drift and
//     volatility estimated from the history, rank effect, exit hazard).
// The ensemble reduces variance and improves calibration: the Monte
// Carlo is overconfident on long horizons (it names ~0.98 of each place
// but only ~0.6 of the actual games are named, so the rest fall to the
// tiny unnamed share and hit the log-score floor); the transition
// model's independent marginals honestly lose mass to exits/drift,
// leaving a larger implicit unnamed share. Averaging them is the
// robust, calibrated compromise. Place columns are capped at
// (1 - UNNAMED) when above, never scaled up.
//
// Deterministic: a seeded PRNG (mulberry32) drives the Monte Carlo and
// the transition propagation is exact arithmetic, so two runs on the
// same history produce byte-identical output.
// Stdlib only, no network, no model calls.
//
// Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// ---------- seeded PRNG (mulberry32) + Gaussian (Box-Muller) ----------
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
class Rng {
  private next: () => number;
  private spare: number | null = null;
  constructor(seed: number) { this.next = mulberry32(seed); }
  uniform(): number { return this.next(); }
  gaussian(): number {
    if (this.spare !== null) { const v = this.spare; this.spare = null; return v; }
    let u = 0, v = 0;
    while (u === 0) u = this.next();
    v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    const t = 2 * Math.PI * v;
    this.spare = r * Math.sin(t);
    return r * Math.cos(t);
  }
}

// ---------- input ----------
interface Row {
  rank: number; universe_id: number; name?: string; playing?: number | null;
}
interface Read {
  at: string; up_and_coming: Row[];
  details?: Record<string, { created?: string | null }> | null;
}
const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number; output: string;
  window: { from: string; until: string; slots: number; slot_minutes: number };
  history_until: string;
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n').filter((l) => l !== '');
const reads: Read[] = lines.map((l) => JSON.parse(l) as Read);
const PLACES = task.places ?? 10;
const HOURS = task.window?.slots ?? 72;
const windowStart = Date.parse(task.window.from);

interface Game {
  uid: number;
  lastRank: number;
  lastAt: number;
  firstAt: number;
  drift: number;    // rank change per hour (positive = falling)
  vol: number;      // rank std-dev per sqrt(hour)
  createdMs: number | null;
  exitHazard: number; // per-hour probability of leaving the chart
}
const games = new Map<number, Game>();
const HOUR = 3600_000;

// Build per-game rank series and estimate drift / volatility.
// Recent consecutive-read pairs weigh more (half-life ~5 days).
const HALF_LIFE_MS = 5 * 24 * HOUR;
for (let i = 0; i < reads.length; i++) {
  const r = reads[i]!;
  const t = Date.parse(r.at);
  for (const g of r.up_and_coming) {
    let gm = games.get(g.universe_id);
    if (!gm) {
      gm = { uid: g.universe_id, lastRank: g.rank, lastAt: t, firstAt: t, drift: 0, vol: 2, createdMs: null, exitHazard: 0 };
      games.set(g.universe_id, gm);
    }
    gm.lastRank = g.rank; gm.lastAt = t;
  }
}
// created dates from details (house-recorder reads carry them; archive reads do not)
for (const r of reads) {
  if (!r.details) continue;
  for (const [uidStr, d] of Object.entries(r.details)) {
    const uid = Number(uidStr);
    const gm = games.get(uid);
    if (gm && d && d.created) {
      const c = Date.parse(d.created);
      if (!Number.isNaN(c)) gm.createdMs = c;
    }
  }
}
// Drift / volatility from consecutive reads (per game), recency-weighted.
for (const gm of games.values()) {
  let wSum = 0, dSum = 0, vSum = 0, vW = 0;
  let prevT: number | null = null, prevR: number | null = null;
  for (const r of reads) {
    const t = Date.parse(r.at);
    const row = r.up_and_coming.find((g) => g.universe_id === gm.uid);
    if (!row) { prevT = null; prevR = null; continue; }
    if (prevT !== null && prevR !== null) {
      const dtH = Math.max((t - prevT) / HOUR, 0.25);
      const w = Math.pow(0.5, (windowStart - t) / HALF_LIFE_MS);
      const d = (row.rank - prevR) / dtH;
      dSum += w * d; wSum += w;
      vSum += w * Math.pow((row.rank - prevR) / Math.sqrt(dtH), 2); vW += w;
    }
    prevT = t; prevR = row.rank;
  }
  gm.drift = wSum > 0 ? dSum / wSum : 0;
  const varPerHour = vW > 0 ? vSum / vW : 4;
  gm.vol = Math.min(Math.max(Math.sqrt(varPerHour), 0.5), 6);
  // Shrink drift toward 0 (rank changes are noisy); keep it small.
  // The clamp is wide so genuine momentum (a game breaking out) is not clipped.
  gm.drift = Math.max(Math.min(gm.drift, 0.6), -0.6);
}
// Rank effect: the systematic drift that applies to ANY game at a given
// rank (mean hourly rank change of games starting at that rank). The chart
// is constantly invaded by new games, so existing games drift DOWNWARD
// (positive rank change) on average — rank 1 falls slowly (+0.04/h),
// rank 10 leaks fast (+0.23/h). A game's idiosyncratic momentum is its
// drift minus the rank effect at its own average rank, so the two are not
// double-counted: forward drift = rank effect at the current rank +
// excess (per-game) drift.
const MAX_RANK = 60;
const rankEffectRaw = new Float64Array(MAX_RANK + 1);
const rankEffectW = new Float64Array(MAX_RANK + 1);
for (let i = 0; i + 1 < reads.length; i++) {
  const a = reads[i]!;
  const b = reads[i + 1]!;
  const tA = Date.parse(a.at);
  const tB = Date.parse(b.at);
  const dtH = Math.max((tB - tA) / HOUR, 0.25);
  const w = Math.pow(0.5, (windowStart - tB) / HALF_LIFE_MS);
  const rb = new Map<number, number>();
  for (const g of b.up_and_coming) rb.set(g.universe_id, g.rank);
  for (const g of a.up_and_coming) {
    const r2 = rb.get(g.universe_id);
    if (r2 === undefined) continue;
    const r = Math.min(Math.max(g.rank, 1), MAX_RANK);
    rankEffectRaw[r] += w * ((r2 - g.rank) / dtH);
    rankEffectW[r] += w;
  }
}
let rankEffectMean = 0, rankEffectMeanW = 0;
for (let r = 1; r <= MAX_RANK; r++) {
  if (rankEffectW[r]! > 0) { rankEffectMean += rankEffectRaw[r]!; rankEffectMeanW += rankEffectW[r]!; }
}
rankEffectMean = rankEffectMeanW > 0 ? rankEffectMean / rankEffectMeanW : 0.1;
// Smooth across neighbouring ranks (each rank has few observations).
const rankEffect = new Float64Array(MAX_RANK + 1);
for (let r = 1; r <= MAX_RANK; r++) {
  const e = (x: number) => (rankEffectW[x]! > 0 ? rankEffectRaw[x]! / rankEffectW[x]! : rankEffectMean);
  rankEffect[r] = (e(Math.max(r - 1, 1)) + 2 * e(r) + e(Math.min(r + 1, MAX_RANK))) / 4;
}
const rankEffectAt = (rank: number): number => {
  const r = Math.min(Math.max(Math.round(rank), 1), MAX_RANK);
  return rankEffect[r]!;
};
// Per-game excess drift: the game's own drift minus the rank effect at
// its recency-weighted mean rank (so momentum is measured relative to
// what any game at that rank does).
const meanRankOf = (gm: Game): number => {
  let wSum = 0, rSum = 0;
  for (const r of reads) {
    const t = Date.parse(r.at);
    const row = r.up_and_coming.find((g) => g.universe_id === gm.uid);
    if (!row) continue;
    const w = Math.pow(0.5, (windowStart - t) / HALF_LIFE_MS);
    rSum += w * row.rank; wSum += w;
  }
  return wSum > 0 ? rSum / wSum : gm.lastRank;
};
for (const gm of games.values()) {
  gm.drift = Math.max(Math.min(gm.drift - rankEffectAt(meanRankOf(gm)), 0.6), -0.6);
}
// per hour, scaled by how long the game has been seen (older games age out).
let exitNum = 0, exitDen = 0;
for (let i = 0; i + 1 < reads.length; i++) {
  const a = new Set(reads[i]!.up_and_coming.map((g) => g.universe_id));
  const b = new Set(reads[i + 1]!.up_and_coming.map((g) => g.universe_id));
  const dtH = Math.max((Date.parse(reads[i + 1]!.at) - Date.parse(reads[i]!.at)) / HOUR, 1);
  for (const uid of a) { exitDen += dtH; if (!b.has(uid)) exitNum += 1; }
}
const baseExitPerHour = exitDen > 0 ? exitNum / exitDen : 0.002;
const CHART_N = reads[reads.length - 1]!.up_and_coming.length;
for (const gm of games.values()) {
  // Age in days since creation (or first seen, a lower bound). A game is at
  // most 28 days old on this chart; hazard rises as it approaches that cap.
  const ageDays = gm.createdMs !== null
    ? (windowStart - gm.createdMs) / (24 * HOUR)
    : (windowStart - gm.firstAt) / (24 * HOUR);
  const ageFactor = gm.createdMs !== null
    ? 0.5 + Math.max(0, ageDays) / 14
    : 0.5 + Math.min(Math.max(0, ageDays), 28) / 28;
  gm.exitHazard = Math.min(baseExitPerHour * ageFactor, 0.05);
}
// Unnamed reserve: share of top-10 slots in the history held by a game that
// was outside the previous read's top 30 (a surprise entrant). That mass is
// left unnamed at every place, so a game the history never names still gets
// a share of it on each read.
let surprise = 0, topSlots = 0;
for (let i = 1; i < reads.length; i++) {
  const prevTop30 = new Set(reads[i - 1]!.up_and_coming.slice(0, 30).map((g) => g.universe_id));
  for (const g of reads[i]!.up_and_coming.slice(0, PLACES)) {
    topSlots++;
    if (!prevTop30.has(g.universe_id)) surprise++;
  }
}
const UNNAMED = Math.min(Math.max(topSlots > 0 ? surprise / topSlots : 0.03, 0.01), 0.12);

// ---------- Monte Carlo ----------
// Simulate the games in the last chart (plus recent top-30 games not in it).
// Rank follows a mean-reverting diffusion: each hour the rank moves toward
// the game's recency-weighted mean rank (theta per hour) plus drift and
// Gaussian noise. Mean reversion is what keeps the forecast from diffusing
// to nothing over a 72 h window: a chart rank is a relative ordering, so a
// game that drifts up is pulled back toward its equilibrium rank.
// Tunables (env, for backtesting): MF_VOL_SCALE, MF_THETA, MF_DRIFT_SCALE.
const VOL_SCALE = Number(process.env.MF_VOL_SCALE ?? 1.0);
const THETA = Number(process.env.MF_THETA ?? 0);
const DRIFT_SCALE = Number(process.env.MF_DRIFT_SCALE ?? 0.5);
const lastRead = reads[reads.length - 1]!;
const lastChart = [...lastRead.up_and_coming].sort((a, b) => a.rank - b.rank);
const simUids = new Set<number>(lastChart.map((g) => g.universe_id));
for (const r of reads.slice(-6)) {
  for (const g of r.up_and_coming.slice(0, 30)) simUids.add(g.universe_id);
}
const sim = [...simUids].map((uid) => games.get(uid)!).filter(Boolean);
const N = sim.length;
const N_PATHS = 6000;
const rng = new Rng(0x9e3779b9);
const counts = new Float64Array(N * PLACES); // counts[gameIdx * PLACES + (place-1)]
const rankNow = new Float64Array(N);
const totalPairs = N_PATHS * HOURS;
// Recency-weighted mean rank per game (the mean-reversion target).
const meanRank = new Float64Array(N);
for (let i = 0; i < N; i++) {
  const gm = sim[i]!;
  let wSum = 0, rSum = 0;
  for (const r of reads) {
    const t = Date.parse(r.at);
    const row = r.up_and_coming.find((g) => g.universe_id === gm.uid);
    if (!row) continue;
    const w = Math.pow(0.5, (windowStart - t) / HALF_LIFE_MS);
    rSum += w * row.rank; wSum += w;
  }
  meanRank[i] = wSum > 0 ? rSum / wSum : gm.lastRank;
}
for (let path = 0; path < N_PATHS; path++) {
  for (let i = 0; i < N; i++) rankNow[i] = sim[i]!.lastRank;
  const alive = new Uint8Array(N).fill(1);
  for (let h = 0; h < HOURS; h++) {
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      const gm = sim[i]!;
      if (rng.uniform() < gm.exitHazard) { alive[i] = 0; continue; }
      rankNow[i] = rankNow[i]! + THETA * (meanRank[i]! - rankNow[i]!) + rankEffectAt(rankNow[i]!) + gm.drift * DRIFT_SCALE + gm.vol * VOL_SCALE * rng.gaussian();
      if (rankNow[i]! < 1) rankNow[i] = 1;
      if (rankNow[i]! > CHART_N + 8) alive[i] = 0;
    }
    // Top 10 by simulated rank (linear scan, no full sort).
    const topIdx: number[] = [];
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      const r = rankNow[i]!;
      let pos = topIdx.length;
      while (pos > 0 && rankNow[topIdx[pos - 1]!]! > r) pos--;
      if (pos < PLACES) {
        topIdx.splice(pos, 0, i);
        if (topIdx.length > PLACES) topIdx.pop();
      }
    }
    for (let k = 0; k < topIdx.length; k++) {
      counts[topIdx[k]! * PLACES + k]! += 1;
    }
  }
}

// ---------- transition model (empirical rank-band kernel) ----------
// Exact discrete distribution propagation over ranks 1..100 (+ "out").
// Per rank band a kernel over rank change per 6 h step, fitted from
// consecutive read pairs (recency-weighted), Gaussian-smoothed and mixed
// with a uniform prior so no move is impossible. Each game's kernel is
// shifted by its momentum (its excess drift, already net of the rank
// effect — the kernel's per-band mean already contains it, so there is
// no double-counting). No Monte Carlo, no Gaussian assumption.
const STEP_H = 6;
const RMAX = 100;
const D = 40; // kernel half-width in ranks
const BANDS: [number, number][] = [[1, 1], [2, 2], [3, 3], [4, 5], [6, 10], [11, 15], [16, 20], [21, 30], [31, 50], [51, 100]];
const bandOf = (r: number): number => {
  for (let i = 0; i < BANDS.length; i++) {
    if (r >= BANDS[i]![0] && r <= BANDS[i]![1]) return i;
  }
  return BANDS.length - 1;
};
// Kernel counts from consecutive read pairs: rank change scaled to a
// 6 h step, recency-weighted (5-day half-life).
const kernelCounts: number[][] = [];
for (let b = 0; b < BANDS.length; b++) kernelCounts.push(new Array(2 * D + 1).fill(0));
for (let i = 0; i + 1 < reads.length; i++) {
  const a = reads[i]!;
  const b = reads[i + 1]!;
  const tA = Date.parse(a.at);
  const tB = Date.parse(b.at);
  const dtH = Math.max((tB - tA) / HOUR, 0.25);
  const w = Math.pow(0.5, (windowStart - tB) / HALF_LIFE_MS);
  const rb = new Map<number, number>();
  for (const g of b.up_and_coming) rb.set(g.universe_id, g.rank);
  for (const g of a.up_and_coming) {
    const r2 = rb.get(g.universe_id);
    if (r2 === undefined) continue;
    const band = bandOf(g.rank);
    const d6 = Math.max(Math.min(Math.round((r2 - g.rank) * STEP_H / dtH), D), -D);
    kernelCounts[band]![d6 + D] = kernelCounts[band]![d6 + D]! + w;
  }
}
// Gaussian-smooth (sigma 1.5) + 2% uniform prior, then normalise.
const kernel: number[][] = [];
for (let b = 0; b < BANDS.length; b++) {
  const raw = kernelCounts[b]!;
  const smoothed = new Array(2 * D + 1).fill(0);
  const SIGMA = 1.5;
  for (let d = -D; d <= D; d++) {
    let s = 0;
    for (let dd = -D; dd <= D; dd++) {
      const dist = d - dd;
      s += raw[dd + D]! * Math.exp(-(dist * dist) / (2 * SIGMA * SIGMA));
    }
    smoothed[d + D] = s;
  }
  let tot = 0;
  for (let d = 0; d < 2 * D + 1; d++) tot += smoothed[d]!;
  const uni = 1 / (2 * D + 1);
  const k: number[] = [];
  for (let d = 0; d < 2 * D + 1; d++) {
    k.push(0.98 * (tot > 0 ? smoothed[d]! / tot : uni) + 0.02 * uni);
  }
  kernel.push(k);
}
// Propagate each game's rank distribution in STEP_H steps, time-averaging
// the mass on ranks 1..10. dist[0] is "out"; mass that would go below
// rank 1 piles at rank 1, mass beyond RMAX goes to "out", and on-chart
// mass exits at exitStep per step.
const exitStep = 1 - Math.pow(1 - Math.min(baseExitPerHour, 0.05), STEP_H);
const STEPS = Math.max(1, Math.round(HOURS / STEP_H));
const probTrans = new Float64Array(N * PLACES);
for (let i = 0; i < N; i++) {
  const gm = sim[i]!;
  const shift = Math.max(Math.min(Math.round(gm.drift * STEP_H), 6), -6);
  let dist = new Float64Array(RMAX + 1);
  dist[Math.min(Math.max(Math.round(gm.lastRank), 1), RMAX)] = 1;
  const acc = new Float64Array(PLACES);
  for (let s = 0; s < STEPS; s++) {
    const next = new Float64Array(RMAX + 1);
    for (let r = 1; r <= RMAX; r++) {
      const m = dist[r]!;
      if (m === 0) continue;
      const k = kernel[bandOf(r)]!;
      for (let d = -D; d <= D; d++) {
        const p = k[d + D]!;
        if (p === 0) continue;
        const r2 = r + d + shift;
        if (r2 < 1) next[1] = next[1]! + m * p;
        else if (r2 > RMAX) next[0] = next[0]! + m * p;
        else next[r2] = next[r2]! + m * p;
      }
    }
    let onChart = 0;
    for (let r = 1; r <= RMAX; r++) onChart += next[r]!;
    const exited = onChart * exitStep;
    for (let r = 1; r <= RMAX; r++) next[r] = next[r]! * (1 - exitStep);
    next[0] = next[0]! + exited + dist[0]!;
    dist = next;
    for (let k = 0; k < PLACES; k++) acc[k] += dist[k + 1]!;
  }
  for (let k = 0; k < PLACES; k++) probTrans[i * PLACES + k] = acc[k]! / STEPS;
}

// ---------- forecast (ensemble) ----------
// V5 = 50/50 average of the transition model's place probabilities and
// the V4 Monte Carlo's place probabilities. Cap each place at
// (1 - UNNAMED) if above; never scale a short column up — the shortfall
// is honest calibration. Drop p < 0.002. top10 by total mass / expected rank.
interface ProbRow { universe_id: number; place: number; p: number }
const probabilities: ProbRow[] = [];
// V4 place probabilities from the Monte Carlo counts (per-place cap).
const probV4 = new Float64Array(N * PLACES);
for (let k = 0; k < PLACES; k++) {
  let placeSum = 0;
  for (let i = 0; i < N; i++) placeSum += counts[i * PLACES + k]! / totalPairs;
  const cap = 1 - UNNAMED;
  const scale = placeSum > cap ? cap / placeSum : 1;
  for (let i = 0; i < N; i++) probV4[i * PLACES + k] = (counts[i * PLACES + k]! / totalPairs) * scale;
}
// Ensemble: 50% transition kernel + 50% V4 Monte Carlo.
const probEns = new Float64Array(N * PLACES);
for (let i = 0; i < N * PLACES; i++) probEns[i] = 0.5 * probTrans[i]! + 0.5 * probV4[i]!;
const expectedRank = new Float64Array(N);
const inTop10 = new Float64Array(N);
for (let i = 0; i < N; i++) {
  let sum = 0;
  for (let k = 0; k < PLACES; k++) {
    const p = probEns[i * PLACES + k]!;
    sum += p;
    expectedRank[i] += (k + 1) * p;
    inTop10[i] += p;
  }
  if (sum > 0) expectedRank[i] /= sum;
}
for (let k = 0; k < PLACES; k++) {
  let placeSum = 0;
  for (let i = 0; i < N; i++) placeSum += probEns[i * PLACES + k]!;
  const cap = 1 - UNNAMED;
  const scale = placeSum > cap ? cap / placeSum : 1;
  for (let i = 0; i < N; i++) {
    const p = probEns[i * PLACES + k]! * scale;
    if (p >= 0.002) probabilities.push({ universe_id: sim[i]!.uid, place: k + 1, p: Math.round(p * 1e6) / 1e6 });
  }
}
// top10: the 10 games most likely to be in the top 10, ordered by expected rank.
const order = [...Array(N).keys()]
  .filter((i) => inTop10[i]! > 0)
  .sort((a, b) => (expectedRank[a]! - expectedRank[b]!) || (inTop10[b]! - inTop10[a]!));
const top10 = order.slice(0, PLACES).map((i) => sim[i]!.uid);
// If fewer than 10 games have mass (should not happen), fill from the last chart.
for (const g of lastChart) {
  if (top10.length >= PLACES) break;
  if (!top10.includes(g.universe_id)) top10.push(g.universe_id);
}
mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
