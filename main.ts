// Roblox Up-and-Coming top-10 forecaster.
//
// Model: each game's chart rank is a latent continuous value that drifts
// (per-hour drift and volatility estimated from the history, recent reads
// weighted more) and diffuses. A seeded Monte Carlo simulates every game
// forward over the 72-hour scored window; at each hour the games are sorted
// by their simulated rank and the top 10 recorded. Per-place probabilities
// are the fraction of (path, hour) pairs where a game holds that place,
// averaged over the whole window — which is exactly what the log score
// rewards, since one forecast is scored against every hourly read.
// Games can leave the chart (age-out / exit hazard estimated from the
// history); a small unnamed reserve covers games no history names.
//
// Deterministic: a seeded PRNG (mulberry32) drives every random draw, so
// two runs on the same history produce byte-identical output.
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
  gm.drift = Math.max(Math.min(gm.drift, 0.35), -0.35);
}
// Exit hazard: fraction of games present at a read that are gone at the next,
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
const VOL_SCALE = Number(process.env.MF_VOL_SCALE ?? 0.85);
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
      rankNow[i] = rankNow[i]! + THETA * (meanRank[i]! - rankNow[i]!) + gm.drift * DRIFT_SCALE + gm.vol * VOL_SCALE * rng.gaussian();
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

// ---------- forecast ----------
// Per-place probabilities, averaged over the window; cap each place at
// (1 - UNNAMED) so the unnamed reserve covers games the history never names.
interface ProbRow { universe_id: number; place: number; p: number }
const probabilities: ProbRow[] = [];
const expectedRank = new Float64Array(N);
const inTop10 = new Float64Array(N);
for (let i = 0; i < N; i++) {
  let sum = 0;
  for (let k = 0; k < PLACES; k++) {
    const p = counts[i * PLACES + k]! / totalPairs;
    sum += p;
    expectedRank[i] += (k + 1) * p;
    inTop10[i] += p;
  }
  if (sum > 0) expectedRank[i] /= sum;
}
for (let k = 0; k < PLACES; k++) {
  let placeSum = 0;
  for (let i = 0; i < N; i++) placeSum += counts[i * PLACES + k]! / totalPairs;
  const cap = 1 - UNNAMED;
  const scale = placeSum > cap ? cap / placeSum : 1;
  for (let i = 0; i < N; i++) {
    const p = (counts[i * PLACES + k]! / totalPairs) * scale;
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
