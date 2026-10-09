import fs from "node:fs";
import path from "node:path";

const defaultInputDir =
  "local-artifacts/outputs/training-checkpoints";
const defaultOutputDir =
  "local-artifacts/outputs/sweeps";

const inputDir = process.argv[2] || defaultInputDir;
const outputDir = process.argv[3] || defaultOutputDir;

const pufferArtifacts = [
  "puffer-mjwarp-local-substrate-smoke.json",
  "puffer-mjwarp-local-env-contract.json",
  "puffer-mjwarp-pufferppo-contract.json",
  "puffer-mjwarp-pufferppo-runtime.json",
  "puffer-mjwarp-gpu-score-kernel-smoke.json",
  "puffer-mjwarp-env-driver.json",
  "puffer-mjwarp-device-rollout.json",
  "puffer-mjwarp-device-rollout-buffer.json",
  "puffer-mjwarp-device-rollout-action-buffer.json",
  "puffer-mjwarp-device-rollout-torch-policy.json",
  "puffer-mjwarp-device-rollout-ppo-update.json",
  "puffer-mjwarp-device-ppo-train.json",
  "puffer-mjwarp-device-ppo-hold-probe.json",
  "puffer-mjwarp-device-ppo-hold-probe-f32.json",
  "puffer-mjwarp-device-ppo-hold-bc-probe.json",
  "puffer-mjwarp-device-ppo-down-swingup-probe.json",
  "puffer-mjwarp-device-ppo-down-swingup-conservative.json",
  "puffer-mjwarp-device-ppo-down-heavy-conservative.json",
  "puffer-mjwarp-device-ppo-link2-down-heavy-diagnostic.json",
  "puffer-mjwarp-device-ppo-link3-down-heavy-diagnostic.json",
  "puffer-mjwarp-device-ppo-link4-down-heavy-diagnostic.json",
  "puffer-mjwarp-device-rollout-link6.json",
  "puffer-mjwarp-device-rollout-random-horizon.json",
  "puffer-mjwarp-local-ppo-smoke.json",
  "puffer-mjwarp-local-ppo-hold-search.json",
  "puffer-mjwarp-stabilizer-bc.json",
  "puffer-mjwarp-down-warmstart.json",
  "puffer-mjwarp-mixed-warmstart.json",
  "puffer-mjwarp-energy-teacher.json",
  "puffer-mjwarp-energy-teacher-bc.json",
  "puffer-mjwarp-energy-teacher-sequence-bc.json",
  "puffer-mjwarp-energy-teacher-dagger.json",
  "puffer-mjwarp-sequence-bc-down-warmstart.json",
  "puffer-mjwarp-anchored-down-warmstart.json",
];

function asNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function readJson(file) {
  const fullPath = path.join(inputDir, file);
  if (!fs.existsSync(fullPath)) return null;
  return {
    file,
    fullPath,
    root: JSON.parse(fs.readFileSync(fullPath, "utf8")),
  };
}

function strictScoreFromMetric(metric) {
  const maxHeldSeconds = asNumber(metric?.maxHeldSeconds, asNumber(metric?.maxHoldSeconds));
  if (maxHeldSeconds < 1) return 0;
  return asNumber(metric?.maxStrictScore, asNumber(metric?.strictScore));
}

function downMetric(root) {
  if (root?.bestDownEvaluation) return root.bestDownEvaluation;
  if (root?.evaluation?.down) return root.evaluation.down;
  return root || {};
}

function holdMetric(root) {
  if (root?.bestHoldEvaluation) return root.bestHoldEvaluation;
  if (Array.isArray(root?.history) && root.history.length > 0) {
    return root.history[root.history.length - 1]?.evaluation?.hold || {};
  }
  return root?.evaluation?.hold || {};
}

function sourceRow(entry, index) {
  const { file, fullPath, root } = entry;
  const down = downMetric(root);
  const hold = holdMetric(root);
  const downHeld = asNumber(down.maxHeldSeconds, asNumber(down.maxHoldSeconds));
  const holdHeld = asNumber(hold.maxHeldSeconds, asNumber(hold.maxHoldSeconds));
  const strictScore = strictScoreFromMetric(down);
  const algorithm = String(root.algorithm || "");
  const plumbingArtifact =
    file.includes("substrate") ||
    file.includes("contract") ||
    file.includes("env-driver") ||
    file.includes("device-rollout") ||
    file.includes("gpu-score-kernel");
  const learned =
    algorithm.includes("behavior-clone") ||
    algorithm.includes("dagger") ||
    (!algorithm.includes("teacher") && !plumbingArtifact);
  const teacher = root.algorithm === "energy-pump-plus-stabilizer-teacher";

  return {
    experimentId: `current-${String(index + 1).padStart(2, "0")}-${file.replace(/\.json$/, "")}`,
    status: teacher ? "scaffold" : learned ? "completed" : "plumbing",
    algorithm: root.algorithm || root.schema || "unknown",
    stack: "MJWarp adapter, local Mac execution",
    linkCount: asNumber(root.links, 1),
    policyFamily: teacher ? "classical energy teacher" : learned ? "tiny GRU / local policy-gradient or BC" : "environment contract",
    policyParams: asNumber(root.policyParameters, learned ? 27267 : 0),
    forceScale: asNumber(root.forceScale, null),
    nworld: asNumber(root.nworld),
    rolloutSteps: asNumber(root.training?.rolloutSteps, asNumber(root.rolloutSteps, asNumber(root.steps))),
    wallclockSeconds: asNumber(root.training?.elapsedSeconds, asNumber(root.elapsedSeconds)),
    score: strictScore,
    maxHeldSeconds: downHeld,
    solvedOneSecond: downHeld >= 1,
    holdStartMaxHeldSeconds: holdHeld,
    holdStartSolvedOneSecond: holdHeld >= 1,
    countsTowardSolve: !teacher && downHeld >= 1,
    sourceArtifact: fullPath,
    note: file.includes("device-ppo-down-heavy-conservative")
      ? "Down-heavy one-link swing-up/catch PPO probe at force 120: mostly pure hanging starts, some pump/catch starts, conservative PPO, and stabilizer BC initialization. Counts only if held-out pure down-start holds for at least one second."
      : file.includes("device-ppo-link")
      ? "Parallel lower-link diagnostic down-heavy PPO run. It is exploratory only: higher links are not promoted until every lower link passes held-out pure down-start for at least one second."
      : file.includes("device-ppo-down-swingup-conservative")
      ? "Conservative one-link mixed-start swing-up/catch PPO probe at force 120 with lower LR, tighter clip, and stabilizer BC initialization. Counts only if held-out pure down-start holds for at least one second."
      : file.includes("device-ppo-down-swingup-probe")
      ? "One-link mixed-start swing-up/catch PPO probe at force 120 with stronger potential/catch reward and stabilizer BC initialization. Counts only if held-out pure down-start holds for at least one second."
      : file.includes("device-ppo-hold-bc-probe")
      ? "Device-buffer recurrent policy learned one-link hold-start with stabilizer BC warmup and passed held-out hold-start over one second. Down-start remains zero, so this is curriculum progress only."
      : file.includes("device-ppo-hold-probe")
      ? "Long-horizon one-link hold-start PPO probe. Force 32 avoids cart terminal but remains subsecond; force 64 is terminal-prone. Counts only if held-out down-start holds for at least one second."
      : file.includes("device-ppo-train")
      ? "Repeated MJWarp rollout-buffer PPO training: stochastic collect, persistent optimizer updates, deterministic down-start eval after each update. Counts only if held-out down-start holds for at least one second."
      : teacher
      ? "Teacher proves swing-up/catch signal in MJWarp, but it is not a learned policy."
      : learned && downHeld < 1
        ? "Learned/plumbing row does not solve down-start; score forced to 0 because hold is under 1s."
        : file.includes("gpu-score-kernel")
          ? "Links 1..6 score/observation/terminal Warp kernel matches NumPy scorer and is wired into the env scorer."
          : file.includes("device-rollout")
            ? file.includes("action-buffer")
              ? "Device-side MJWarp rollout smoke: external fixed-shape action tensor is consumed by a Warp ctrl kernel; precomputed buffer only, not learned."
              : file.includes("torch-policy")
                ? "Device-side MJWarp rollout smoke: recurrent Torch actor-critic actions/logprobs/values are bridged through wp.to_torch/wp.from_torch into fixed PPO buffers and Warp ctrl; untrained policy only."
                : file.includes("ppo-update")
                  ? "Device-side MJWarp rollout smoke: three PPO epochs consume fixed recurrent buffers and change parameters; smoke only, not learned."
              : "Device-side MJWarp rollout smoke: scripted action kernel only, no per-step CPU metric reads, not a learned policy."
          : "Current MJWarp/Puffer substrate evidence.",
  };
}

function queuedRows() {
  return [
    {
      experimentId: "queued-pufferppo-mingru-1link-sweep-a",
      status: "blocked-modal-spend-limit",
      algorithm: "PufferPPO",
      stack: "PufferLib + MJWarp on GPU",
      linkCount: 1,
      policyFamily: "Puffer MinGRU / PufferNet",
      policyParams: 1000000,
      forceScale: 240,
      nworld: 4096,
      rolloutSteps: 256,
      bpttHorizon: 128,
      rewardProfile: "energy + whip + catch basin + strict one-second gate",
      randomizedEpisodeLength: false,
      wallclockSeconds: 0,
      score: 0,
      maxHeldSeconds: 0,
      solvedOneSecond: false,
      countsTowardSolve: false,
      note: "First real Yacine-aligned run. Fixed horizon until whipping behavior appears.",
    },
    {
      experimentId: "queued-pufferppo-mingru-1link-random-horizon",
      status: "blocked-modal-spend-limit",
      algorithm: "PufferPPO",
      stack: "PufferLib + MJWarp on GPU",
      linkCount: 1,
      policyFamily: "Puffer MinGRU / PufferNet",
      policyParams: 1000000,
      forceScale: 240,
      nworld: 4096,
      rolloutSteps: 512,
      bpttHorizon: 256,
      rewardProfile: "same as sweep-a, randomized episode length after whip plateau",
      randomizedEpisodeLength: true,
      wallclockSeconds: 0,
      score: 0,
      maxHeldSeconds: 0,
      solvedOneSecond: false,
      countsTowardSolve: false,
      note: "Matches Yacine's trick only after policy learns whip but gets lazy on fixed endings.",
    },
    {
      experimentId: "queued-pufferppo-mingru-2link-promotion",
      status: "gated-by-1link",
      algorithm: "PufferPPO",
      stack: "PufferLib + MJWarp on GPU",
      linkCount: 2,
      policyFamily: "Puffer MinGRU / PufferNet",
      policyParams: 1000000,
      forceScale: 240,
      nworld: 4096,
      rolloutSteps: 512,
      bpttHorizon: 256,
      rewardProfile: "bend-order-preserving whip, no early straightness reward",
      randomizedEpisodeLength: false,
      wallclockSeconds: 0,
      score: 0,
      maxHeldSeconds: 0,
      solvedOneSecond: false,
      countsTowardSolve: false,
      note: "Do not run until held-out one-link down-start holds for at least 1s.",
    },
  ];
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined) return "";
  if (!Number.isFinite(Number(value))) return String(value);
  if (digits === 0) return Number(value).toFixed(0);
  return Number(value).toFixed(digits).replace(/\.?0+$/, "");
}

const currentRows = pufferArtifacts
  .map(readJson)
  .filter(Boolean)
  .map(sourceRow);

const rows = [...currentRows, ...queuedRows()];
fs.mkdirSync(outputDir, { recursive: true });

const jsonlPath = path.join(outputDir, "puffer-mjwarp-one-link-sweep-ledger.jsonl");
fs.writeFileSync(jsonlPath, rows.map((row) => JSON.stringify(row)).join("\n") + "\n");

const completedLearnedRows = rows.filter((row) => row.status === "completed");
const solvedLearnedRows = completedLearnedRows.filter((row) => row.countsTowardSolve);
const teacherRows = rows.filter((row) => row.status === "scaffold");
const bestLearned = completedLearnedRows
  .sort((a, b) => b.score - a.score || b.maxHeldSeconds - a.maxHeldSeconds)[0];
const bestScaffold = teacherRows
  .sort((a, b) => b.score - a.score || b.maxHeldSeconds - a.maxHeldSeconds)[0];

const tableRows = rows.map((row) =>
  [
    row.experimentId,
    row.status,
    row.algorithm,
    row.linkCount,
    row.policyFamily,
    formatNumber(row.policyParams, 0),
    formatNumber(row.wallclockSeconds, 1),
    formatNumber(row.score, 2),
    formatNumber(row.maxHeldSeconds, 3),
    row.solvedOneSecond ? "yes" : "no",
    row.countsTowardSolve ? "yes" : "no",
    row.note,
  ].join(" | "),
);

const markdown = [
  "# Puffer MJWarp One-Link Sweep Ledger",
  "",
  "This is the Yacine-shaped ledger: each row is a dot candidate with wallclock on x and strict score on y.",
  "Rows only count toward solve when a learned policy holds held-out down-start upright for at least one continuous second. Subsecond holds score 0.",
  "",
  "Current honest state:",
  "",
  `- Learned policy rows solved held-out down-start: ${solvedLearnedRows.length}/${completedLearnedRows.length}.`,
  `- Teacher scaffold rows with swing-up/catch signal: ${teacherRows.length}.`,
  `- Best learned row: ${bestLearned?.experimentId || "none"} with counting score ${formatNumber(bestLearned?.score || 0, 2)} and down-start hold ${formatNumber(bestLearned?.maxHeldSeconds || 0, 3)}s.`,
  `- Best non-counting scaffold row: ${bestScaffold?.experimentId || "none"} with score ${formatNumber(bestScaffold?.score || 0, 2)} and down-start hold ${formatNumber(bestScaffold?.maxHeldSeconds || 0, 3)}s.`,
  "- True PufferPPO/MinGRU sweep rows are queued, not run, because Modal GPU execution is blocked by the workspace spend limit.",
  "",
  "experiment | status | algorithm | links | policy | params | wallclock_s | score | down_hold_s | solved_1s | counts | note",
  "--- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---",
  ...tableRows,
  "",
  `JSONL: ${jsonlPath}`,
  "",
].join("\n");

const markdownPath = path.join(outputDir, "puffer-mjwarp-one-link-sweep-ledger.md");
fs.writeFileSync(markdownPath, markdown);

console.log(markdownPath);
console.log(jsonlPath);
