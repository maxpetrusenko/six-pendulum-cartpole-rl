import fs from "node:fs";
import path from "node:path";

const defaultDir =
  "local-artifacts/outputs/training-checkpoints";
const defaultOut =
  "local-artifacts/outputs/sweeps";

const inputDir = process.argv[2] || defaultDir;
const outputDir = process.argv[3] || defaultOut;

function asNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function pickValidation(training, root) {
  return (
    training?.validationByPose?.down ||
    training?.validation ||
    root?.validation ||
    root?.training?.validation ||
    {}
  );
}

function profileLabel(training) {
  if (typeof training.profile === "string") return training.profile;
  if (typeof training.curriculum === "string") return training.curriculum;
  if (Array.isArray(training.curriculum)) return `${training.curriculum.length} stages`;
  return "";
}

function maxHistoryHold(training, targetLinks, stageFilter = () => true) {
  let maxHold = 0;
  for (const entry of training?.history || []) {
    if (!stageFilter(entry)) continue;
    const candidates = [entry, entry.down, entry.hold, entry.mixed].filter(Boolean);
    for (const candidate of candidates) {
      if (targetLinks && candidate.links && candidate.links !== targetLinks) continue;
      maxHold = Math.max(
        maxHold,
        asNumber(candidate.maxHoldSeconds),
        asNumber(candidate.maxHeldSeconds),
        asNumber(candidate.held),
      );
    }
  }
  return maxHold;
}

function strictScore(metrics, horizonSeconds) {
  if (asNumber(metrics.strictScore) > 0) return asNumber(metrics.strictScore);
  const maxHold = asNumber(metrics.maxHoldSeconds, asNumber(metrics.maxHeldSeconds));
  const maxHoldP10 = asNumber(metrics.maxHoldSecondsP10, asNumber(metrics.maxHeldSecondsP10));
  const solvedRate = asNumber(metrics.solvedOneSecondRate);
  if (maxHold < 1) return 0;
  const centerRatio = asNumber(metrics.centerRatio, 0.62);
  const smoothPenalty = asNumber(metrics.smoothPenalty);
  const railPenalty = Math.max(0, 0.62 - centerRatio) * 10;
  return Math.max(0,
    100 * solvedRate +
    10 * Math.min(maxHold, horizonSeconds) +
    5 * Math.min(maxHoldP10, horizonSeconds) -
    railPenalty -
    smoothPenalty
  );
}

function simSteps(root, training) {
  const population = asNumber(training.population);
  const generations = asNumber(training.generations);
  const controlHz = asNumber(training.controlHz);
  const horizonSeconds = asNumber(training.horizonSeconds);
  if (population && generations && controlHz && horizonSeconds) {
    const stageGenerations = Array.isArray(training.stages)
      ? training.stages.reduce((sum, stage) => sum + asNumber(stage.generations), 0)
      : generations;
    return population * stageGenerations * controlHz * horizonSeconds;
  }
  if (population && generations && asNumber(root.steps)) {
    return population * generations * asNumber(root.steps);
  }
  return asNumber(training.totalTimesteps);
}

function readRows() {
  return fs
    .readdirSync(inputDir)
    .filter((file) => file.endsWith(".json"))
    .sort()
    .flatMap((file) => {
      const fullPath = path.join(inputDir, file);
      const root = JSON.parse(fs.readFileSync(fullPath, "utf8"));
      const training = root.training || {};
      const validation = pickValidation(training, root);
      const horizonSeconds = asNumber(training.horizonSeconds, asNumber(root.steps) * asNumber(root.dt), 8);
      const targetLinks = asNumber(root.links);
      const finalDownMaxHold = asNumber(validation.maxHoldSeconds, asNumber(validation.maxHeldSeconds));
      const bestStageMaxHold = maxHistoryHold(training, targetLinks);
      const bestHoldStartMaxHold = maxHistoryHold(training, targetLinks, (entry) =>
        String(entry.stage || "").includes("hold") || entry.pose === "hold",
      );
      const bestAngleMaxHold = maxHistoryHold(training, targetLinks, (entry) =>
        String(entry.stage || "").includes("angle") || entry.pose === "angle",
      );
      const metrics = {
        ...validation,
        maxHoldSeconds: finalDownMaxHold,
      };
      const elapsed = asNumber(training.elapsedSeconds);
      const steps = simSteps(root, training);
      const row = {
        artifact: file,
        algorithm: root.algorithm || "unknown",
        modelType: root.modelType || "",
        profile: profileLabel(training),
        wallclockSeconds: elapsed,
        simSteps: steps,
        sps: elapsed > 0 && steps > 0 ? steps / elapsed : 0,
        strictScore: strictScore(metrics, horizonSeconds),
        maxHoldSeconds: finalDownMaxHold,
        finalDownMaxHoldSeconds: finalDownMaxHold,
        bestStageMaxHoldSeconds: bestStageMaxHold,
        bestHoldStartMaxHoldSeconds: bestHoldStartMaxHold,
        bestAngleMaxHoldSeconds: bestAngleMaxHold,
        maxHoldSecondsP10: asNumber(validation.maxHoldSecondsP10, asNumber(validation.maxHeldSecondsP10)),
        solvedOneSecondRate: asNumber(validation.solvedOneSecondRate),
        whiplashSeconds: asNumber(validation.whiplashSeconds, asNumber(validation.whip)),
        centerRatio: asNumber(validation.centerRatio),
        population: asNumber(training.population),
        generations: asNumber(training.generations),
        totalTimesteps: asNumber(training.totalTimesteps),
        controlHz: asNumber(training.controlHz),
        highCompute: !training.smoke && elapsed > 0,
        path: fullPath,
      };
      const dots = Array.isArray(training.experimentDots)
        ? training.experimentDots.map((dot) => ({
            ...row,
            artifact: `${file}#${dot.stage}-${dot.wallclockSeconds}`,
            stage: dot.stage,
            wallclockSeconds: asNumber(dot.wallclockSeconds, row.wallclockSeconds),
            strictScore: asNumber(dot.metrics?.strictScore),
            maxHoldSeconds: asNumber(dot.metrics?.maxHoldSeconds),
            maxHoldSecondsP10: asNumber(dot.metrics?.maxHoldSecondsP10),
            solvedOneSecondRate: asNumber(dot.metrics?.solvedOneSecondRate),
            whiplashSeconds: asNumber(dot.metrics?.whiplashSeconds),
            centerRatio: asNumber(dot.metrics?.centerRatio),
          }))
        : [];
      return [row, ...dots];
    });
}

function fmt(value, digits = 3) {
  if (!Number.isFinite(value)) return "";
  return value.toFixed(digits).replace(/\.?0+$/, "");
}

const rows = readRows().sort(
  (a, b) =>
    b.strictScore - a.strictScore ||
    b.finalDownMaxHoldSeconds - a.finalDownMaxHoldSeconds ||
    b.bestStageMaxHoldSeconds - a.bestStageMaxHoldSeconds,
);
fs.mkdirSync(outputDir, { recursive: true });
const jsonlPath = path.join(outputDir, "latest-six-pendulum-comparison.jsonl");
fs.writeFileSync(jsonlPath, rows.map((row) => JSON.stringify(row)).join("\n") + "\n");

const tableRows = rows
  .filter((row) => !row.artifact.includes("#"))
  .map((row) =>
    [
      row.artifact,
      row.algorithm,
      row.profile,
      fmt(row.wallclockSeconds, 1),
      fmt(row.sps, 0),
      fmt(row.strictScore, 2),
      fmt(row.finalDownMaxHoldSeconds, 3),
      fmt(row.bestStageMaxHoldSeconds, 3),
      fmt(row.bestHoldStartMaxHoldSeconds, 3),
      fmt(row.bestAngleMaxHoldSeconds, 3),
      fmt(row.maxHoldSecondsP10, 3),
      fmt(row.solvedOneSecondRate, 3),
      fmt(row.whiplashSeconds, 3),
      fmt(row.centerRatio, 3),
    ].join(" | "),
  );

const markdown = [
  "# Six Pendulum Experiment Comparison",
  "",
  "Score and sorting use final down-start validation only. Runs with less than one second mean hold keep score 0; stage holds stay diagnostics.",
  "",
  "artifact | algorithm | profile | wallclock_s | sps | strict_score | final_down_hold_s | best_stage_hold_s | best_hold_start_s | best_angle_s | p10_hold_s | solved_rate | whiplash_s | center_ratio",
  "--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
  ...tableRows,
  "",
  `JSONL dots: ${jsonlPath}`,
  "",
].join("\n");

const markdownPath = path.join(outputDir, "latest-six-pendulum-comparison.md");
fs.writeFileSync(markdownPath, markdown);
console.log(markdownPath);
console.log(jsonlPath);
