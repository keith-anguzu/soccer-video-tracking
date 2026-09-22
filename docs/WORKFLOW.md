# Analysis workflow

Run commands from the repository root. Paths and pitch dimensions below are examples: use the actual measured dimensions and your own footage.

## 1. Record or select footage

```bash
python analyze_match.py record --source 0 --out footage/match-001.mp4
```

Use a fixed camera and preserve the original recording for repeat analysis.

## 2. Calibrate the pitch

```bash
python analyze_match.py calibrate --video footage/match-001.mp4 \
  --pitch-length 40 --pitch-width 25 --halfway --verify-length 10 \
  --calibration calibrations/match-001.json
```

Create the `calibrations` folder first. The dimensions above are examples in metres. Follow the point-selection prompts. Halfway points and an independently measured distance help check calibration; a four-corner fit alone cannot validate its own scale.

## 3. Analyse into a new run folder

```bash
python analyze_match.py analyse footage/match-001.mp4 \
  --calibration calibrations/match-001.json \
  --output-dir reports/match-001-run-01
```

Use `--model yolo11s.pt` to select your local model instead of the default `yolo11n.pt`. Add `--ball` for experimental ball metrics. Use `--device mps` only if your installed runtime supports Apple acceleration; otherwise let the runtime choose or use `--device cpu`.

The single-video analysis command refuses to reuse a folder containing certain existing results. A unique folder for every run keeps comparisons understandable.

## 4. Inspect results

| File | What to inspect |
| --- | --- |
| `_raw_detections.csv` | Detection observations retained for troubleshooting. |
| `player_tracks.csv` | Processed player trajectories. |
| `player_summary.csv` | Movement summaries and smoothing corrections. |
| `tracking_quality.csv` | Quality findings to review before interpreting metrics. |
| `annotated_match.mp4` | Visual tracking output when video output is enabled; codec fallback may use AVI. |
| `_raw_ball.csv` | Ball detections when ball tracking is enabled. |
| `possession_summary.csv` | Conditional possession/pass estimates when usable ball results are available. |

Check the coordinate units, track continuity, team assignments, and raw versus smoothed distance. Visually inspect representative footage. Treat low-quality estimates as unresolved rather than validated performance statistics.

## 5. Record provenance

Copy [RUN_LOG_TEMPLATE.md](RUN_LOG_TEMPLATE.md) into the run folder as `RUN_LOG.md`. Capture the code and environment:

```bash
git rev-parse HEAD
git status --short
python -m pip freeze > reports/match-001-run-01/environment.txt
```

A commit hash only identifies committed code: record any uncommitted changes too. Keep the exact command and a copy of the calibration alongside the outputs. Run folders are ignored by Git, so back them up separately if needed.

## Advanced commands

Use the built-in help for required inputs and flags:

```bash
python analyze_match.py multicam --help
python analyze_match.py gps --help
python analyze_match.py sensor-fusion --help
```

Multi-camera analysis needs synchronization offsets and a calibration for each camera. Inspect `multicam_associations.csv` before equating global tracks with players. GPS/video fusion requires an explicit player mapping; preserve that mapping with the run.
