# Soccer Video Tracking & Analytics

Turning soccer match footage into player movement data and interpretable match insights.

This Python project explores accessible soccer analytics using computer vision: record a match, calibrate the playing area, track players, and inspect the resulting measurements. It also includes experimental ball metrics, multi-camera tracking, and GPS/video fusion.

**Status:** development prototype. Outputs need quality checks before being used as performance evidence.

## About me

I’m Keith Anguzu, building soccer analytics tools that turn match footage into insights about player movement and team performance. My work explores computer vision, tracking, and data analysis to make match analysis more accessible and traceable.

## What the project does

| Command | Purpose |
| --- | --- |
| `record` | Save footage from a camera or video source. |
| `calibrate` | Map image coordinates to pitch coordinates using measured dimensions. |
| `analyse` | Track players, assign teams, smooth movement, and export summaries. |
| `live` | Show a live tracking demonstration. |
| `multicam` | Combine synchronized, individually calibrated camera recordings. |
| `gps` | Normalize player GPS exports. |
| `sensor-fusion` | Combine video and GPS tracks using a player mapping file. |

Ball tracking is optional (`--ball`). Possession and pass estimates depend on ball detection quality and should be treated as experimental.

## Repository guide

- [`analyze_match.py`](analyze_match.py) — current pipeline and command-line entry point.
- [`requirements.txt`](requirements.txt) — direct third-party dependencies.
- [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — commands, output guide, and quality checks.
- [`docs/RUN_LOG_TEMPLATE.md`](docs/RUN_LOG_TEMPLATE.md) — template for tracing an analysis run.
- [`CHANGELOG.md`](CHANGELOG.md) — human-readable project changes.

## Get started

From the project folder, create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analyze_match.py --help
```

Dependencies are not pinned yet; this is not a reproducible environment lock. Record the installed versions for each run. Camera recording and interactive calibration require a local display. Model weights and footage are supplied separately; Ultralytics may download a named model when it is first used.

Try a short analysis using your own video:

```bash
python analyze_match.py analyse footage/match.mp4 \
  --max-seconds 30 --no-video \
  --output-dir reports/match-001-test
```

Without calibration, spatial measurements are in **pixels**, not metres. See the [workflow](docs/WORKFLOW.md) to calibrate and run a full analysis.

## Keep results traceable

Give every run a unique output folder. Save its command, code commit, calibration, dependency versions, and quality findings using the [run log template](docs/RUN_LOG_TEMPLATE.md). Commit code and documentation changes with a descriptive message, then push them to GitHub.

Raw footage, model weights, GPS data, and generated reports stay outside version control by default. The repository holds code and documentation; it is not a backup of those local assets.

## Current limitations

- Tracking IDs can change after occlusion; they are not guaranteed player identities.
- Metre-based measurements depend on valid pitch calibration.
- Noisy tracks can inflate distance; review smoothing corrections and tracking quality.
- Ball detection can be unreliable in wide-angle footage.
- Review camera associations and GPS player mappings before interpreting fused results.
