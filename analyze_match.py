"""Soccer pipeline: record, calibrate, analyse, multi-camera fusion, and live demo.

Intended workflow for a tripod-mounted camera at a pickup game
--------------------------------------------------------------
1. Record raw footage (no inference, full resolution, nothing to re-run later):
       python analyze_match.py record --source 0 --out footage/2026-09-05.mp4

2. Calibrate once per session. Click the four corners of the playing area, and
   ideally the two halfway-line touchline points as well, then measure one known
   distance on the ground and let the tool check itself against it:
       python analyze_match.py calibrate --video footage/2026-09-05.mp4 \
              --pitch-length 40 --pitch-width 25 --halfway --verify-length 10

3. Analyse the recording. This is where the real data comes from, because you
   can re-run it as many times as you like:
       python analyze_match.py analyse footage/2026-09-05.mp4

4. Live view for showing people on the sideline. Treat its numbers as a demo:
       python analyze_match.py live --source 0

5. Fuse two or more synchronized, individually calibrated recordings:
       python analyze_match.py multicam \
              --video footage/sideline.mp4 --video footage/drone.mp4 \
              --camera-name Sideline --camera-name Drone \
              --camera-calibration calibration_sideline.json \
              --camera-calibration calibration_drone.json \
              --camera-offset 0 --camera-offset 2.4 \
              --team-a Orange --team-b Green

6. Normalize player GPS exports, then optionally fuse them with video tracks:
       python analyze_match.py gps --gps-file player_7.gpx --player 7
       python analyze_match.py sensor-fusion \
              --video-tracks reports/video_mvp/multicam_fused_tracks.csv \
              --gps-tracks reports/video_mvp/gps_normalized_tracks.csv \
              --player-map player_map.csv

Design notes
------------
* Teams are integers (0 and 1) everywhere internally. Colour names are display
  only, so a name flip can never orphan an accumulated statistic.
* Cluster centres are matched to the previous fit on every refit, so "Team 0"
  keeps meaning the same shirt even when the two kits share a lightness.
* Every measurement in metres requires a calibration file. Without one the
  pipeline still runs but reports pixels and says so.

Why distance is smoothed before it is summed
--------------------------------------------
A bounding-box foot point jitters by a few centimetres every frame even when the
player is standing still, and summing raw frame-to-frame displacement turns that
jitter into distance that nobody ran. Measured on synthetic tracks at 30 fps,
against a real youth-match total of 7 to 9 km:

    foot-point noise    added to a 90-minute distance
    2 cm                +0.8 km
    3 cm                +1.8 km
    5 cm                +5.4 km

So positions are smoothed per track with a Savitzky-Golay filter before any
differencing, which recovers a known distance to within about 0.3% at all three
noise levels. The summary reports the unsmoothed total beside the smoothed one,
as distance_raw and jitter_removed_percent, so the size of the correction is
always visible. If those two numbers are far apart the tracking is noisy and the
distances should not be quoted.

Why a four-corner calibration cannot check itself
-------------------------------------------------
Four point correspondences give eight equations for a homography's eight degrees
of freedom, so the fit is exact and its reprojection residual is always ~1e-13
no matter where the corners were clicked. That residual proves nothing. Click
the two halfway-line points as well (--halfway) to make the residual meaningful,
and measure a known distance on the ground (--verify-length) for an independent
check of the scale.

Ball metrics
------------
Ball tracking is OFF by default. COCO's sports-ball class often misses a football
on a wide pitch view. With --ball the analyse command now produces possession and
pass estimates alongside a ball detection rate, and the quality report refuses to
endorse them when that rate is low. Check the rate before quoting any of it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from ultralytics import YOLO

PERSON_CLASS = 0
BALL_CLASS = 32
UNKNOWN_TEAM = -1
MAX_PLAUSIBLE_SPEED_MS = 12.0
VERSION = "2026.09.13-csv-checkpoints"

# Write each row immediately and checkpoint after this many observations,
# as well as at each progress update. Raw files are kept for troubleshooting.
DETECTION_FLUSH_ROWS = 500

DETECTION_FIELDS = (
    "frame", "time_seconds", "track_id", "confidence",
    "lab_l", "lab_a", "lab_b", "image_x", "image_y",
    "pitch_x", "pitch_y", "x1", "y1", "x2", "y2",
)
BALL_FIELDS = ("frame", "time_seconds", "ball_x", "ball_y", "confidence")

TEAM_PALETTE = {
    "Black": (255, 110, 40),
    "White": (20, 190, 255),
    "Red": (40, 40, 240),
    "Orange": (0, 140, 255),
    "Yellow": (0, 230, 255),
    "Green": (40, 200, 40),
    "Cyan": (220, 220, 20),
    "Blue": (240, 100, 20),
    "Purple": (200, 70, 180),
    "Pink": (180, 90, 255),
}
UNKNOWN_COLOUR = (170, 170, 170)
# Display colours are deliberately independent of bib colours.  Blue and
# magenta remain visible against green turf, orange/green bibs and white lines.
DISPLAY_TEAM_COLOURS = {
    0: (255, 80, 20),       # blue (OpenCV uses BGR)
    1: (255, 30, 255),      # magenta
    UNKNOWN_TEAM: UNKNOWN_COLOUR,
}
SUPPORTED_TEAM_COLOURS = tuple(TEAM_PALETTE)
REFERENCE_BGR = {
    "Black": (25, 25, 25), "White": (235, 235, 235), "Red": (35, 35, 220),
    "Orange": (20, 125, 245), "Yellow": (25, 220, 235), "Green": (45, 165, 55),
    "Cyan": (210, 200, 35), "Blue": (210, 80, 35), "Purple": (155, 65, 145),
    "Pink": (175, 105, 240),
}


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, default=Path("reports/video_mvp"))
    common.add_argument("--calibration", type=Path, default=Path("calibration.json"),
                        help="Session calibration written by the calibrate command")

    model_args = argparse.ArgumentParser(add_help=False)
    model_args.add_argument("--model", default="yolo11n.pt")
    model_args.add_argument("--device", default=None,
                            help="torch device, e.g. mps on Apple Silicon or cpu")
    model_args.add_argument("--imgsz", type=int, default=960,
                            help="Inference size. Raise it if distant players are missed.")
    model_args.add_argument("--confidence", type=float, default=0.30)
    model_args.add_argument("--iou", type=float, default=0.55,
                            help="YOLO overlap threshold; lower values suppress more duplicates")
    model_args.add_argument("--tracker", choices=("bytetrack.yaml", "botsort.yaml"),
                            default="botsort.yaml",
                            help="BoT-SORT is more stable when players cross; ByteTrack is faster")
    model_args.add_argument("--team-min-samples", type=int, default=12)
    model_args.add_argument("--team-a", choices=SUPPORTED_TEAM_COLOURS, default=None,
                            help="Known colour for Team 0, for example Orange")
    model_args.add_argument("--team-b", choices=SUPPORTED_TEAM_COLOURS, default=None,
                            help="Known colour for Team 1, for example Green")
    model_args.add_argument("--team-max-distance", type=float, default=95.0,
                            help="Reject shirt colours too far from both requested colours")
    model_args.add_argument("--duplicate-iou", type=float, default=0.82,
                            help="Suppress duplicate person boxes above this overlap")
    model_args.add_argument("--ball", action="store_true",
                            help="Enable experimental ball, possession and pass estimates")
    model_args.add_argument("--ball-confidence", type=float, default=0.12)
    model_args.add_argument("--possession-radius", type=float, default=1.5,
                            help="Max ball-to-player distance in player heights")
    model_args.add_argument("--pass-separation", type=float, default=2.0,
                            help="Min distance between consecutive owners, in player "
                                 "heights, before a change counts as a pass rather "
                                 "than a tracking ID switch")
    model_args.add_argument("--smoothing-seconds", type=float, default=0.5,
                            help="Savitzky-Golay window applied to positions before "
                                 "distances are summed. 0 disables smoothing, which "
                                 "will inflate distance. See the module docstring.")
    model_args.add_argument("--min-speed", type=float, default=0.20,
                            help="Speeds below this (m/s when calibrated) count as "
                                 "standing still, so residual noise is not summed")
    model_args.add_argument("--max-gap-seconds", type=float, default=1.0,
                            help="Longer gaps in a track split it into segments. "
                                 "Movement across a gap is not counted as distance.")

    record = sub.add_parser("record", parents=[common],
                            help="Capture raw footage with no inference")
    record.add_argument("--source", default="0")
    record.add_argument("--out", type=Path, default=None)
    record.add_argument("--width", type=int, default=None)
    record.add_argument("--height", type=int, default=None)
    record.add_argument("--fps", type=float, default=30.0)

    calibrate = sub.add_parser("calibrate", parents=[common],
                               help="Click pitch reference points and save a homography")
    calibrate.add_argument("--video", type=Path, default=None)
    calibrate.add_argument("--source", default=None, help="Live source instead of a video")
    calibrate.add_argument("--pitch-length", type=float, required=True,
                           help="Goal-to-goal distance in metres")
    calibrate.add_argument("--pitch-width", type=float, required=True,
                           help="Touchline-to-touchline distance in metres")
    calibrate.add_argument("--margin", type=float, default=3.0,
                           help="Metres outside the pitch still counted as in play")
    calibrate.add_argument("--halfway", action="store_true",
                           help="Also click the two halfway-line touchline points. "
                                "Six points make the reprojection error meaningful; "
                                "with only four corners the fit is exact by "
                                "construction and the error is always about zero.")
    calibrate.add_argument("--verify-length", type=float, default=None,
                           help="Independent scale check. Measure a known distance on "
                                "the ground, pass it in metres, then click its two "
                                "ends when prompted.")

    analyse = sub.add_parser("analyse", parents=[common, model_args],
                             help="Process a recording into CSVs and an annotated video")
    analyse.add_argument("video", type=Path)
    analyse.add_argument("--sample-every", type=int, default=1)
    analyse.add_argument("--max-seconds", type=float, default=None)
    analyse.add_argument("--no-video", action="store_true",
                         help="Skip writing the annotated video")

    live = sub.add_parser("live", parents=[common, model_args],
                          help="Live sideline view. Demo quality, not a data source.")
    live.add_argument("--source", default="0")
    live.add_argument("--record-live", action="store_true")

    multicam = sub.add_parser(
        "multicam", parents=[common, model_args],
        help="Analyse calibrated camera recordings and fuse them on one pitch",
    )
    multicam.add_argument(
        "--video", type=Path, action="append", required=True,
        help="Recording to include. Repeat once per camera.",
    )
    multicam.add_argument(
        "--camera-name", action="append", default=None,
        help="Short camera label. Repeat in the same order as --video.",
    )
    multicam.add_argument(
        "--camera-calibration", type=Path, action="append", required=True,
        help="Calibration for each camera. Repeat in the same order as --video.",
    )
    multicam.add_argument(
        "--camera-offset", type=float, action="append", default=None,
        help=("Seconds to add to each recording timestamp. If a camera started "
              "2.4 seconds after the reference camera, use 2.4."),
    )
    multicam.add_argument("--sample-every", type=int, default=1)
    multicam.add_argument("--max-seconds", type=float, default=None)
    multicam.add_argument("--no-video", action="store_true",
                         help="Do not create per-camera annotated videos")
    multicam.add_argument(
        "--sync-tolerance", type=float, default=0.20,
        help="Maximum timestamp difference for cross-camera comparison",
    )
    multicam.add_argument(
        "--fusion-distance", type=float, default=2.0,
        help="Maximum median pitch distance in metres for the same player",
    )
    multicam.add_argument(
        "--fusion-min-overlap", type=int, default=3,
        help="Minimum synchronized observations needed to merge two tracks",
    )

    gps = sub.add_parser(
        "gps", parents=[common],
        help="Normalize and synchronize GPS tracker exports (CSV, GPX, TCX or FIT)",
    )
    gps.add_argument("--gps-file", type=Path, action="append", required=True,
                     help="Tracker export. Repeat once per player/file.")
    gps.add_argument("--player", action="append", default=None,
                     help="Player label for each file; defaults to its filename.")
    gps.add_argument("--gps-offset", type=float, action="append", default=None,
                     help="Seconds added to each file's relative time.")
    gps.add_argument("--match-start", default=None,
                     help="ISO timestamp for match clock 0, e.g. 2026-09-12T18:00:00-07:00")
    gps.add_argument("--resample-hz", type=float, default=1.0,
                     help="Output sampling rate; 0 preserves original samples.")
    gps.add_argument("--max-speed", type=float, default=12.0,
                     help="Mark samples above this speed (m/s) as implausible.")
    gps.add_argument("--pitch-origin-lat", type=float, default=None)
    gps.add_argument("--pitch-origin-lon", type=float, default=None)
    gps.add_argument("--pitch-bearing", type=float, default=0.0,
                     help="Clockwise degrees from north along pitch x axis.")

    fusion = sub.add_parser(
        "sensor-fusion", parents=[common],
        help="Combine an existing calibrated video-track CSV with normalized GPS",
    )
    fusion.add_argument("--video-tracks", type=Path, required=True,
                        help="multicam_fused_tracks.csv or player_tracks.csv")
    fusion.add_argument("--gps-tracks", type=Path, required=True,
                        help="gps_normalized_tracks.csv created by the gps command")
    fusion.add_argument("--player-map", type=Path, required=True,
                        help="CSV with gps_player and global_player_id columns")
    fusion.add_argument("--sync-tolerance", type=float, default=0.60)
    fusion.add_argument("--prefer-gps-position", action="store_true",
                        help="Use GPS position when both sensors observe a player")

    return parser


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

class Calibration:
    """Maps image pixels to pitch metres for a camera that does not move."""

    def __init__(self, homography, pitch_length: float, pitch_width: float,
                 margin: float, image_size, diagnostics: dict | None = None) -> None:
        self.homography = np.asarray(homography, dtype=np.float64)
        self.inverse = np.linalg.inv(self.homography)
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.margin = float(margin)
        self.image_size = tuple(int(v) for v in image_size)
        self.diagnostics = dict(diagnostics or {})

    @classmethod
    def load(cls, path: Path | None) -> "Calibration | None":
        if path is None or not Path(path).exists():
            return None
        data = json.loads(Path(path).read_text())
        return cls(data["homography"], data["pitch_length"], data["pitch_width"],
                   data.get("margin", 3.0), data.get("image_size", (0, 0)),
                   data.get("diagnostics"))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "homography": self.homography.tolist(),
            "pitch_length": self.pitch_length,
            "pitch_width": self.pitch_width,
            "margin": self.margin,
            "image_size": list(self.image_size),
            "diagnostics": self.diagnostics,
            "created": datetime.now().isoformat(timespec="seconds"),
            "pipeline_version": VERSION,
        }, indent=2))

    def to_pitch(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def to_image(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.inverse).reshape(-1, 2)

    def on_pitch(self, pitch_xy) -> bool:
        x, y = float(pitch_xy[0]), float(pitch_xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        return (-self.margin <= x <= self.pitch_length + self.margin
                and -self.margin <= y <= self.pitch_width + self.margin)

    def warns_about_size(self, frame) -> bool:
        height, width = frame.shape[:2]
        return self.image_size != (0, 0) and self.image_size != (width, height)

    def size_warning(self, frame) -> str | None:
        """One message, so live and analyse cannot disagree about this."""
        if not self.warns_about_size(frame):
            return None
        height, width = frame.shape[:2]
        return (f"Calibration was made at {self.image_size[0]}x{self.image_size[1]}, "
                f"this footage is {width}x{height}. Every metre value will be wrong. "
                f"Re-run calibrate against this footage.")


def click_points(frame: np.ndarray, labels: list[str],
                 window: str = "Calibration") -> list[tuple[float, float]]:
    picked: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < len(labels):
            picked.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(1400, frame.shape[1]), min(900, frame.shape[0]))
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            display = frame.copy()
            for index, point in enumerate(picked):
                centre = (int(point[0]), int(point[1]))
                cv2.circle(display, centre, 7, (0, 255, 255), -1)
                cv2.putText(display, f"{index + 1} {labels[index]}",
                            (centre[0] + 12, centre[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if len(picked) < len(labels):
                banner = f"Click: {labels[len(picked)]}    (u = undo, q = cancel)"
            else:
                banner = "Enter = accept    u = undo    q = cancel"
            cv2.rectangle(display, (0, 0), (display.shape[1], 46), (10, 10, 10), -1)
            cv2.putText(display, banner, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("u") and picked:
                picked.pop()
            elif key == ord("q"):
                raise SystemExit("Calibration cancelled.")
            elif key in (13, 10) and len(picked) == len(labels):
                return picked
    finally:
        cv2.destroyWindow(window)


def draw_pitch_grid(frame: np.ndarray, calibration: Calibration,
                    step: float = 5.0) -> np.ndarray:
    """Overlay a metre grid so you can eyeball whether the homography is sane."""
    overlay = frame.copy()
    length, width = calibration.pitch_length, calibration.pitch_width
    for x in np.arange(0.0, length + 1e-6, step):
        line = calibration.to_image([[x, 0.0], [x, width]])
        cv2.line(overlay, tuple(line[0].astype(int)), tuple(line[1].astype(int)),
                 (60, 220, 60), 1, cv2.LINE_AA)
    for y in np.arange(0.0, width + 1e-6, step):
        line = calibration.to_image([[0.0, y], [length, y]])
        cv2.line(overlay, tuple(line[0].astype(int)), tuple(line[1].astype(int)),
                 (60, 220, 60), 1, cv2.LINE_AA)
    corners = calibration.to_image(
        [[0, 0], [length, 0], [length, width], [0, width]]).astype(int)
    cv2.polylines(overlay, [corners.reshape(-1, 1, 2)], True, (0, 255, 255), 3, cv2.LINE_AA)
    return overlay


def fit_homography(image_points: np.ndarray, pitch_points: np.ndarray
                   ) -> tuple[np.ndarray, float, bool]:
    """Fit image-to-pitch homography and say whether the residual means anything.

    With exactly four correspondences the system is square: the fit reproduces
    the clicks exactly and the residual is numerical noise regardless of whether
    the clicks were correct. Only with five or more points does the residual
    carry information, so RANSAC is used there and the caller is told which case
    it is in.
    """
    exact = len(image_points) <= 4
    method = 0 if exact else cv2.RANSAC
    homography, _ = cv2.findHomography(image_points, pitch_points,
                                       method=method, ransacReprojThreshold=0.75)
    if homography is None:
        raise SystemExit("Could not fit a homography. Re-click the reference points.")
    projected = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    residual = float(np.linalg.norm(projected - pitch_points, axis=1).mean())
    return homography, residual, exact


def verify_scale(frame: np.ndarray, calibration: Calibration,
                 known_length: float) -> dict:
    """Independent check: measure something you have measured with a tape."""
    labels = [f"START of your {known_length} m measurement",
              f"END of your {known_length} m measurement"]
    points = np.array(click_points(frame, labels, window="Scale check"),
                      dtype=np.float64)
    pitch = calibration.to_pitch(points)
    measured = float(np.linalg.norm(pitch[1] - pitch[0]))
    error = measured - known_length
    percent = 100.0 * error / known_length if known_length else float("nan")
    print(f"\nScale check: you said {known_length:.2f} m, "
          f"the calibration measures {measured:.2f} m "
          f"({error:+.2f} m, {percent:+.1f}%)")
    if abs(percent) <= 3:
        print("That is good. Distances from this calibration are usable.")
    elif abs(percent) <= 8:
        print("Usable but loose. Expect a few percent of error in every distance.")
    else:
        print("That is too far out. The corner clicks are probably not on the "
              "corners you think, or the pitch dimensions passed in are wrong.")
    return {"known_length_m": known_length, "measured_length_m": measured,
            "error_m": error, "error_percent": percent}


def run_calibrate(args: argparse.Namespace) -> None:
    frame = grab_reference_frame(args)
    length, width = args.pitch_length, args.pitch_width

    labels = ["FAR-LEFT corner", "FAR-RIGHT corner",
              "NEAR-RIGHT corner", "NEAR-LEFT corner"]
    pitch_reference = [[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]]
    if args.halfway:
        labels += ["HALFWAY line at the FAR touchline",
                   "HALFWAY line at the NEAR touchline"]
        pitch_reference += [[length / 2.0, 0.0], [length / 2.0, width]]

    image_points = np.array(click_points(frame, labels), dtype=np.float64)
    pitch_points = np.array(pitch_reference, dtype=np.float64)

    homography, residual, exact = fit_homography(image_points, pitch_points)
    height, frame_width = frame.shape[:2]
    diagnostics = {"reference_points": len(image_points),
                   "mean_reprojection_error_m": residual,
                   "residual_is_meaningful": not exact}
    calibration = Calibration(homography, length, width, args.margin,
                              (frame_width, height), diagnostics)

    if exact:
        print("\nFour points fit a homography exactly, so the reprojection error "
              "below is arithmetic, not evidence. Re-run with --halfway for an "
              "error that means something.")
        print(f"Mean reprojection error: {residual:.6f} m (expected to be ~0)")
    else:
        print(f"\nMean reprojection error over {len(image_points)} points: "
              f"{residual:.3f} m")
        if residual > 0.5:
            print("That is high. At least one click is probably in the wrong place.")

    if args.verify_length:
        diagnostics.update(verify_scale(frame, calibration, args.verify_length))
        calibration.diagnostics = diagnostics
    else:
        print("\nNo scale check was run. Pass --verify-length with a distance you "
              "have actually measured on the ground to confirm the metres are real.")

    preview = draw_pitch_grid(frame, calibration)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 46), (10, 10, 10), -1)
    cv2.putText(preview, "Grid should sit flat on the ground. Enter = save, q = discard.",
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.namedWindow("Calibration check", cv2.WINDOW_NORMAL)
    cv2.imshow("Calibration check", preview)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            break
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Discarded.")
    cv2.destroyAllWindows()

    calibration.save(args.calibration)
    print(f"Saved calibration: {args.calibration}")


def grab_reference_frame(args: argparse.Namespace) -> np.ndarray:
    if getattr(args, "video", None) is not None:
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise SystemExit(f"Could not open video: {args.video}")
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise SystemExit("Could not read a frame from that video.")
        return frame
    if getattr(args, "source", None) is None:
        raise SystemExit("Pass --video or --source to calibrate.")
    capture = open_capture(args.source)
    frame = None
    for _ in range(30):  # let the camera settle and auto-expose
        ok, candidate = capture.read()
        if ok:
            frame = candidate
    capture.release()
    if frame is None:
        raise SystemExit(f"No frames from source {args.source}.")
    return frame


# --------------------------------------------------------------------------
# capture and writing helpers
# --------------------------------------------------------------------------

def open_capture(source: str) -> cv2.VideoCapture:
    parsed: int | str = int(source) if str(source).isdigit() else str(source)
    capture = cv2.VideoCapture(parsed)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source {source}. Try --source 1.")
    return capture


def fourcc(code: str) -> int:
    """Works on OpenCV 4.x and 5.x.

    5.x documents the codec constructor as VideoWriter.fourcc. The free-function
    alias VideoWriter_fourcc is not listed as removed in the migration guide, but
    this covers both spellings so the script does not depend on that.
    """
    legacy = getattr(cv2, "VideoWriter_fourcc", None)
    if legacy is not None:
        return legacy(*code)
    return cv2.VideoWriter.fourcc(*code)


def probe_video(path: Path) -> tuple[float, int, int]:
    """Frame rate and size, read in a way that survives OpenCV 5.

    In OpenCV 4.x an unsupported capture property returned 0, so the common
    `cap.get(CAP_PROP_FPS) or 30.0` idiom worked. OpenCV 5.0 returns -1 instead,
    which is truthy: the fallback never fires and fps silently becomes -1. That
    produces negative timestamps and a writer that refuses to open. Frame size is
    taken from an actual decoded frame rather than a property, for the same reason.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    reported = capture.get(cv2.CAP_PROP_FPS)
    fps = float(reported) if 0 < reported <= 240 else 30.0
    if fps != reported:
        print(f"Frame rate reported as {reported}; assuming {fps} fps instead.")
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {path}")
    height, width = frame.shape[:2]
    return fps, width, height


def open_writer(path: Path, fps: float, width: int, height: int):
    """Open a writer that actually works, instead of failing on every frame.

    The mpeg4 encoder needs even dimensions. Without this you get thousands of
    'Failed to write frame' warnings and an unusable file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    even_width, even_height = width - width % 2, height - height % 2
    writer = cv2.VideoWriter(str(path), fourcc("mp4v"),
                             float(fps), (even_width, even_height))
    if not writer.isOpened():
        fallback = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(fallback), fourcc("MJPG"),
                                 float(fps), (even_width, even_height))
        if not writer.isOpened():
            raise SystemExit(f"Could not open a video writer for {path}")
        print(f"mp4v unavailable, writing {fallback} instead")
        path = fallback
    return writer, (even_width, even_height), path


class RowWriter:
    """Exclusive CSV sink with a durable header and frequent checkpoints."""

    def __init__(self, path: Path, fields) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path.resolve()
        self.fields = list(fields)
        # Refuse to truncate a previous run, including a zero-byte failed file.
        try:
            self.handle = open(self.path, "x", newline="", encoding="utf-8")
        except FileExistsError as exc:
            raise SystemExit(
                f"Raw results already exist: {self.path}\n"
                "Use a new --output-dir. Existing results have been preserved."
            ) from exc
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields)
        self.count = 0
        try:
            self.writer.writeheader()
            self.flush()
            with self.path.open(newline="", encoding="utf-8") as check:
                if next(csv.reader(check), None) != self.fields:
                    raise OSError(f"CSV header read-back failed: {self.path}")
        except BaseException:
            self.handle.close()
            raise

    def append(self, row: dict) -> None:
        self.writer.writerow(row)
        self.count += 1
        if self.count % DETECTION_FLUSH_ROWS == 0:
            self.flush()

    def flush(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        # Notice truncation/replacement instead of continuing an expensive run.
        actual = self.path.stat()
        opened = os.fstat(self.handle.fileno())
        if ((actual.st_dev, actual.st_ino) != (opened.st_dev, opened.st_ino)
                or actual.st_size != self.handle.tell()):
            raise OSError(f"CSV was replaced or its size changed unexpectedly: {self.path}")

    def close(self) -> None:
        if not self.handle.closed:
            try:
                self.flush()
            finally:
                self.handle.close()


def check_csv_output(directory: Path) -> None:
    """Check a real CSV round trip before loading YOLO or processing frames."""
    try:
        with tempfile.TemporaryDirectory(prefix="_csv_check_", dir=directory) as tmp:
            path = Path(tmp) / "probe.csv"
            with path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame", "track_id"])
                writer.writerow([0, 1])
                handle.flush()
                os.fsync(handle.fileno())
            probe = pd.read_csv(path)
            if list(probe.columns) != ["frame", "track_id"] or probe.values.tolist() != [[0, 1]]:
                raise ValueError("CSV contents did not survive the write/read check")
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise SystemExit(
            f"CSV saving check failed in {directory.resolve()}: {exc}\n"
            "No video analysis was started. Check free disk space and use a writable "
            "local output folder, for example under Movies."
        ) from exc
    print(f"CSV saving check passed. Results: {directory.resolve()}", flush=True)


def read_saved_rows(sink: RowWriter) -> pd.DataFrame:
    """Only postprocess a CSV whose schema and row count match this run."""
    try:
        if sink.path.stat().st_size == 0:
            raise ValueError("file is zero bytes")
        rows = pd.read_csv(sink.path)
        if list(rows.columns) != sink.fields:
            raise ValueError("CSV header does not match the expected columns")
        if len(rows) != sink.count:
            raise ValueError(f"expected {sink.count} rows, found {len(rows)}")
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise SystemExit(
            f"Could not read saved detections: {sink.path}\n"
            f"Recorded observations: {sink.count}. Read-back problem: {exc}\n"
            "Raw files have been preserved. Do not rerun into this same output folder."
        ) from exc
    return rows


def run_record(args: argparse.Namespace) -> None:
    capture = open_capture(args.source)
    if args.width and args.height:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ok, frame = capture.read()
    if not ok:
        raise SystemExit("Source opened but delivered no frames.")
    height, width = frame.shape[:2]
    out = args.out or Path("footage") / f"{datetime.now():%Y-%m-%d_%H%M%S}.mp4"
    writer, size, out = open_writer(out, args.fps, width, height)
    print(f"Recording {size[0]}x{size[1]} to {out}. Press q in the window to stop.")

    frames, misses, start = 0, 0, time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                misses += 1
                if misses > 60:
                    print("Source stopped delivering frames.")
                    break
                time.sleep(0.02)
                continue
            misses = 0
            writer.write(frame[:size[1], :size[0]])
            frames += 1
            preview = cv2.resize(frame, None, fx=0.5, fy=0.5)
            elapsed = time.perf_counter() - start
            cv2.putText(preview, f"REC {elapsed:6.1f}s  {frames} frames",
                        (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Recording", preview)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
        elapsed = max(time.perf_counter() - start, 0.001)
        print(f"Wrote {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} fps). "
              f"Set --fps {frames / elapsed:.0f} next time for correct playback speed.")


# --------------------------------------------------------------------------
# appearance and team assignment
# --------------------------------------------------------------------------

def jersey_colour(frame: np.ndarray, xyxy: np.ndarray) -> tuple[float, float, float] | None:
    """Robust LAB colour from the central bib/shirt area.

    The crop deliberately avoids the head, arms, shorts and box edges. Three
    horizontal bands vote independently, making one shadow or printed logo less
    likely to change the team assignment.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy.astype(int)
    x1, x2 = np.clip([x1, x2], 0, width - 1)
    y1, y2 = np.clip([y1, y2], 0, height - 1)
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 8 or box_h < 16:
        return None

    rx1, rx2 = x1 + int(0.30 * box_w), x1 + int(0.70 * box_w)
    ry1, ry2 = y1 + int(0.18 * box_h), y1 + int(0.50 * box_h)
    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None

    lab_image = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    band_colours = []
    for band in np.array_split(lab_image, 3, axis=0):
        pixels = band.reshape(-1, 3)
        usable = pixels[(pixels[:, 0] > 12) & (pixels[:, 0] < 248)]
        if len(usable) >= 8:
            band_colours.append(np.median(usable, axis=0))
    if not band_colours:
        return None
    return tuple(np.median(np.asarray(band_colours), axis=0).astype(float))


def reference_lab(name: str) -> np.ndarray:
    """Convert a named shirt colour to OpenCV LAB for explicit-team mode."""
    bgr = np.uint8([[REFERENCE_BGR[name]]])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_a = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_b = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(area_a + area_b - intersection, 1e-6)


def deduplicate_people(detections: list[tuple], threshold: float) -> list[tuple]:
    """Keep the strongest box when two tracked person boxes almost coincide."""
    kept = []
    for item in sorted(detections, key=lambda row: float(row[2]), reverse=True):
        if all(box_iou(item[0], previous[0]) < threshold for previous in kept):
            kept.append(item)
    return kept


def colour_name(lab_colour: np.ndarray) -> str:
    """Broad bib-colour name for a LAB centroid. Display only."""
    pixel = np.uint8([[np.clip(lab_colour, 0, 255)]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)
    hue, saturation, value = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    lightness = float(lab_colour[0])
    hue, saturation, value = int(hue), int(saturation), int(value)
    if lightness < 85 or value < 70:
        return "Black"
    if lightness > 175 and saturation < 85:
        return "White"
    if hue < 7 or hue >= 173:
        return "Red"
    if hue < 19:
        return "Orange"
    if hue < 36:
        return "Yellow"
    if hue < 85:
        return "Green"
    if hue < 100:
        return "Cyan"
    if hue < 130:
        return "Blue"
    if hue < 155:
        return "Purple"
    return "Pink"


class TeamClassifier:
    """Two shirt clusters with stable integer identities across refits.

    Teams are 0 and 1 for the whole session. Names are cosmetic. Refits are
    matched to the previous centres so a refit can never swap the meaning of
    team 0, which is what silently corrupts accumulated statistics.
    """

    def __init__(self, min_samples: int, team_a: str | None = None,
                 team_b: str | None = None, max_distance: float = 95.0,
                 vote_window: int = 25, switch_threshold: float = 0.85,
                 initial_votes: int = 4) -> None:
        if (team_a is None) != (team_b is None):
            raise SystemExit("Use --team-a and --team-b together, or omit both.")
        if team_a is not None and team_a == team_b:
            raise SystemExit("--team-a and --team-b must be different colours.")
        self.min_samples = min_samples
        self.explicit_names = [team_a, team_b] if team_a is not None else None
        self.max_distance = max_distance
        self.initial_votes = initial_votes
        self.switch_threshold = switch_threshold
        self.samples: dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self.votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=vote_window))
        self.assigned: dict[int, int] = {}
        self.last_seen: dict[int, int] = {}
        self.centres = (np.asarray([reference_lab(team_a), reference_lab(team_b)])
                        if team_a is not None else None)
        self.names = list(self.explicit_names) if self.explicit_names else ["Team 0", "Team 1"]

    def reset(self) -> None:
        self.samples.clear()
        self.votes.clear()
        self.assigned.clear()
        self.last_seen.clear()
        self.centres = (np.asarray([reference_lab(name) for name in self.explicit_names])
                        if self.explicit_names else None)
        self.names = list(self.explicit_names) if self.explicit_names else ["Team 0", "Team 1"]

    def observe(self, track_id: int, colour, frame_number: int = 0) -> None:
        if colour is not None:
            self.samples[track_id].append(np.asarray(colour, dtype=float))
        self.last_seen[track_id] = frame_number

    def prune(self, frame_number: int, max_age: int = 900) -> None:
        """Drop tracks not seen for a while so memory does not grow all match."""
        stale = [tid for tid, seen in self.last_seen.items()
                 if frame_number - seen > max_age]
        for tid in stale:
            self.samples.pop(tid, None)
            self.votes.pop(tid, None)
            self.assigned.pop(tid, None)
            self.last_seen.pop(tid, None)

    def update(self) -> None:
        stable = [np.median(values, axis=0) for values in self.samples.values()
                  if len(values) >= self.min_samples]
        if len(stable) < 2:
            return
        centres = KMeans(n_clusters=2, random_state=42,
                         n_init=10).fit(np.asarray(stable)).cluster_centers_

        if self.centres is None:
            # First fit only: darker shirt becomes team 0, purely for determinism.
            centres = centres[np.argsort(centres[:, 0])]
        else:
            # Later fits: keep whichever ordering stays closest to the previous
            # centres. Sorting by lightness alone breaks for red vs blue bibs,
            # which have almost identical L.
            direct = float(np.linalg.norm(centres - self.centres, axis=1).sum())
            swapped = float(np.linalg.norm(centres[::-1] - self.centres, axis=1).sum())
            if swapped < direct:
                centres = centres[::-1]

        self.centres = centres
        names = list(self.explicit_names) if self.explicit_names else [colour_name(centre) for centre in centres]
        if names[0] == names[1]:
            names = [f"{names[0]} A", f"{names[1]} B"]
        self.names = names

    def classify(self, track_id: int, colour) -> int:
        if self.centres is None or colour is None:
            return self.assigned.get(track_id, UNKNOWN_TEAM)
        distances = np.linalg.norm(self.centres - np.asarray(colour, dtype=float), axis=1)
        if self.explicit_names and float(np.min(distances)) > self.max_distance:
            return self.assigned.get(track_id, UNKNOWN_TEAM)
        self.votes[track_id].append(int(np.argmin(distances)))

        votes = list(self.votes[track_id])
        counts = np.bincount(votes, minlength=2)
        winner = int(np.argmax(counts))
        current = self.assigned.get(track_id)
        if current is None and len(votes) >= self.initial_votes:
            self.assigned[track_id] = winner
        elif winner != current and counts[winner] / len(votes) >= self.switch_threshold:
            # Hysteresis: switching away from an existing label needs a
            # supermajority, but it stays possible, unlike a hard lock.
            self.assigned[track_id] = winner
        return self.assigned.get(track_id, UNKNOWN_TEAM)

    def name(self, team: int) -> str:
        if team in (0, 1) and self.centres is not None:
            return self.names[team]
        return "Unknown"

    def ready(self) -> bool:
        return self.centres is not None


# --------------------------------------------------------------------------
# ball and possession
# --------------------------------------------------------------------------

class BallTracker:
    """Rejects ball detections that teleport, which cause phantom possessions."""

    def __init__(self, max_step_px: float = 60.0, miss_tolerance: int = 20) -> None:
        self.max_step_px = max_step_px
        self.miss_tolerance = miss_tolerance
        self.position: np.ndarray | None = None
        self.frames_missing = 0

    def update(self, candidates: list[tuple[np.ndarray, float]]) -> np.ndarray | None:
        if not candidates:
            return self._miss()
        if self.position is not None:
            limit = self.max_step_px * (self.frames_missing + 1)
            near = [c for c in candidates
                    if float(np.linalg.norm(c[0] - self.position)) <= limit]
        else:
            near = candidates
        if not near:
            return self._miss()
        best = max(near, key=lambda item: item[1])
        self.position = best[0]
        self.frames_missing = 0
        return self.position

    def _miss(self) -> None:
        self.frames_missing += 1
        if self.frames_missing > self.miss_tolerance:
            self.position = None
        return None


class PossessionTracker:
    """Owner confirmation, possession clock, and pass/turnover counts by team index."""

    def __init__(self, radius: float, confirm_frames: int = 5,
                 min_separation: float = 2.0, stale_seconds: float = 5.0) -> None:
        self.radius = radius
        self.min_separation = min_separation
        self.stale_seconds = stale_seconds
        self.window: deque = deque(maxlen=confirm_frames)
        self.owner_id: int | None = None
        self.owner_team: int | None = None
        self.owner_position: np.ndarray | None = None
        self.last_contact: float | None = None
        self.seconds = defaultdict(float)
        self.passes = defaultdict(int)
        self.turnovers = 0
        self.rejected_id_switches = 0

    def update(self, ball_xy, players: list[dict], dt: float, now: float) -> None:
        candidate = None
        if ball_xy is not None and players:
            scored = [
                (float(np.linalg.norm(ball_xy - p["foot"])) / max(p["height"], 1.0), p)
                for p in players if p["team"] != UNKNOWN_TEAM
            ]
            if scored:
                distance, nearest = min(scored, key=lambda item: item[0])
                if distance <= self.radius:
                    candidate = nearest

        self.window.append(candidate["track_id"] if candidate else None)
        confirmed = (candidate is not None
                     and len(self.window) == self.window.maxlen
                     and len(set(self.window)) == 1)

        if confirmed:
            self.last_contact = now
            if self.owner_id is None:
                self._take(candidate)
            elif candidate["track_id"] != self.owner_id:
                separation = float(np.linalg.norm(
                    candidate["foot"] - self.owner_position)) / max(candidate["height"], 1.0)
                if separation >= self.min_separation:
                    if candidate["team"] == self.owner_team:
                        self.passes[self.owner_team] += 1
                    else:
                        self.turnovers += 1
                else:
                    # Same spot, different track ID. That is the tracker losing the
                    # player, not a pass.
                    self.rejected_id_switches += 1
                self._take(candidate)
            else:
                self.owner_position = candidate["foot"]

        # Keep crediting the last confirmed team while the ball is in flight,
        # otherwise possession share is systematically under-counted.
        if (self.owner_team is not None and self.last_contact is not None
                and now - self.last_contact <= self.stale_seconds):
            self.seconds[self.owner_team] += dt

    def _take(self, player: dict) -> None:
        self.owner_id = player["track_id"]
        self.owner_team = player["team"]
        self.owner_position = player["foot"].copy()

    def shares(self) -> tuple[float, float] | None:
        total = self.seconds[0] + self.seconds[1]
        if total <= 0:
            return None
        return 100 * self.seconds[0] / total, 100 * self.seconds[1] / total


# --------------------------------------------------------------------------
# shared detection step
# --------------------------------------------------------------------------

def detect(model: YOLO, frame: np.ndarray, args: argparse.Namespace,
           calibration: Calibration | None, classifier: TeamClassifier,
           frame_number: int) -> tuple[list[dict], list[tuple[np.ndarray, float]]]:
    classes = [PERSON_CLASS, BALL_CLASS] if args.ball else [PERSON_CLASS]
    threshold = min(args.confidence, args.ball_confidence) if args.ball else args.confidence
    result = model.track(
        frame, persist=True, classes=classes, conf=threshold,
        tracker=args.tracker, imgsz=args.imgsz, device=args.device, iou=args.iou,
        verbose=False,
    )[0]

    players: list[dict] = []
    balls: list[tuple[np.ndarray, float]] = []
    if result.boxes is None:
        return players, balls

    boxes = result.boxes.xyxy.cpu().numpy()
    ids = (result.boxes.id.int().cpu().tolist() if result.boxes.id is not None
           else [-1] * len(boxes))
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.int().cpu().tolist()

    raw_people = []
    for box, track_id, confidence, class_id in zip(boxes, ids, confidences, class_ids):
        x1, y1, x2, y2 = box
        if class_id == BALL_CLASS:
            if confidence >= args.ball_confidence:
                balls.append((np.array([(x1 + x2) / 2, (y1 + y2) / 2]), float(confidence)))
            continue
        if confidence < args.confidence or track_id < 0:
            continue
        raw_people.append((box, int(track_id), float(confidence)))

    for box, track_id, confidence in deduplicate_people(raw_people, args.duplicate_iou):
        x1, y1, x2, y2 = box
        foot = np.array([(x1 + x2) / 2, float(y2)])
        pitch_xy = None
        if calibration is not None:
            pitch_xy = calibration.to_pitch([foot])[0]
            if not calibration.on_pitch(pitch_xy):
                continue  # sideline watchers, dog walkers, parked cars

        colour = jersey_colour(frame, box)
        classifier.observe(track_id, colour, frame_number)
        players.append({
            "track_id": track_id,
            "box": box.astype(float),
            "foot": foot,
            "height": float(y2 - y1),
            "confidence": float(confidence),
            "colour": colour,
            "pitch": pitch_xy,
            "team": UNKNOWN_TEAM,
        })
    return players, balls


class MotionState:
    """Per-track distance and smoothed speed for the LIVE view only.

    This is deliberately a rough online estimate: it cannot smooth positions it
    has not seen yet. The analyse command re-derives distance offline with a
    proper filter, and that is the number to report.
    """

    def __init__(self, calibrated: bool, min_speed: float = 0.0) -> None:
        self.calibrated = calibrated
        self.min_speed = min_speed if calibrated else 0.0
        self.state: dict[int, dict] = {}

    def update(self, track_id: int, position: np.ndarray, height_px: float,
               now: float) -> tuple[float, float]:
        entry = self.state.get(track_id)
        if entry is None:
            self.state[track_id] = {"position": position.copy(), "time": now,
                                    "distance": 0.0, "first_seen": now,
                                    "speeds": deque(maxlen=7)}
            return 0.0, 0.0

        dt = max(now - entry["time"], 1e-4)
        step = float(np.linalg.norm(position - entry["position"]))
        speed = step / dt
        plausible = (speed <= MAX_PLAUSIBLE_SPEED_MS if self.calibrated
                     else step <= max(height_px * 1.5, 50.0))
        if plausible and speed >= self.min_speed:
            entry["distance"] += step
            entry["speeds"].append(speed)
        else:
            entry["speeds"].append(0.0)
        entry["position"] = position.copy()
        entry["time"] = now
        smoothed = float(np.median(entry["speeds"])) if entry["speeds"] else 0.0
        return smoothed, entry["distance"]


# --------------------------------------------------------------------------
# live view
# --------------------------------------------------------------------------

def run_live(args: argparse.Namespace) -> None:
    calibration = Calibration.load(args.calibration)
    capture = open_capture(args.source)
    model = YOLO(args.model)
    classifier = TeamClassifier(args.team_min_samples, args.team_a, args.team_b,
                                args.team_max_distance)
    ball_tracker = BallTracker()
    possession = PossessionTracker(args.possession_radius,
                                   min_separation=args.pass_separation)
    motion = MotionState(calibrated=calibration is not None, min_speed=args.min_speed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    writer, writer_size = None, None
    frame_number, misses = 0, 0
    start = time.perf_counter()
    previous = 0.0
    processing_times: deque = deque(maxlen=30)

    if calibration is None:
        print("No calibration found. Distances and speeds will be in PIXELS.")
        print(f"Run the calibrate command to write {args.calibration}.")
    print("Live view: q = quit, r = relearn shirts, s = screenshot")
    print("Live distances are unsmoothed and will read high. Use analyse for data.")
    if args.ball:
        print("Ball metrics are experimental. Do not report them as data.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                misses += 1
                if misses > 60:
                    print("Live source stopped delivering frames.")
                    break
                if (cv2.waitKey(10) & 0xFF) == ord("q"):
                    break
                continue
            misses = 0
            frame_number += 1

            if calibration is not None and frame_number == 1:
                warning = calibration.size_warning(frame)
                if warning:
                    print(warning)

            frame_start = time.perf_counter()
            players, balls = detect(model, frame, args, calibration, classifier, frame_number)

            if frame_number % 10 == 0:
                classifier.update()
            if frame_number % 300 == 0:
                classifier.prune(frame_number)

            now = time.perf_counter() - start
            dt = max(0.0, now - previous)
            previous = now

            ball_xy = ball_tracker.update(balls) if args.ball else None
            if ball_xy is not None:
                centre = tuple(ball_xy.astype(int))
                cv2.circle(frame, centre, 10, (0, 0, 255), 3)
                cv2.putText(frame, "BALL", (centre[0] + 12, centre[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            counts = defaultdict(int)
            for player in players:
                player["team"] = classifier.classify(player["track_id"], player["colour"])
                counts[player["team"]] += 1

                position = player["pitch"] if player["pitch"] is not None else player["foot"]
                speed, distance = motion.update(player["track_id"], np.asarray(position),
                                                player["height"], now)
                lab = player["colour"] or (np.nan, np.nan, np.nan)
                x1, y1, x2, y2 = player["box"]
                rows.append({
                    "elapsed_seconds": now,
                    "frame": frame_number,
                    "track_id": player["track_id"],
                    "team_index": player["team"],
                    "team_name": classifier.name(player["team"]),
                    "confidence": player["confidence"],
                    "lab_l": lab[0], "lab_a": lab[1], "lab_b": lab[2],
                    "image_x": float((x1 + x2) / 2), "image_y": float(y2),
                    "pitch_x": float(player["pitch"][0]) if player["pitch"] is not None else np.nan,
                    "pitch_y": float(player["pitch"][1]) if player["pitch"] is not None else np.nan,
                    "speed": speed,
                    "distance": distance,
                    "units": "metres" if calibration is not None else "pixels",
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                })

                bgr = DISPLAY_TEAM_COLOURS.get(player["team"], UNKNOWN_COLOUR)
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                unit = "m/s" if calibration is not None else "px/s"
                cv2.rectangle(frame, p1, p2, bgr, 2)
                cv2.putText(frame, f"#{player['track_id']} {speed:.1f}{unit}",
                            (p1[0], max(18, p1[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, bgr, 2, cv2.LINE_AA)

            if args.ball:
                possession.update(ball_xy, players, dt, now)

            processing_times.append(time.perf_counter() - frame_start)
            draw_panel(frame, classifier, counts, possession, args,
                       now, processing_times, calibration, ball_xy)

            if args.record_live:
                if writer is None:
                    height, width = frame.shape[:2]
                    fps_guess = 1 / max(float(np.mean(processing_times)), 0.05)
                    writer, writer_size, _ = open_writer(
                        args.output_dir / "live_annotated.mp4", fps_guess, width, height)
                writer.write(frame[:writer_size[1], :writer_size[0]])

            cv2.imshow("Soccer Live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                classifier.reset()
                print("Shirt learning reset.")
            if key == ord("s"):
                path = args.output_dir / f"live_{int(time.time())}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Saved screenshot: {path}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        save_live_outputs(args, rows, classifier, possession, calibration)


def draw_panel(frame, classifier, counts, possession, args, now,
               processing_times, calibration, ball_xy) -> None:
    live_fps = 1 / max(float(np.mean(processing_times)), 1e-4)
    lines = [
        f"LIVE {now:6.1f}s   {live_fps:4.1f} fps   "
        f"{'metres' if calibration else 'PIXELS (uncalibrated)'}",
        f"On pitch: {counts[0] + counts[1] + counts[UNKNOWN_TEAM]}",
    ]
    if classifier.ready():
        lines.append(f"{classifier.name(0)}: {counts[0]}    "
                     f"{classifier.name(1)}: {counts[1]}    "
                     f"unknown: {counts[UNKNOWN_TEAM]}")
    else:
        lines.append("Learning the two shirt colours...")

    if args.ball:
        lines.append(f"Ball: {'tracked' if ball_xy is not None else 'not visible'}")
        shares = possession.shares()
        if shares and classifier.ready():
            lines.append(f"Possession: {classifier.name(0)} {shares[0]:.0f}%   "
                         f"{classifier.name(1)} {shares[1]:.0f}%")
            lines.append(f"Est. passes: {possession.passes[0]} / {possession.passes[1]}   "
                         f"turnovers: {possession.turnovers}")
            lines.append(f"Rejected ID switches: {possession.rejected_id_switches}")
        else:
            lines.append("Possession: waiting for the ball")

    panel = frame.copy()
    height = 26 + 22 * len(lines)
    # OpenCV 5 renders FONT_HERSHEY_* through a new engine with different metrics,
    # so measure the text rather than hardcoding a panel width.
    widest = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                 for line in lines)
    cv2.rectangle(panel, (12, 12), (36 + widest, height), (10, 10, 10), -1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (24, 36 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)


def save_live_outputs(args, rows, classifier, possession, calibration) -> None:
    if not rows:
        print("No detections recorded.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.output_dir / "live_player_tracks.csv"
    pd.DataFrame(rows).to_csv(frame_path, index=False)
    print(f"Saved per-frame tracks: {frame_path}")

    if args.ball and classifier.ready():
        units = "metres" if calibration is not None else "pixels"
        summary = pd.DataFrame([{
            "team_index": team,
            "team_name": classifier.name(team),
            "possession_seconds": possession.seconds[team],
            "estimated_passes": possession.passes[team],
            "match_turnovers": possession.turnovers,
            "rejected_id_switches": possession.rejected_id_switches,
            "units": units,
            "note": "Live estimates. Use the analyse command for reportable data.",
        } for team in (0, 1)])
        path = args.output_dir / "live_metrics_summary.csv"
        summary.to_csv(path, index=False)
        print(f"Saved live summary: {path}")


# --------------------------------------------------------------------------
# motion: smoothing, segmentation and distance
# --------------------------------------------------------------------------

def _smooth_series(values: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay with the window clamped to what the segment can support."""
    count = len(values)
    if count < 3 or window < 3:
        return values
    window = min(window, count)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    order = min(polyorder, window - 1)
    return savgol_filter(values, window_length=window, polyorder=order)


def add_motion_columns(tracks: pd.DataFrame, calibrated: bool, fps: float = 30.0,
                       smoothing_seconds: float = 0.5, min_speed: float = 0.20,
                       max_gap_seconds: float = 1.0) -> pd.DataFrame:
    """Smooth each track, then differentiate. Order matters, see module docstring.

    Also reports the unsmoothed distance as distance_raw so the size of the
    jitter correction stays visible instead of being quietly absorbed.
    """
    x_col, y_col = ("pitch_x", "pitch_y") if calibrated else ("image_x", "image_y")
    tracks = tracks.sort_values(["track_id", "time_seconds"]).copy()

    # A long gap means the tracker lost the player. Splitting on it stops the
    # straight-line jump across the gap being counted as running.
    gap = tracks.groupby("track_id")["time_seconds"].diff()
    new_segment = (gap.isna()) | (gap > max_gap_seconds)
    tracks["segment"] = new_segment.groupby(tracks["track_id"]).cumsum().astype(int)

    window = int(round(smoothing_seconds * fps)) if smoothing_seconds > 0 else 0
    smooth_x = np.empty(len(tracks), dtype=float)
    smooth_y = np.empty(len(tracks), dtype=float)
    position = 0
    for _, segment in tracks.groupby(["track_id", "segment"], sort=False):
        size = len(segment)
        raw_x = segment[x_col].to_numpy(dtype=float)
        raw_y = segment[y_col].to_numpy(dtype=float)
        if window >= 3 and np.isfinite(raw_x).all() and np.isfinite(raw_y).all():
            smooth_x[position:position + size] = _smooth_series(raw_x, window)
            smooth_y[position:position + size] = _smooth_series(raw_y, window)
        else:
            smooth_x[position:position + size] = raw_x
            smooth_y[position:position + size] = raw_y
        position += size

    ordered = tracks.groupby(["track_id", "segment"], sort=False).cumcount()  # noqa: F841
    tracks = tracks.sort_values(["track_id", "segment", "time_seconds"])
    tracks["smooth_x"] = smooth_x
    tracks["smooth_y"] = smooth_y

    grouped = tracks.groupby(["track_id", "segment"], sort=False)
    dt = grouped["time_seconds"].diff()
    step = np.hypot(grouped["smooth_x"].diff(), grouped["smooth_y"].diff())
    raw_step = np.hypot(grouped[x_col].diff(), grouped[y_col].diff())
    speed = step / dt.replace(0, np.nan)

    ceiling = MAX_PLAUSIBLE_SPEED_MS if calibrated else np.inf
    floor = min_speed if calibrated else 0.0
    valid = speed.notna() & (speed <= ceiling) & (speed >= floor)
    tracks["speed"] = speed.where(valid, 0.0).fillna(0.0)
    tracks["step"] = step.where(valid, 0.0).fillna(0.0)
    tracks["step_raw"] = raw_step.where(speed.notna() & (speed <= ceiling), 0.0).fillna(0.0)
    tracks["distance"] = tracks.groupby("track_id")["step"].cumsum()
    tracks["distance_raw"] = tracks.groupby("track_id")["step_raw"].cumsum()
    tracks["units"] = "metres" if calibrated else "pixels"
    return tracks.sort_values(["track_id", "time_seconds"])


def summarise(tracks: pd.DataFrame, calibrated: bool) -> pd.DataFrame:
    x_col, y_col = ("pitch_x", "pitch_y") if calibrated else ("image_x", "image_y")
    summary = (
        tracks.groupby(["team_index", "team_name", "track_id"])
        .agg(first_seen=("time_seconds", "min"),
             last_seen=("time_seconds", "max"),
             detections=("frame", "size"),
             distance=("step", "sum"),
             distance_raw=("step_raw", "sum"),
             top_speed=("speed", lambda s: float(s.quantile(0.95))),
             mean_x=(x_col, "mean"),
             mean_y=(y_col, "mean"),
             median_confidence=("confidence", "median"))
        .reset_index()
    )
    summary["tracked_seconds"] = summary["last_seen"] - summary["first_seen"]
    # How much of the raw total was jitter. Large values mean noisy tracking,
    # not a hard-working player.
    summary["jitter_removed_percent"] = np.where(
        summary["distance_raw"] > 0,
        100.0 * (1.0 - summary["distance"] / summary["distance_raw"]), 0.0)
    summary["units"] = "metres" if calibrated else "pixels"
    # 95th percentile rather than the max, because the single fastest frame is
    # almost always a tracking error rather than a sprint.
    return summary.sort_values(["team_index", "distance"], ascending=[True, False])


# --------------------------------------------------------------------------
# batch analysis
# --------------------------------------------------------------------------

def run_analyse(args: argparse.Namespace) -> None:
    if not args.video.exists():
        raise SystemExit(f"Video does not exist: {args.video}")
    calibration = Calibration.load(args.calibration)
    if calibration is None:
        print("No calibration found. Output will be in PIXELS, not metres.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Fail before inference if this directory belongs to an earlier run.
    for name in ("_raw_detections.csv", "_raw_ball.csv", "player_tracks.csv",
                 "player_summary.csv", "tracking_quality.csv", "annotated_match.mp4",
                 "annotated_match.avi"):
        if (args.output_dir / name).exists():
            raise SystemExit(
                f"Existing results found in {args.output_dir.resolve()}. "
                "Choose a new --output-dir to preserve them."
            )
    check_csv_output(args.output_dir)

    fps, width, height = probe_video(args.video)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    model = YOLO(args.model)
    classifier = TeamClassifier(args.team_min_samples, args.team_a, args.team_b,
                                args.team_max_distance)

    detections = RowWriter(args.output_dir / "_raw_detections.csv", DETECTION_FIELDS)
    ball_rows = RowWriter(args.output_dir / "_raw_ball.csv", BALL_FIELDS) if args.ball else None
    frames_processed = 0
    frame_number = -1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if args.max_seconds is not None and frame_number / fps > args.max_seconds:
                break
            if frame_number % args.sample_every:
                continue

            if calibration is not None and frames_processed == 0:
                warning = calibration.size_warning(frame)
                if warning:
                    print(warning)
            frames_processed += 1

            players, balls = detect(model, frame, args, calibration, classifier,
                                    frame_number)
            timestamp = frame_number / fps
            for player in players:
                lab = player["colour"] if player["colour"] is not None else (np.nan,) * 3
                x1, y1, x2, y2 = player["box"]
                detections.append({
                    "frame": frame_number,
                    "time_seconds": timestamp,
                    "track_id": player["track_id"],
                    "confidence": player["confidence"],
                    "lab_l": lab[0], "lab_a": lab[1], "lab_b": lab[2],
                    "image_x": float((x1 + x2) / 2), "image_y": float(y2),
                    "pitch_x": (float(player["pitch"][0])
                                if player["pitch"] is not None else ""),
                    "pitch_y": (float(player["pitch"][1])
                                if player["pitch"] is not None else ""),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                })
            if ball_rows is not None:
                for position, confidence in balls:
                    ball_rows.append({
                        "frame": frame_number, "time_seconds": timestamp,
                        "ball_x": float(position[0]), "ball_y": float(position[1]),
                        "confidence": float(confidence),
                    })
            if frame_number % 300 == 0:
                detections.flush()
                if ball_rows is not None:
                    ball_rows.flush()
                print(f"  frame {frame_number} ({timestamp:6.1f}s), "
                      f"{detections.count} detections; "
                      f"{detections.path.stat().st_size:,} CSV bytes saved", flush=True)
    finally:
        capture.release()
        try:
            detections.close()
        finally:
            if ball_rows is not None:
                ball_rows.close()

    if detections.count == 0:
        raise SystemExit("No players found. Try --confidence 0.20 or a larger --imgsz.")

    tracks = read_saved_rows(detections)
    tracks = assign_teams_offline(tracks, args.team_a, args.team_b,
                                  args.team_max_distance)
    tracks = add_motion_columns(
        tracks, calibrated=calibration is not None,
        fps=fps / max(args.sample_every, 1),
        smoothing_seconds=args.smoothing_seconds,
        min_speed=args.min_speed,
        max_gap_seconds=args.max_gap_seconds,
    )
    tracks.to_csv(args.output_dir / "player_tracks.csv", index=False)

    summary = summarise(tracks, calibrated=calibration is not None)
    summary.to_csv(args.output_dir / "player_summary.csv", index=False)

    possession_summary = None
    ball_stats = {}
    if args.ball:
        balls = read_saved_rows(ball_rows) if ball_rows.count else pd.DataFrame(
            columns=list(BALL_FIELDS))
        possession_summary, ball_stats = possession_offline(
            tracks, balls, frames_processed, fps / max(args.sample_every, 1), args)
        if possession_summary is not None:
            possession_summary.to_csv(
                args.output_dir / "possession_summary.csv", index=False)

    quality = quality_report(tracks, summary, fps, calibration is not None,
                             args, ball_stats)
    quality.to_csv(args.output_dir / "tracking_quality.csv", index=False)

    print(f"Raw detections kept: {detections.path}", flush=True)

    print(f"Saved tracks:  {args.output_dir / 'player_tracks.csv'}")
    print(f"Saved summary: {args.output_dir / 'player_summary.csv'}")
    print(f"Saved quality: {args.output_dir / 'tracking_quality.csv'}")
    if possession_summary is not None:
        print(f"Saved ball:    {args.output_dir / 'possession_summary.csv'}")
    print(summary.to_string(index=False))
    if possession_summary is not None:
        print("\nBall and possession")
        print(possession_summary.to_string(index=False))
    print("\nTracking quality")
    print(quality.to_string(index=False))

    if not args.no_video:
        path = write_annotated_video(args, tracks, fps, width, height)
        print(f"Saved video:   {path}")
    print("Team 0/1 are colour clusters. Check the video before trusting them.")


def possession_offline(tracks: pd.DataFrame, balls: pd.DataFrame,
                       frames_processed: int, fps: float,
                       args: argparse.Namespace) -> tuple[pd.DataFrame | None, dict]:
    """Replay stored detections through the live possession logic.

    Teams are only known after the offline clustering pass, so possession cannot
    be computed during detection. Replaying the stored positions gives the same
    answer without running the detector twice.
    """
    detected_frames = int(balls["frame"].nunique()) if len(balls) else 0
    rate = 100.0 * detected_frames / max(frames_processed, 1)
    stats = {"ball_frames_detected": detected_frames,
             "ball_detection_rate_percent": rate}
    if detected_frames == 0:
        print("\nNo ball was detected in any frame. Possession was not computed. "
              "COCO's sports-ball class often misses a football on a wide view.")
        return None, stats

    ball_tracker = BallTracker()
    possession = PossessionTracker(args.possession_radius,
                                   min_separation=args.pass_separation)
    balls_by_frame: dict[int, list] = defaultdict(list)
    for row in balls.itertuples(index=False):
        balls_by_frame[int(row.frame)].append(
            (np.array([float(row.ball_x), float(row.ball_y)]), float(row.confidence)))

    columns = ["frame", "track_id", "team_index", "image_x", "image_y", "y1", "y2"]
    ordered = tracks[columns].sort_values("frame", kind="mergesort")
    frame_column = ordered["frame"].to_numpy()
    values = ordered.drop(columns="frame").to_numpy(dtype=float)
    unique_frames = np.unique(frame_column)
    starts = np.searchsorted(frame_column, unique_frames, side="left")
    ends = np.searchsorted(frame_column, unique_frames, side="right")

    previous_frame = None
    for frame_id, start, end in zip(unique_frames, starts, ends):
        block = values[start:end]
        players = [{
            "track_id": int(row[0]),
            "team": int(row[1]),
            "foot": np.array([row[2], row[3]]),
            "height": float(row[5] - row[4]),
        } for row in block]
        dt = 0.0 if previous_frame is None else (frame_id - previous_frame) / fps
        now = float(frame_id) / fps
        previous_frame = frame_id
        ball_xy = ball_tracker.update(balls_by_frame.get(int(frame_id), []))
        possession.update(ball_xy, players, dt, now)

    shares = possession.shares()
    if shares is None:
        print("\nThe ball was detected but never stayed near one player long "
              "enough to confirm an owner. Possession was not computed.")
        return None, stats

    names = tracks.drop_duplicates("team_index").set_index("team_index")["team_name"]
    caveat = ("Reliable enough to discuss" if rate >= 40 else
              "LOW BALL DETECTION - do not quote these numbers")
    summary = pd.DataFrame([{
        "team_index": team,
        "team_name": names.get(team, f"Team {team}"),
        "possession_seconds": possession.seconds[team],
        "possession_percent": shares[team],
        "estimated_passes": possession.passes[team],
        "match_turnovers": possession.turnovers,
        "rejected_id_switches": possession.rejected_id_switches,
        "ball_detection_rate_percent": rate,
        "reliability": caveat,
    } for team in (0, 1)])
    return summary, stats


def quality_report(tracks: pd.DataFrame, summary: pd.DataFrame, fps: float,
                   calibrated: bool, args: argparse.Namespace | None = None,
                   ball_stats: dict | None = None) -> pd.DataFrame:
    """Create honest diagnostics before anyone interprets player statistics."""
    duration = float(tracks["time_seconds"].max() - tracks["time_seconds"].min())
    known = tracks[tracks["team_index"].isin([0, 1])]
    unknown_pct = 100.0 * float((tracks["team_index"] == UNKNOWN_TEAM).mean())
    short_pct = 100.0 * float((summary["detections"] < 10).mean())
    total_ids = int(summary["track_id"].nunique())
    longest = float(summary["tracked_seconds"].max()) if len(summary) else 0.0
    jitter = (float(summary["jitter_removed_percent"].median())
              if "jitter_removed_percent" in summary else float("nan"))

    warnings = []
    if total_ids > 100 or short_pct > 35:
        warnings.append("HIGH ID FRAGMENTATION - do not rank individual players")
    elif total_ids > 50:
        warnings.append("MODERATE ID FRAGMENTATION - inspect IDs before player totals")
    if np.isfinite(jitter) and jitter > 40:
        warnings.append("NOISY POSITIONS - most raw distance was jitter, "
                        "treat distances as indicative only")
    if not calibrated:
        warnings.append("UNCALIBRATED - values are pixels and are not comparable "
                        "between sessions")
    if args is not None and args.smoothing_seconds <= 0:
        warnings.append("SMOOTHING DISABLED - distances will read high")
    if ball_stats and ball_stats.get("ball_detection_rate_percent", 0) < 40:
        warnings.append("LOW BALL DETECTION - possession and passes are unreliable")

    record = {
        "pipeline_version": VERSION,
        "video_minutes": duration / 60.0,
        "processed_observations": int(len(tracks)),
        "temporary_track_ids": total_ids,
        "short_track_percent": short_pct,
        "unknown_team_percent": unknown_pct,
        "median_detection_confidence": float(tracks["confidence"].median()),
        "longest_track_seconds": longest,
        "processed_fps": float(fps),
        "calibrated": bool(calibrated),
        "median_jitter_removed_percent": jitter,
        "smoothing_seconds": float(args.smoothing_seconds) if args else float("nan"),
        "known_team_observations": int(len(known)),
    }
    record.update(ball_stats or {})
    record["warning"] = " | ".join(warnings) if warnings else "OK"
    return pd.DataFrame([record])


def assign_teams_offline(tracks: pd.DataFrame, team_a: str | None = None,
                         team_b: str | None = None,
                         max_distance: float = 95.0) -> pd.DataFrame:
    summaries = (
        tracks.groupby("track_id")
        .agg(lab_l=("lab_l", "median"), lab_a=("lab_a", "median"),
             lab_b=("lab_b", "median"), detections=("frame", "size"))
        .reset_index()
    )
    # Detections with no readable shirt colour are kept in the track data, so
    # positions are not lost, but they cannot take part in the clustering.
    usable = summaries[["lab_l", "lab_a", "lab_b"]].notna().all(axis=1)
    eligible = summaries[(summaries["detections"] >= 5) & usable].copy()
    if len(eligible) < 2:
        raise SystemExit("Too few stable tracks to identify two teams.")

    colours = eligible[["lab_l", "lab_a", "lab_b"]].to_numpy()
    if team_a is not None and team_b is not None:
        centres = np.asarray([reference_lab(team_a), reference_lab(team_b)])
        distances = np.linalg.norm(colours[:, None, :] - centres[None, :, :], axis=2)
        winners = np.argmin(distances, axis=1)
        eligible["team_index"] = [int(winner) if distances[index, winner] <= max_distance
                                  else UNKNOWN_TEAM
                                  for index, winner in enumerate(winners)]
        names = [team_a, team_b]
    else:
        model = KMeans(n_clusters=2, random_state=42, n_init=20).fit(colours)
        order = np.argsort(model.cluster_centers_[:, 0])  # dark shirt becomes team 0
        remap = {int(old): int(new) for new, old in enumerate(order)}
        eligible["team_index"] = [remap[int(label)] for label in model.labels_]
        names = [colour_name(model.cluster_centers_[old]) for old in order]
    if names[0] == names[1]:
        names = [f"{names[0]} A", f"{names[1]} B"]

    mapping = eligible.set_index("track_id")["team_index"]
    tracks["team_index"] = tracks["track_id"].map(mapping).fillna(UNKNOWN_TEAM).astype(int)
    tracks["team_name"] = tracks["team_index"].map(
        {0: names[0], 1: names[1], UNKNOWN_TEAM: "Unknown"})
    return tracks


def write_annotated_video(args, tracks: pd.DataFrame, fps: float,
                          width: int, height: int) -> Path:
    """Draw boxes without materialising one DataFrame per frame.

    A ninety-minute match is about 162,000 frames, and grouping the track table
    into that many small DataFrames up front costs more memory than the video.
    Sorting once and slicing numpy arrays by frame keeps it flat.
    """
    writer, size, path = open_writer(args.output_dir / "annotated_match.mp4",
                                     fps, width, height)
    ordered = tracks.sort_values("frame", kind="mergesort")
    frame_column = ordered["frame"].to_numpy()
    boxes = ordered[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    team_index = ordered["team_index"].to_numpy(dtype=int)
    track_ids = ordered["track_id"].to_numpy(dtype=int)
    team_names = ordered["team_name"].astype(str).to_numpy()

    capture = cv2.VideoCapture(str(args.video))
    frame_number = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            start = int(np.searchsorted(frame_column, frame_number, side="left"))
            end = int(np.searchsorted(frame_column, frame_number, side="right"))
            for index in range(start, end):
                x1, y1, x2, y2 = boxes[index]
                bgr = DISPLAY_TEAM_COLOURS.get(int(team_index[index]), UNKNOWN_COLOUR)
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, bgr, 2)
                cv2.putText(frame, f"{team_names[index]} #{int(track_ids[index])}",
                            (p1[0], max(18, p1[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, bgr, 2, cv2.LINE_AA)
            writer.write(frame[:size[1], :size[0]])
    finally:
        capture.release()
        writer.release()
    return path


# --------------------------------------------------------------------------
# multi-camera batch analysis and pitch-space fusion
# --------------------------------------------------------------------------

class DisjointTracks:
    """Small union-find that refuses to put two tracks from one camera together."""

    def __init__(self, nodes: list[tuple[str, int]]) -> None:
        self.parent = {node: node for node in nodes}
        self.cameras = {node: {node[0]} for node in nodes}

    def find(self, node: tuple[str, int]) -> tuple[str, int]:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: tuple[str, int], right: tuple[str, int]) -> bool:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return True
        # A component may contain no more than one local track from each camera.
        # This prevents two nearby teammates in one view being collapsed together.
        if self.cameras[root_left] & self.cameras[root_right]:
            return False
        if len(self.cameras[root_left]) < len(self.cameras[root_right]):
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.cameras[root_left] |= self.cameras[root_right]
        return True


def validate_multicam_args(args: argparse.Namespace) -> tuple[list[str], list[float]]:
    count = len(args.video)
    if count < 2:
        raise SystemExit("multicam needs at least two --video arguments.")
    if len(args.camera_calibration) != count:
        raise SystemExit("Provide one --camera-calibration for every --video.")
    if args.team_a is None or args.team_b is None:
        raise SystemExit("multicam requires --team-a and --team-b so all views agree.")
    if args.sample_every < 1:
        raise SystemExit("--sample-every must be at least 1.")
    if args.sync_tolerance <= 0 or args.fusion_distance <= 0:
        raise SystemExit("--sync-tolerance and --fusion-distance must be positive.")
    if args.fusion_min_overlap < 1:
        raise SystemExit("--fusion-min-overlap must be at least 1.")

    names = args.camera_name or [f"Camera_{index + 1}" for index in range(count)]
    offsets = args.camera_offset or [0.0] * count
    if len(names) != count:
        raise SystemExit("Provide one --camera-name for every --video, or omit all names.")
    if len(offsets) != count:
        raise SystemExit("Provide one --camera-offset for every --video, or omit all offsets.")
    if len(set(names)) != len(names):
        raise SystemExit("Every --camera-name must be unique.")

    missing_videos = [str(path) for path in args.video if not path.exists()]
    missing_calibrations = [str(path) for path in args.camera_calibration if not path.exists()]
    if missing_videos:
        raise SystemExit("Missing video(s): " + ", ".join(missing_videos))
    if missing_calibrations:
        raise SystemExit("Missing calibration(s): " + ", ".join(missing_calibrations))

    calibrations = [Calibration.load(path) for path in args.camera_calibration]
    dimensions = {(round(c.pitch_length, 3), round(c.pitch_width, 3))
                  for c in calibrations if c is not None}
    if len(dimensions) != 1:
        raise SystemExit(
            "All calibration files must use the same pitch length and width."
        )
    return names, [float(value) for value in offsets]


def analyse_multicam_view(args: argparse.Namespace, camera_index: int,
                          name: str, offset: float) -> pd.DataFrame:
    """Run the proven single-camera pipeline in an isolated camera directory."""
    camera_args = argparse.Namespace(**vars(args))
    camera_args.command = "analyse"
    camera_args.video = args.video[camera_index]
    camera_args.calibration = args.camera_calibration[camera_index]
    camera_args.output_dir = args.output_dir / f"camera_{camera_index + 1}_{name}"
    run_analyse(camera_args)

    path = camera_args.output_dir / "player_tracks.csv"
    tracks = pd.read_csv(path)
    tracks = tracks.rename(columns={"track_id": "local_track_id"})
    tracks.insert(0, "camera_index", camera_index)
    tracks.insert(1, "camera_name", name)
    tracks["camera_time_seconds"] = tracks["time_seconds"]
    tracks["global_time_seconds"] = tracks["time_seconds"] + offset
    tracks["camera_offset_seconds"] = offset
    return tracks


def synchronized_track_distance(left: pd.DataFrame, right: pd.DataFrame,
                                tolerance: float) -> tuple[int, float, float]:
    """Number of overlaps, median distance and 90th-percentile distance."""
    left_points = left[["global_time_seconds", "pitch_x", "pitch_y"]].dropna()
    right_points = right[["global_time_seconds", "pitch_x", "pitch_y"]].dropna()
    if left_points.empty or right_points.empty:
        return 0, float("inf"), float("inf")
    left_points = left_points.sort_values("global_time_seconds")
    right_points = right_points.sort_values("global_time_seconds").rename(
        columns={"pitch_x": "other_x", "pitch_y": "other_y"})
    paired = pd.merge_asof(
        left_points, right_points, on="global_time_seconds", direction="nearest",
        tolerance=tolerance,
    ).dropna(subset=["other_x", "other_y"])
    if paired.empty:
        return 0, float("inf"), float("inf")
    distances = np.hypot(
        paired["pitch_x"] - paired["other_x"],
        paired["pitch_y"] - paired["other_y"],
    )
    return len(distances), float(distances.median()), float(distances.quantile(0.90))


def associate_multicam_tracks(tracks: pd.DataFrame, args: argparse.Namespace
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Associate local IDs across views using synchronized pitch positions."""
    keys = [(str(camera), int(track_id))
            for (camera, track_id), _ in tracks.groupby(
                ["camera_name", "local_track_id"], sort=False)]
    union = DisjointTracks(keys)
    groups = {
        (str(camera), int(track_id)): group.sort_values("global_time_seconds")
        for (camera, track_id), group in tracks.groupby(
            ["camera_name", "local_track_id"], sort=False)
    }

    edges: list[dict] = []
    for index, left_key in enumerate(keys):
        left = groups[left_key]
        left_team = int(left["team_index"].mode().iloc[0])
        if left_team == UNKNOWN_TEAM:
            continue
        left_start, left_end = (float(left["global_time_seconds"].min()),
                                float(left["global_time_seconds"].max()))
        for right_key in keys[index + 1:]:
            if left_key[0] == right_key[0]:
                continue
            right = groups[right_key]
            right_team = int(right["team_index"].mode().iloc[0])
            if right_team != left_team:
                continue
            right_start, right_end = (float(right["global_time_seconds"].min()),
                                      float(right["global_time_seconds"].max()))
            if min(left_end, right_end) + args.sync_tolerance < max(left_start, right_start):
                continue
            overlap, median_distance, p90_distance = synchronized_track_distance(
                left, right, args.sync_tolerance)
            if (overlap >= args.fusion_min_overlap
                    and median_distance <= args.fusion_distance
                    and p90_distance <= args.fusion_distance * 2.0):
                edges.append({
                    "left_camera": left_key[0], "left_track_id": left_key[1],
                    "right_camera": right_key[0], "right_track_id": right_key[1],
                    "team_index": left_team, "overlap_observations": overlap,
                    "median_distance_metres": median_distance,
                    "p90_distance_metres": p90_distance,
                })

    # Strongest spatial matches claim components first.
    edges.sort(key=lambda edge: (-edge["overlap_observations"],
                                 edge["median_distance_metres"]))
    accepted: list[dict] = []
    for edge in edges:
        left_key = (edge["left_camera"], edge["left_track_id"])
        right_key = (edge["right_camera"], edge["right_track_id"])
        if union.union(left_key, right_key):
            accepted.append(edge)

    roots = sorted({union.find(key) for key in keys})
    root_to_global = {root: index + 1 for index, root in enumerate(roots)}
    mapping = {key: root_to_global[union.find(key)] for key in keys}
    tracks = tracks.copy()
    tracks["global_track_id"] = [
        mapping[(str(camera), int(track_id))]
        for camera, track_id in zip(tracks["camera_name"], tracks["local_track_id"])
    ]
    return tracks, pd.DataFrame(accepted)


def build_fused_observations(tracks: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Average simultaneous views so one player contributes one pitch observation."""
    fused = tracks[tracks["team_index"].isin([0, 1])].copy()
    fused["sync_bin"] = np.rint(fused["global_time_seconds"] / tolerance).astype(int)
    fused = (
        fused.groupby(["global_track_id", "team_index", "team_name", "sync_bin"])
        .agg(global_time_seconds=("global_time_seconds", "mean"),
             pitch_x=("pitch_x", "mean"), pitch_y=("pitch_y", "mean"),
             confidence=("confidence", "max"),
             cameras_seen=("camera_name", "nunique"),
             camera_names=("camera_name", lambda values: ",".join(sorted(set(values)))))
        .reset_index()
        .sort_values(["global_track_id", "global_time_seconds"])
    )
    grouped = fused.groupby("global_track_id")
    dt = grouped["global_time_seconds"].diff()
    step = np.hypot(grouped["pitch_x"].diff(), grouped["pitch_y"].diff())
    speed = step / dt.replace(0, np.nan)
    valid = speed.notna() & (speed <= MAX_PLAUSIBLE_SPEED_MS)
    fused["speed_mps"] = speed.where(valid, 0.0).fillna(0.0)
    fused["step_metres"] = step.where(valid, 0.0).fillna(0.0)
    fused["distance_metres"] = fused.groupby("global_track_id")["step_metres"].cumsum()
    return fused


def summarise_fused_tracks(fused: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fused.groupby(["global_track_id", "team_index", "team_name"])
        .agg(first_seen=("global_time_seconds", "min"),
             last_seen=("global_time_seconds", "max"),
             fused_observations=("sync_bin", "size"),
             cameras_seen=("cameras_seen", "max"),
             distance_metres=("step_metres", "sum"),
             top_speed_mps=("speed_mps", lambda values: float(values.quantile(0.95))),
             mean_pitch_x=("pitch_x", "mean"), mean_pitch_y=("pitch_y", "mean"),
             median_confidence=("confidence", "median"))
        .reset_index()
    )
    summary["tracked_seconds"] = summary["last_seen"] - summary["first_seen"]
    return summary.sort_values(["team_index", "distance_metres"],
                               ascending=[True, False])


def run_multicam(args: argparse.Namespace) -> None:
    names, offsets = validate_multicam_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Multi-camera analysis uses calibrated pitch coordinates in metres.")
    print("Camera offsets are added to local timestamps; Camera 1 is normally 0.")

    camera_tracks = []
    camera_quality = []
    for index, (name, offset) in enumerate(zip(names, offsets)):
        print(f"\n[{index + 1}/{len(names)}] Analysing {name}: {args.video[index]}")
        tracks = analyse_multicam_view(args, index, name, offset)
        camera_tracks.append(tracks)
        quality_path = (args.output_dir / f"camera_{index + 1}_{name}"
                        / "tracking_quality.csv")
        quality = pd.read_csv(quality_path)
        quality.insert(0, "camera_index", index)
        quality.insert(1, "camera_name", name)
        quality.insert(2, "video", str(args.video[index]))
        quality.insert(3, "offset_seconds", offset)
        camera_quality.append(quality)

    combined = pd.concat(camera_tracks, ignore_index=True)
    combined, associations = associate_multicam_tracks(combined, args)
    fused = build_fused_observations(combined, args.sync_tolerance)
    if fused.empty:
        raise SystemExit("No known-team observations remained for multi-camera fusion.")
    summary = summarise_fused_tracks(fused)

    combined_path = args.output_dir / "multicam_camera_tracks.csv"
    fused_path = args.output_dir / "multicam_fused_tracks.csv"
    summary_path = args.output_dir / "multicam_player_summary.csv"
    associations_path = args.output_dir / "multicam_associations.csv"
    quality_path = args.output_dir / "multicam_camera_quality.csv"
    combined.to_csv(combined_path, index=False)
    fused.to_csv(fused_path, index=False)
    summary.to_csv(summary_path, index=False)
    associations.to_csv(associations_path, index=False)
    pd.concat(camera_quality, ignore_index=True).to_csv(quality_path, index=False)

    multi_view = int((summary["cameras_seen"] > 1).sum())
    print("\nMulti-camera outputs")
    print(f"  Per-camera observations: {combined_path}")
    print(f"  Fused pitch observations: {fused_path}")
    print(f"  Global player summary:    {summary_path}")
    print(f"  Accepted associations:    {associations_path}")
    print(f"  Camera quality report:    {quality_path}")
    print(f"  Global tracks seen by multiple cameras: {multi_view}/{len(summary)}")
    print("Review multicam_associations.csv before treating global IDs as players.")


# --------------------------------------------------------------------------
# GPS import and video/GPS sensor fusion
# --------------------------------------------------------------------------

GPS_ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date_time", "utc_time", "recorded_at"),
    "latitude": ("latitude", "lat", "position_lat"),
    "longitude": ("longitude", "lon", "lng", "long", "position_long"),
    "speed_mps": ("speed_mps", "speed", "velocity", "enhanced_speed"),
    "heart_rate_bpm": ("heart_rate_bpm", "heart_rate", "heartrate", "hr", "bpm"),
    "accuracy_m": ("accuracy_m", "accuracy", "horizontal_accuracy", "hdop"),
    "x_m": ("x_m", "pitch_x", "local_x"),
    "y_m": ("y_m", "pitch_y", "local_y"),
}

# Column names the rest of this pipeline actually writes, in the order they
# should be preferred. The fusion command used to look for names nothing
# produced, so it exited on its own documented inputs.
VIDEO_TIME_COLUMNS = ("global_time_seconds", "time_seconds", "global_time_s", "time_s")
VIDEO_ID_COLUMNS = ("global_track_id", "global_player_id", "track_id")


def _gps_number(value):
    try:
        return float(value) if value not in (None, "") else np.nan
    except (TypeError, ValueError):
        return np.nan


def _canonical_gps_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Map vendor-specific headings onto a deliberately small open schema."""
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in frame.columns}
    result = pd.DataFrame(index=frame.index)
    for canonical, aliases in GPS_ALIASES.items():
        source = next((lookup[a] for a in aliases if a in lookup), None)
        result[canonical] = frame[source] if source is not None else np.nan
    if result["timestamp"].isna().all():
        raise ValueError("GPS data has no recognizable timestamp/time column.")
    return result


def _read_gpx(path: Path) -> pd.DataFrame:
    root = ET.parse(path).getroot()
    rows = []
    for point in root.findall(".//{*}trkpt") + root.findall(".//{*}rtept"):
        def value(name):
            node = point.find(".//{*}" + name)
            return node.text if node is not None else None
        rows.append({"timestamp": value("time"), "latitude": point.get("lat"),
                     "longitude": point.get("lon"), "speed_mps": value("speed"),
                     "heart_rate_bpm": value("hr"), "accuracy_m": value("hdop")})
    return pd.DataFrame(rows)


def _read_tcx(path: Path) -> pd.DataFrame:
    root = ET.parse(path).getroot()
    rows = []
    for point in root.findall(".//{*}Trackpoint"):
        def value(name):
            node = point.find(".//{*}" + name)
            return node.text if node is not None else None
        rows.append({"timestamp": value("Time"), "latitude": value("LatitudeDegrees"),
                     "longitude": value("LongitudeDegrees"),
                     "speed_mps": value("Speed"), "heart_rate_bpm": value("Value")})
    return pd.DataFrame(rows)


def _read_fit(path: Path) -> pd.DataFrame:
    try:
        from fitparse import FitFile
    except ImportError as exc:
        raise SystemExit("FIT input needs: pip install fitparse") from exc
    rows = []
    for message in FitFile(str(path)).get_messages("record"):
        data = {field.name: field.value for field in message}
        # FIT semicircles are the normal raw latitude/longitude representation.
        factor = 180.0 / (2 ** 31)
        lat, lon = data.get("position_lat"), data.get("position_long")
        rows.append({"timestamp": data.get("timestamp"),
                     "latitude": lat * factor if lat is not None else None,
                     "longitude": lon * factor if lon is not None else None,
                     "speed_mps": data.get("enhanced_speed", data.get("speed")),
                     "heart_rate_bpm": data.get("heart_rate")})
    return pd.DataFrame(rows)


def read_gps_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        raw = pd.read_csv(path, sep=None, engine="python")
    elif suffix == ".gpx":
        raw = _read_gpx(path)
    elif suffix == ".tcx":
        raw = _read_tcx(path)
    elif suffix == ".fit":
        raw = _read_fit(path)
    else:
        raise ValueError(f"Unsupported GPS format {suffix}; use CSV, GPX, TCX or FIT.")
    return _canonical_gps_frame(raw)


def latlon_to_pitch(lat, lon, origin_lat, origin_lon, bearing_deg):
    """Equirectangular local metres, rotated onto pitch x/y axes."""
    radius = 6371008.8
    north = np.radians(lat - origin_lat) * radius
    east = np.radians(lon - origin_lon) * radius * math.cos(math.radians(origin_lat))
    angle = math.radians(bearing_deg)
    x = north * math.cos(angle) + east * math.sin(angle)
    y = -north * math.sin(angle) + east * math.cos(angle)
    return x, y


def normalize_gps_file(path: Path, player: str, offset: float,
                       match_start, args) -> pd.DataFrame:
    data = read_gps_file(path)
    data["timestamp_utc"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    numeric_time = pd.to_numeric(data["timestamp"], errors="coerce")
    if data["timestamp_utc"].notna().any():
        origin = match_start if match_start is not None else data["timestamp_utc"].dropna().iloc[0]
        data["match_time_s"] = (data["timestamp_utc"] - origin).dt.total_seconds() + offset
    elif numeric_time.notna().any():
        data["match_time_s"] = numeric_time - numeric_time.dropna().iloc[0] + offset
    else:
        raise ValueError(f"Could not parse timestamps in {path.name}")
    for column in ("latitude", "longitude", "speed_mps", "heart_rate_bpm",
                   "accuracy_m", "x_m", "y_m"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if args.pitch_origin_lat is not None and args.pitch_origin_lon is not None:
        valid = data["latitude"].notna() & data["longitude"].notna()
        data.loc[valid, "x_m"], data.loc[valid, "y_m"] = latlon_to_pitch(
            data.loc[valid, "latitude"].to_numpy(), data.loc[valid, "longitude"].to_numpy(),
            args.pitch_origin_lat, args.pitch_origin_lon, args.pitch_bearing)
    data = data.sort_values("match_time_s").drop_duplicates("match_time_s")
    # Derive speed from coordinates when the device omitted it.
    coordinate_x = data["x_m"] if data["x_m"].notna().any() else data["longitude"]
    coordinate_y = data["y_m"] if data["y_m"].notna().any() else data["latitude"]
    if data["speed_mps"].isna().all() and coordinate_x.notna().any():
        if data["x_m"].notna().any():
            distance = np.hypot(coordinate_x.diff(), coordinate_y.diff())
        else:
            _, north = latlon_to_pitch(data["latitude"].to_numpy(), data["longitude"].to_numpy(),
                                       data["latitude"].dropna().iloc[0],
                                       data["longitude"].dropna().iloc[0], 0)
            east, _ = latlon_to_pitch(data["latitude"].to_numpy(), data["longitude"].to_numpy(),
                                      data["latitude"].dropna().iloc[0],
                                      data["longitude"].dropna().iloc[0], 90)
            distance = pd.Series(np.hypot(east, north), index=data.index).diff().abs()
        data["speed_mps"] = distance / data["match_time_s"].diff()
    data["gps_player"] = player
    data["source_file"] = path.name
    data["plausible"] = data["speed_mps"].isna() | data["speed_mps"].between(0, args.max_speed)
    keep = ["gps_player", "match_time_s", "timestamp_utc", "latitude", "longitude",
            "x_m", "y_m", "speed_mps", "heart_rate_bpm", "accuracy_m",
            "plausible", "source_file"]
    return data[keep]


def run_gps(args: argparse.Namespace) -> None:
    count = len(args.gps_file)
    players = args.player or [path.stem for path in args.gps_file]
    offsets = args.gps_offset or [0.0] * count
    if len(players) != count or len(offsets) != count:
        raise SystemExit("Repeat --player and --gps-offset exactly once per --gps-file.")
    if args.resample_hz < 0:
        raise SystemExit("--resample-hz cannot be negative.")
    match_start = pd.to_datetime(args.match_start, utc=True) if args.match_start else None
    tracks = pd.concat([normalize_gps_file(path, player, offset, match_start, args)
                        for path, player, offset in zip(args.gps_file, players, offsets)],
                       ignore_index=True)
    if args.resample_hz > 0:
        period = 1.0 / args.resample_hz
        output = []
        for player, group in tracks.groupby("gps_player", sort=False):
            grid = pd.DataFrame({"match_time_s": np.arange(group.match_time_s.min(),
                                                             group.match_time_s.max() + period / 2,
                                                             period)})
            merged = pd.merge_asof(grid, group.sort_values("match_time_s"),
                                   on="match_time_s", direction="nearest",
                                   tolerance=period)
            merged["gps_player"] = player
            output.append(merged)
        tracks = pd.concat(output, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "gps_normalized_tracks.csv"
    quality = (tracks.groupby("gps_player").agg(samples=("match_time_s", "size"),
               start_s=("match_time_s", "min"), end_s=("match_time_s", "max"),
               valid_position_pct=("latitude", lambda s: round(100*s.notna().mean(), 2)),
               median_speed_mps=("speed_mps", "median"),
               plausible_pct=("plausible", lambda s: round(100*s.fillna(False).mean(), 2)))
               .reset_index())
    tracks.to_csv(out, index=False)
    quality.to_csv(args.output_dir / "gps_quality.csv", index=False)
    print(f"GPS normalized: {out}")


def resolve_column(frame: pd.DataFrame, candidates, what: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise SystemExit(
        f"Could not find a {what} column. Looked for {', '.join(candidates)}; "
        f"the file has {', '.join(map(str, frame.columns[:12]))}"
    )


def run_sensor_fusion(args: argparse.Namespace) -> None:
    video, gps, mapping = (pd.read_csv(p) for p in
                           (args.video_tracks, args.gps_tracks, args.player_map))
    required_map = {"gps_player", "global_player_id"}
    if not required_map.issubset(mapping.columns):
        raise SystemExit("player-map CSV needs gps_player and global_player_id columns.")

    time_col = resolve_column(video, VIDEO_TIME_COLUMNS, "video timestamp")
    id_col = resolve_column(video, VIDEO_ID_COLUMNS, "video player id")
    video = video.rename(columns={time_col: "match_time_s",
                                  id_col: "global_player_id"})

    # Both sides carry speed_mps, so merge_asof would suffix them and every later
    # reference to speed_mps would raise KeyError. Rename the GPS side up front.
    gps = gps.merge(mapping[list(required_map)], on="gps_player", how="inner")
    gps = gps.rename(columns={"speed_mps": "gps_speed_mps",
                              "heart_rate_bpm": "gps_heart_rate_bpm"})
    if gps.empty:
        raise SystemExit("No gps_player value in the map matched the GPS file.")

    pieces = []
    for player_id, gps_player in gps.groupby("global_player_id", sort=False):
        camera_player = video[video["global_player_id"].astype(str) == str(player_id)].copy()
        if camera_player.empty:
            continue
        # Both frames carry the join key. Left in place, merge_asof would suffix
        # it to global_player_id_video/_gps and the summary groupby would fail.
        gps_player = gps_player.drop(columns=["global_player_id"])
        joined = pd.merge_asof(camera_player.sort_values("match_time_s"),
                               gps_player.sort_values("match_time_s"), on="match_time_s",
                               direction="nearest", tolerance=args.sync_tolerance,
                               suffixes=("_video", "_gps"))
        joined["position_source"] = np.where(joined["x_m"].notna(), "gps", "video")
        video_x = joined.get("pitch_x", pd.Series(np.nan, index=joined.index))
        video_y = joined.get("pitch_y", pd.Series(np.nan, index=joined.index))
        if args.prefer_gps_position:
            joined["fused_x_m"] = joined["x_m"].fillna(video_x)
            joined["fused_y_m"] = joined["y_m"].fillna(video_y)
        else:
            joined["fused_x_m"] = video_x.fillna(joined["x_m"])
            joined["fused_y_m"] = video_y.fillna(joined["y_m"])
        pieces.append(joined)
    if not pieces:
        raise SystemExit(
            "No mapped player IDs matched between the video and GPS files. The "
            f"video file identifies players by '{id_col}'; check that the "
            "global_player_id column in your player map uses the same values."
        )
    fused = pd.concat(pieces, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fused.to_csv(args.output_dir / "video_gps_fused_tracks.csv", index=False)
    summary = fused.groupby("global_player_id").agg(
        synchronized_samples=("match_time_s", "size"),
        gps_position_samples=("x_m", lambda s: int(s.notna().sum())),
        median_gps_speed_mps=("gps_speed_mps", "median"),
        max_gps_speed_mps=("gps_speed_mps", "max"),
        mean_heart_rate_bpm=("gps_heart_rate_bpm", "mean"),
    ).reset_index()
    summary.to_csv(args.output_dir / "video_gps_player_summary.csv", index=False)
    print(f"Sensor fusion complete: {args.output_dir / 'video_gps_fused_tracks.csv'}")
    print("Ball, possession, pass and event fusion are not included in this version.")


# --------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    print(f"Soccer tracker {VERSION}")
    if args.command == "record":
        run_record(args)
    elif args.command == "calibrate":
        run_calibrate(args)
    elif args.command == "analyse":
        run_analyse(args)
    elif args.command == "live":
        run_live(args)
    elif args.command == "multicam":
        run_multicam(args)
    elif args.command == "gps":
        run_gps(args)
    elif args.command == "sensor-fusion":
        run_sensor_fusion(args)


if __name__ == "__main__":
    main()
