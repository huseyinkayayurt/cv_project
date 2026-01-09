#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Highway vehicle counting (6 lanes) using classic image processing.

Key design (your case):
- Lane assignment is done ONLY at the counting line y = H*line_y_ratio, using a horizontal band histogram.
- Counting is done when a track ENTERS the counting band for the first time (fixes late-detection after crossing).
- Warmup window disables counting at the start so vehicles already on screen are not counted.

Dependencies:
  pip install opencv-python numpy

Run:
  python3 main.py --video input.mp4 --show 1
Optional:
  python3 main.py --video input.mp4 --save out.mp4
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input .mp4")
    ap.add_argument("--scale", type=float, default=1.0, help="Resize scale (e.g. 0.75)")
    ap.add_argument("--show", type=int, default=1, help="Show windows (1/0)")
    ap.add_argument("--debug", type=int, default=1, help="Debug windows (1/0)")
    ap.add_argument("--save", type=str, default="", help="Output video path (optional)")
    ap.add_argument("--time_limit_sec", type=int, default=11 * 60, help="Stop processing after this many seconds")

    # Lane estimation / counting line
    ap.add_argument("--sample_secs", type=float, default=4.0, help="Seconds to sample for lane boundary estimation")
    ap.add_argument("--max_samples", type=int, default=140, help="Max frames to sample for lane boundary estimation")
    ap.add_argument("--line_y_ratio", type=float, default=0.90, help="Counting line at y = H * ratio")
    ap.add_argument("--lane_band_h", type=int, default=32, help="Half height of band around counting line (px)")
    ap.add_argument("--lane_peaks_min_prom", type=float, default=0.06, help="Min peak prominence (0..1)")
    ap.add_argument("--lane_peaks_min_dist", type=int, default=30, help="Min peak distance in pixels")

    # Detection / tracking
    ap.add_argument("--min_area", type=int, default=450, help="Min contour area for vehicle candidate")
    ap.add_argument("--max_area", type=int, default=220000, help="Max contour area for vehicle candidate")
    ap.add_argument("--max_missed", type=int, default=12, help="Frames to keep a missing track alive")
    ap.add_argument("--match_dist", type=int, default=70, help="Max centroid distance to match detections to tracks")

    # Counting logic
    ap.add_argument("--side_margin_px", type=int, default=14, help="Band half-height used for counting decision")
    ap.add_argument("--warmup_sec", type=float, default=1.0, help="Disable counting for first N seconds")
    return ap.parse_args()


# -----------------------------
# Data
# -----------------------------

@dataclass
class LaneAtLineModel:
    # 3 boundaries per side (outer boundaries are the frame borders)
    # left_x  = [b1, b2, b3(inner near refuge)]
    # right_x = [b4(inner near refuge), b5, b6]
    left_x: List[int]
    right_x: List[int]
    x_mid: int
    line_y: int
    frame_w: int


@dataclass
class Track:
    track_id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    last_seen: int
    missed: int

    counted: bool
    lane_id: Optional[int]

    # new: count when first entering the counting band
    entered_band: bool


# -----------------------------
# Helpers
# -----------------------------

def resize_frame(frame, scale):
    if abs(scale - 1.0) < 1e-6:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def apply_roi_mask(img, roi_poly):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_poly], 255)
    if img.ndim == 2:
        return cv2.bitwise_and(img, img, mask=mask)
    return cv2.bitwise_and(img, img, mask=mask)


def default_road_roi(h, w):
    # wide trapezoid: only used for motion mask cleanup
    y_top = int(h * 0.28)
    y_bottom = h - 1
    roi = np.array([
        [int(w * 0.02), y_bottom],
        [int(w * 0.30), y_top],
        [int(w * 0.70), y_top],
        [int(w * 0.98), y_bottom],
    ], dtype=np.int32)
    return roi


def smooth_1d(arr, k=31):
    k = int(k)
    if k < 3:
        return arr
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(arr, kernel, mode="same")


def find_peaks_1d(sig, min_dist=40, min_prom=0.08):
    sig = sig.astype(np.float32)
    mx = float(np.max(sig) + 1e-6)
    s = sig / mx

    candidates = []
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] >= min_prom:
            candidates.append((s[i], i))

    candidates.sort(reverse=True, key=lambda t: t[0])

    chosen = []
    for val, idx in candidates:
        if all(abs(idx - j) >= min_dist for j in chosen):
            chosen.append(idx)

    chosen.sort()
    return chosen


def pick_left3(peaks, x_min, x_max):
    p = sorted([x for x in peaks if x_min <= x <= x_max])
    if len(p) < 3:
        return None
    # [outer1, outer2, inner near refuge]
    return [p[0], p[1], p[-1]]


def pick_right3(peaks, x_min, x_max):
    p = sorted([x for x in peaks if x_min <= x <= x_max])
    if len(p) < 3:
        return None
    # [inner near refuge, outer2, outer1 near right border]
    return [p[0], p[-2], p[-1]]


# -----------------------------
# Lane@Line response
# -----------------------------

def lane_response_band(frame_bgr, y_line, band_half_h):
    """
    Build a robust lane marking response on a horizontal band around y_line.
    Combines vertical edge energy with bright pixels (lane paint).
    """
    h, w = frame_bgr.shape[:2]
    y1 = max(0, y_line - band_half_h)
    y2 = min(h - 1, y_line + band_half_h)
    band = frame_bgr[y1:y2, :]

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 4))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    absx = np.abs(sobelx)
    absx = (absx / (absx.max() + 1e-6) * 255.0).astype(np.uint8)

    thr_bright = int(np.percentile(gray, 88))
    bright = (gray >= thr_bright).astype(np.uint8) * 255

    resp = cv2.bitwise_and(absx, bright)
    resp = cv2.morphologyEx(resp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)), iterations=1)
    resp = cv2.morphologyEx(resp, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return resp, (y1, y2)


def estimate_lane_boundaries_at_line(
        cap,
        scale,
        sample_secs,
        max_samples,
        line_y_ratio,
        band_half_h,
        min_prom,
        min_dist,
        debug=1
):
    """
    Sample early frames and estimate lane boundary X positions at the counting line.
    Uses 3-band stabilization per frame: y-d, y, y+d and takes median to reduce jitter.
    """

    def _estimate_once_for_line(frame, y_line):
        H, W = frame.shape[:2]
        resp, _ = lane_response_band(frame, y_line, band_half_h)

        hist = np.sum(resp.astype(np.float32) / 255.0, axis=0)
        hist = smooth_1d(hist, k=max(31, (W // 70) | 1))

        lo = int(W * 0.40)
        hi = int(W * 0.60)
        x_mid = lo + int(np.argmin(hist[lo:hi]))
        if abs(x_mid - (W // 2)) > int(W * 0.12):
            x_mid = W // 2

        peaks = find_peaks_1d(hist, min_dist=min_dist, min_prom=min_prom)

        # refuge gap (0.90 works well with ~0.018)
        gap = int(W * 0.018)
        left_min, left_max = int(W * 0.02), max(int(W * 0.22), x_mid - gap)
        right_min, right_max = min(int(W * 0.78), x_mid + gap), int(W * 0.98)

        left3 = pick_left3(peaks, left_min, left_max)
        right3 = pick_right3(peaks, right_min, right_max)

        return left3, right3, x_mid, resp, hist

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0
    sample_frames = int(min(max_samples, max(15, sample_secs * fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, fr0 = cap.read()
    if not ok:
        raise RuntimeError("Cannot read video.")
    fr0 = resize_frame(fr0, scale)
    H, W = fr0.shape[:2]
    line_y = int(H * line_y_ratio)

    left_candidates = []
    right_candidates = []
    mid_candidates = []

    dbg_hist_last = None
    dbg_resp_last = None
    dbg_frame_last = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(sample_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_frame(frame, scale)

        d = max(10, band_half_h // 2)
        y_candidates = [line_y - d, line_y, line_y + d]

        left_list = []
        right_list = []
        mid_list = []

        resp_keep = None
        hist_keep = None

        for yy in y_candidates:
            yy = int(np.clip(yy, 0, H - 1))
            l3, r3, xm, resp, hist = _estimate_once_for_line(frame, yy)
            if l3 is None or r3 is None:
                continue
            left_list.append(l3)
            right_list.append(r3)
            mid_list.append(xm)
            resp_keep = resp
            hist_keep = hist

        if len(left_list) >= 2 and len(right_list) >= 2 and len(mid_list) >= 2:
            left3 = np.median(np.array(left_list, dtype=np.float32), axis=0).round().astype(int).tolist()
            right3 = np.median(np.array(right_list, dtype=np.float32), axis=0).round().astype(int).tolist()
            x_mid = int(np.median(np.array(mid_list, dtype=np.float32)).round())

            left3 = sorted(left3)
            right3 = sorted(right3)

            left_candidates.append(left3)
            right_candidates.append(right3)
            mid_candidates.append(x_mid)

            if debug and resp_keep is not None and hist_keep is not None:
                dbg_resp_last = resp_keep
                dbg_hist_last = hist_keep
                dbg_frame_last = frame

    if len(left_candidates) < 8:
        raise RuntimeError(
            "Lane-at-line estimation failed: not enough reliable samples. "
            "Try adjusting --lane_peaks_min_prom / --lane_peaks_min_dist / --lane_band_h."
        )

    left_candidates = np.array(left_candidates, dtype=np.int32)  # Nx3
    right_candidates = np.array(right_candidates, dtype=np.int32)  # Nx3
    mid_candidates = np.array(mid_candidates, dtype=np.int32)

    left_med = np.median(left_candidates, axis=0).round().astype(int).tolist()
    right_med = np.median(right_candidates, axis=0).round().astype(int).tolist()
    mid_med = int(np.median(mid_candidates).round())

    left_med = sorted(left_med)
    right_med = sorted(right_med)

    if not (left_med[-1] < mid_med and right_med[0] > mid_med):
        left_med = [min(x, mid_med - 10) for x in left_med]
        right_med = [max(x, mid_med + 10) for x in right_med]
        left_med = sorted(left_med)
        right_med = sorted(right_med)

    model = LaneAtLineModel(
        left_x=left_med,
        right_x=right_med,
        x_mid=mid_med,
        line_y=line_y,
        frame_w=W
    )

    if debug and dbg_frame_last is not None:
        vis = dbg_frame_last.copy()

        cv2.rectangle(
            vis,
            (0, max(0, line_y - band_half_h)),
            (W - 1, min(H - 1, line_y + band_half_h)),
            (0, 255, 255),
            2
        )
        cv2.line(vis, (0, line_y), (W - 1, line_y), (0, 0, 255), 2)
        cv2.line(vis, (mid_med, 0), (mid_med, H - 1), (0, 255, 255), 2)

        for x in left_med:
            cv2.line(vis, (x, 0), (x, H - 1), (255, 0, 0), 2)
        for x in right_med:
            cv2.line(vis, (x, 0), (x, H - 1), (255, 0, 0), 2)

        cv2.imshow("Lane@Line Calibration", vis)
        if dbg_resp_last is not None:
            cv2.imshow("Lane@Line Response Band", dbg_resp_last)

        if dbg_hist_last is not None:
            hist = dbg_hist_last.astype(np.float32)
            hist = hist / (hist.max() + 1e-6)
            plot_h = 140
            plot = np.zeros((plot_h, W, 3), dtype=np.uint8)
            for x in range(W):
                yv = int((1.0 - hist[x]) * (plot_h - 1))
                plot[yv:, x] = (255, 255, 255)

            cv2.line(plot, (mid_med, 0), (mid_med, plot_h - 1), (0, 255, 255), 2)
            for x in left_med:
                cv2.line(plot, (x, 0), (x, plot_h - 1), (255, 0, 0), 1)
            for x in right_med:
                cv2.line(plot, (x, 0), (x, plot_h - 1), (255, 0, 0), 1)

            cv2.imshow("Lane@Line Histogram", plot)

        cv2.waitKey(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return model


def lane_id_at_crossing(cx: int, lane_model: LaneAtLineModel, width: int) -> Optional[int]:
    """
    left_x = [b1,b2,b3]
      Lane1: [0, b1)
      Lane2: [b1, b2)
      Lane3: [b2, b3)
    right_x = [b4,b5,b6]
      Lane4: [b4, b5)
      Lane5: [b5, b6)
      Lane6: [b6, width)
    """
    if cx < lane_model.x_mid:
        b1, b2, b3 = lane_model.left_x
        if cx < b1:
            return 1
        if cx < b2:
            return 2
        if cx < b3:
            return 3
        return None
    else:
        b4, b5, b6 = lane_model.right_x
        if cx < b4:
            return None
        if cx < b5:
            return 4
        if cx < b6:
            return 5
        return 6


# -----------------------------
# Vehicle detection
# -----------------------------

def build_bg_subtractor():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=False)


def detect_vehicles(frame_bgr, roi_poly, subtractor,
                    min_area, max_area,
                    line_y: int, margin: int):
    """
    Returns list of bboxes (x,y,w,h) and fg mask.
    Uses a dynamic min_area: smaller near the counting line.
    """
    roi_frame = apply_roi_mask(frame_bgr, roi_poly)
    fg = subtractor.apply(roi_frame)

    fg = cv2.medianBlur(fg, 5)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
    fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    h, w = frame_bgr.shape[:2]
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 12 or bh < 12:
            continue
        if y < int(h * 0.18):
            continue

        cy = y + bh // 2
        near_line = abs(cy - line_y) <= (margin * 2)
        min_area_eff = max(120, min_area // 2) if near_line else min_area

        if area < min_area_eff:
            continue

        bboxes.append((x, y, bw, bh))

    return bboxes, fg


# -----------------------------
# Tracking
# -----------------------------

class CentroidTracker:
    def __init__(self, max_missed=12, match_dist=70):
        self.max_missed = max_missed
        self.match_dist = match_dist
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def centroid_of(bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def update(self, detections: List[Tuple[int, int, int, int]], frame_idx: int):
        det_centroids = [self.centroid_of(b) for b in detections]

        # mark tracks missed
        for tid in list(self.tracks.keys()):
            t = self.tracks[tid]
            t.missed += 1
            self.tracks[tid] = t

        used_dets = set()
        for tid, t in list(self.tracks.items()):
            best_j = None
            best_d = 1e9
            for j, c in enumerate(det_centroids):
                if j in used_dets:
                    continue
                d = np.hypot(c[0] - t.centroid[0], c[1] - t.centroid[1])
                if d < best_d:
                    best_d = d
                    best_j = j

            if best_j is not None and best_d <= self.match_dist:
                bbox = detections[best_j]
                self.tracks[tid] = Track(
                    track_id=tid,
                    centroid=det_centroids[best_j],
                    bbox=bbox,
                    last_seen=frame_idx,
                    missed=0,
                    counted=t.counted,
                    lane_id=t.lane_id,
                    entered_band=t.entered_band
                )
                used_dets.add(best_j)

        # new tracks
        for j, bbox in enumerate(detections):
            if j in used_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(
                track_id=tid,
                centroid=det_centroids[j],
                bbox=bbox,
                last_seen=frame_idx,
                missed=0,
                counted=False,
                lane_id=None,
                entered_band=False
            )

        # prune
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]

        return list(self.tracks.values())


# -----------------------------
# Counting logic (FIX)
# -----------------------------

def update_counts_for_tracks(tracks: List[Track],
                             lane_model: LaneAtLineModel,
                             margin: int,
                             counts: Dict[int, int],
                             counting_enabled: bool):
    """
    Count when a track ENTERS the counting band for the first time.
    Fixes missed counts when detection happens only after the vehicle already crossed.
    """
    y_line = lane_model.line_y

    for i in range(len(tracks)):
        t = tracks[i]
        cx, cy = t.centroid

        in_band = (y_line - margin) <= cy <= (y_line + margin)

        if in_band and (not t.entered_band):
            t.entered_band = True

            if counting_enabled and (not t.counted):
                lane_id = lane_id_at_crossing(cx, lane_model, lane_model.frame_w)
                t.lane_id = lane_id
                if lane_id is not None:
                    counts[lane_id] += 1
                    t.counted = True

        tracks[i] = t


# -----------------------------
# Overlay
# -----------------------------
def overlay_lane_at_line(frame, lane_model: LaneAtLineModel):
    out = frame.copy()
    h, w = out.shape[:2]

    # ONLY counting line
    cv2.line(
        out,
        (0, lane_model.line_y),
        (w - 1, lane_model.line_y),
        (0, 0, 255),  # red
        2
    )

    return out


def overlay_lane_at_line_debug(frame, lane_model: LaneAtLineModel, roi_poly, band_half_h: int):
    out = frame.copy()
    h, w = out.shape[:2]

    # ROI (for debugging)
    cv2.polylines(out, [roi_poly], True, (0, 255, 255), 2)

    # band rectangle
    cv2.rectangle(out,
                  (0, max(0, lane_model.line_y - band_half_h)),
                  (w - 1, min(h - 1, lane_model.line_y + band_half_h)),
                  (0, 255, 255), 2)

    # counting line
    cv2.line(out, (0, lane_model.line_y), (w - 1, lane_model.line_y), (0, 0, 255), 2)

    # mid split
    cv2.line(out, (lane_model.x_mid, 0), (lane_model.x_mid, h - 1), (0, 255, 255), 2)

    # boundaries
    for x in lane_model.left_x:
        cv2.line(out, (x, 0), (x, h - 1), (255, 0, 0), 2)
    for x in lane_model.right_x:
        cv2.line(out, (x, 0), (x, h - 1), (255, 0, 0), 2)

    # lane labels at counting line
    lx = lane_model.left_x  # [b1,b2,b3]
    rx = lane_model.right_x  # [b4,b5,b6]

    def mid(a, b):
        return int((a + b) * 0.5)

    ytxt = max(25, lane_model.line_y - 18)

    cv2.putText(out, "Lane 1", (mid(0, lx[0]) - 25, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(out, "Lane 2", (mid(lx[0], lx[1]) - 25, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(out, "Lane 3", (mid(lx[1], lx[2]) - 25, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(out, "Lane 4", (mid(rx[0], rx[1]) - 25, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(out, "Lane 5", (mid(rx[1], rx[2]) - 25, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(out, "Lane 6", (mid(rx[2], lane_model.frame_w - 1) - 25, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return out


def overlay_tracks_and_counts(frame, tracks: List[Track], counts: Dict[int, int]):
    out = frame.copy()

    # counters
    x0, y0 = 20, 35
    for lane in range(1, 7):
        cv2.putText(out, f"Lane {lane}: {counts[lane]}",
                    (x0, y0 + (lane - 1) * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

    for t in tracks:
        x, y, w, h = t.bbox
        cx, cy = t.centroid
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        lane_txt = f"L{t.lane_id}" if t.lane_id is not None else "L?"
        cnt_txt = "C" if t.counted else ""
        cv2.putText(out, f"ID{t.track_id} {lane_txt} {cnt_txt}",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)

    return out


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # Lane boundaries at counting line
    lane_model = estimate_lane_boundaries_at_line(
        cap=cap,
        scale=args.scale,
        sample_secs=args.sample_secs,
        max_samples=args.max_samples,
        line_y_ratio=args.line_y_ratio,
        band_half_h=args.lane_band_h,
        min_prom=args.lane_peaks_min_prom,
        min_dist=args.lane_peaks_min_dist,
        debug=args.debug
    )

    # read one frame to init sizes and ROI
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, tmp = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame.")
    tmp = resize_frame(tmp, args.scale)
    H, W = tmp.shape[:2]
    roi_poly = default_road_roi(H, W)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0
    warmup_frames = int(args.warmup_sec * fps)

    # Video writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (W, H))

    subtractor = build_bg_subtractor()
    tracker = CentroidTracker(max_missed=args.max_missed, match_dist=args.match_dist)
    counts = {i: 0 for i in range(1, 7)}

    t_start = time.time()
    frame_idx = 0

    while True:
        # if time.time() - t_start > args.time_limit_sec:
        #     break

        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_frame(frame, args.scale)
        frame_idx += 1

        det_bboxes, fg = detect_vehicles(
            frame, roi_poly, subtractor,
            args.min_area, args.max_area,
            lane_model.line_y, args.side_margin_px
        )
        tracks = tracker.update(det_bboxes, frame_idx)

        counting_enabled = (frame_idx > warmup_frames)
        update_counts_for_tracks(tracks, lane_model, args.side_margin_px, counts, counting_enabled)

        vis = overlay_lane_at_line_debug(frame, lane_model, roi_poly, args.lane_band_h)
        # vis = overlay_lane_at_line(frame, lane_model)
        vis = overlay_tracks_and_counts(vis, tracks, counts)

        if args.show:
            cv2.imshow("Vehicle Counting (Lane@Line)", vis)
            # if args.debug:
            #     cv2.imshow("FG Mask", fg)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if writer is not None:
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("Final lane counts:")
    for i in range(1, 7):
        print(f"  Lane {i}: {counts[i]}")


if __name__ == "__main__":
    main()
