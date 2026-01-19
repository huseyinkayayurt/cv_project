import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input .mp4")
    ap.add_argument("--scale", type=float, default=1.0, help="Resize scale (e.g. 0.75)")
    ap.add_argument("--show", type=int, default=1, help="Show window (1/0)")
    ap.add_argument("--save", type=str, default="", help="Output video path (optional)")

    # Lane estimation / counting line
    ap.add_argument("--sample_secs", type=float, default=4.0, help="Seconds to sample for lane boundary estimation")
    ap.add_argument("--max_samples", type=int, default=140, help="Max frames to sample for lane boundary estimation")
    ap.add_argument("--line_y_ratio", type=float, default=0.90, help="Counting line at y = H * ratio")
    ap.add_argument("--lane_band_h", type=int, default=32, help="Half height of band around counting line (px)")
    ap.add_argument("--lane_peaks_min_prom", type=float, default=0.06, help="Min peak prominence (0..1)")
    ap.add_argument("--lane_peaks_min_dist", type=int, default=30, help="Min peak distance in pixels")

    # Detection / tracking
    ap.add_argument("--min_area", type=int, default=600, help="Min contour area for vehicle candidate")
    ap.add_argument("--max_area", type=int, default=220000, help="Max contour area for vehicle candidate")
    ap.add_argument("--max_missed", type=int, default=12, help="Frames to keep a missing track alive")
    ap.add_argument("--match_dist", type=int, default=90, help="Max centroid distance to match detections to tracks")

    # Counting logic
    ap.add_argument("--side_margin_px", type=int, default=14, help="Counting band half-height (px)")
    ap.add_argument("--warmup_sec", type=float, default=1.0, help="Disable counting for first N seconds")

    ap.add_argument("--post_count_px_above", type=int, default=140,
                    help="Early-count band ABOVE the counting line (px), spawn-frame only")
    ap.add_argument("--post_min_age", type=int, default=5,
                    help="Min track age (frames) before allowing post_count_above counting")
    ap.add_argument("--post_min_up_px", type=float, default=2.0,
                    help="Lane 1-3: require upward motion (prev_y - cur_y >= this) for post_count_above")
    ap.add_argument("--lane_cooldown_frames", type=int, default=18,
                    help="After counting a lane, ignore post_count_above counts in same lane for this many frames")

    return ap.parse_args()


# -----------------------------
# Data
# -----------------------------
@dataclass
class LaneAtLineModel:
    left_x: List[int]  # [b1,b2,b3]
    right_x: List[int]  # [b4,b5,b6]
    x_mid: int
    line_y: int
    frame_w: int


@dataclass
class Track:
    track_id: int
    centroid: Tuple[int, int]
    prev_centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    last_seen: int
    missed: int
    age: int

    counted: bool
    lane_id: Optional[int]
    entered_band: bool
    spawn_frame: int


# -----------------------------
# Small utils
# -----------------------------
def resize_frame(frame, scale: float):
    if abs(scale - 1.0) < 1e-6:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def sec_to_mmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def draw_info_panel(frame, frame_idx: int, total_frames: int, fps: float,
                    x: int, y: int, alpha: float = 0.55):
    out = frame.copy()
    h, w = out.shape[:2]
    if fps <= 1:
        fps = 30.0

    total_frames = int(total_frames) if total_frames and total_frames > 0 else 0
    cur_sec = frame_idx / fps
    total_sec = (total_frames / fps) if total_frames > 0 else 0.0

    lines = [
        f"Frame: {frame_idx} / {total_frames if total_frames > 0 else '?'}",
        f"Time : {sec_to_mmss(cur_sec)} / {sec_to_mmss(total_sec) if total_frames > 0 else '??:??'}",
        f"FPS  : {fps:.1f}",
    ]

    pad = 10
    line_h = 24
    panel_w = 320
    panel_h = pad * 2 + line_h * len(lines)

    x2 = min(w - 1, x + panel_w)
    y2 = min(h - 1, y + panel_h)

    overlay = out.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    ty = y + pad + line_h - 6
    for s in lines:
        cv2.putText(out, s, (x + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ty += line_h

    return out


def apply_roi_mask(img, roi_poly):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_poly], 255)
    return cv2.bitwise_and(img, img, mask=mask)


def default_road_roi(h, w, top_pad=0.07, bottom_pad=0.001, y_top_ratio=0.28):
    y_top = int(h * y_top_ratio)
    y_bottom = h - 1
    left_bottom = int(w * bottom_pad)
    right_bottom = int(w * (1.0 - bottom_pad))
    left_top = int(w * top_pad)
    right_top = int(w * (1.0 - top_pad))
    return np.array(
        [[left_bottom, y_bottom], [left_top, y_top], [right_top, y_top], [right_bottom, y_bottom]],
        dtype=np.int32
    )


def bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union + 1e-9)


# -----------------------------
# Lane-at-line estimation
# -----------------------------
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
    for _, idx in candidates:
        if all(abs(idx - j) >= min_dist for j in chosen):
            chosen.append(idx)

    chosen.sort()
    return chosen


def pick_left3(peaks, x_min, x_max):
    p = sorted([x for x in peaks if x_min <= x <= x_max])
    if len(p) < 3:
        return None
    return [p[0], p[1], p[-1]]  # [outer1, outer2, inner near refuge]


def pick_right3(peaks, x_min, x_max):
    p = sorted([x for x in peaks if x_min <= x <= x_max])
    if len(p) < 3:
        return None
    return [p[0], p[-2], p[-1]]  # [inner near refuge, outer2, outer1 near right border]


def lane_response_band(frame_bgr, y_line, band_half_h):
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

    return resp


def estimate_lane_boundaries_at_line(
        cap,
        scale,
        sample_secs,
        max_samples,
        line_y_ratio,
        band_half_h,
        min_prom,
        min_dist
) -> LaneAtLineModel:
    def _estimate_once(frame, y_line):
        H, W = frame.shape[:2]
        resp = lane_response_band(frame, y_line, band_half_h)

        hist = np.sum(resp.astype(np.float32) / 255.0, axis=0)
        hist = smooth_1d(hist, k=max(31, (W // 70) | 1))

        lo = int(W * 0.40)
        hi = int(W * 0.60)
        x_mid = lo + int(np.argmin(hist[lo:hi]))
        if abs(x_mid - (W // 2)) > int(W * 0.12):
            x_mid = W // 2

        peaks = find_peaks_1d(hist, min_dist=min_dist, min_prom=min_prom)

        gap = int(W * 0.018)
        left_min, left_max = int(W * 0.02), max(int(W * 0.22), x_mid - gap)
        right_min, right_max = min(int(W * 0.78), x_mid + gap), int(W * 0.98)

        left3 = pick_left3(peaks, left_min, left_max)
        right3 = pick_right3(peaks, right_min, right_max)
        return left3, right3, x_mid

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

        for yy in y_candidates:
            yy = int(np.clip(yy, 0, H - 1))
            l3, r3, xm = _estimate_once(frame, yy)
            if l3 is None or r3 is None:
                continue
            left_list.append(l3)
            right_list.append(r3)
            mid_list.append(xm)

        if len(left_list) >= 2 and len(right_list) >= 2 and len(mid_list) >= 2:
            left3 = np.median(np.array(left_list, dtype=np.float32), axis=0).round().astype(int).tolist()
            right3 = np.median(np.array(right_list, dtype=np.float32), axis=0).round().astype(int).tolist()
            x_mid = int(np.median(np.array(mid_list, dtype=np.float32)).round())

            left_candidates.append(sorted(left3))
            right_candidates.append(sorted(right3))
            mid_candidates.append(x_mid)

    if len(left_candidates) < 8:
        raise RuntimeError(
            "Lane-at-line estimation failed: not enough reliable samples. "
            "Try adjusting --lane_peaks_min_prom / --lane_peaks_min_dist / --lane_band_h."
        )

    left_med = np.median(np.array(left_candidates, dtype=np.int32), axis=0).round().astype(int).tolist()
    right_med = np.median(np.array(right_candidates, dtype=np.int32), axis=0).round().astype(int).tolist()
    mid_med = int(np.median(np.array(mid_candidates, dtype=np.int32)).round())

    left_med = sorted(left_med)
    right_med = sorted(right_med)

    if not (left_med[-1] < mid_med and right_med[0] > mid_med):
        left_med = [min(x, mid_med - 10) for x in left_med]
        right_med = [max(x, mid_med + 10) for x in right_med]
        left_med = sorted(left_med)
        right_med = sorted(right_med)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return LaneAtLineModel(left_x=left_med, right_x=right_med, x_mid=mid_med, line_y=line_y, frame_w=W)


def lane_id_at_crossing(cx: int, lane_model: LaneAtLineModel) -> Optional[int]:
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
# Detection
# -----------------------------
def build_bg_subtractor():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=False)


def detect_vehicles(frame_bgr, roi_poly, subtractor,
                    min_area, max_area,
                    line_y: int, margin: int):
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

    return bboxes


# -----------------------------
# Tracking (ID-stable)
# -----------------------------
class CentroidTracker:
    def __init__(self, max_missed=12, match_dist=90, revive_max_age=45, iou_match=0.15):
        self.max_missed = max_missed
        self.match_dist = match_dist
        self.revive_max_age = revive_max_age
        self.iou_match = iou_match

        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.lost: Dict[int, Track] = {}

    @staticmethod
    def centroid_of(bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def _score(self, t: Track, det_bbox, det_centroid):
        iou = bbox_iou(t.bbox, det_bbox)
        d = np.hypot(det_centroid[0] - t.centroid[0], det_centroid[1] - t.centroid[1])
        score = (2.5 * iou) - (d / (self.match_dist + 1e-6)) * 0.7
        return score, iou, d

    def _best_match(self, candidates: List[Track], det_bbox, det_centroid):
        best = None
        best_score = -1e9
        best_iou = 0.0
        best_d = 1e9
        for t in candidates:
            score, iou, d = self._score(t, det_bbox, det_centroid)
            if score > best_score:
                best_score, best, best_iou, best_d = score, t, iou, d
        return best, best_iou, best_d

    def update(self, detections: List[Tuple[int, int, int, int]], frame_idx: int):
        det_centroids = [self.centroid_of(b) for b in detections]

        # mark missed
        for tid in list(self.tracks.keys()):
            t = self.tracks[tid]
            t.missed += 1
            self.tracks[tid] = t

        used_dets = set()
        used_tracks = set()

        active_list = list(self.tracks.values())

        # 1) match to active
        for j, (bbox, c) in enumerate(zip(detections, det_centroids)):
            if not active_list:
                break

            best_t, best_iou, best_d = self._best_match(active_list, bbox, c)
            if best_t is None:
                continue

            if (best_iou >= self.iou_match) or (best_d <= self.match_dist):
                tid = best_t.track_id
                if tid in used_tracks:
                    continue

                old = self.tracks[tid]
                self.tracks[tid] = Track(
                    track_id=tid,
                    centroid=c,
                    prev_centroid=old.centroid,
                    bbox=bbox,
                    last_seen=frame_idx,
                    missed=0,
                    age=old.age + 1,
                    counted=old.counted,
                    lane_id=old.lane_id,
                    entered_band=old.entered_band,
                    spawn_frame=old.spawn_frame,
                )
                used_tracks.add(tid)
                used_dets.add(j)

        # 2) resurrect from lost
        remaining = [j for j in range(len(detections)) if j not in used_dets]
        if remaining and self.lost:
            lost_list = [lt for lt in self.lost.values() if frame_idx - lt.last_seen <= self.revive_max_age]

            for j in remaining:
                if not lost_list:
                    break
                bbox = detections[j]
                c = det_centroids[j]

                best_lt, best_iou, best_d = self._best_match(lost_list, bbox, c)
                if best_lt is None:
                    continue

                if (best_iou >= self.iou_match) or (best_d <= self.match_dist * 0.9):
                    tid = best_lt.track_id
                    old = best_lt
                    self.tracks[tid] = Track(
                        track_id=tid,
                        centroid=c,
                        prev_centroid=old.centroid,
                        bbox=bbox,
                        last_seen=frame_idx,
                        missed=0,
                        age=old.age + 1,
                        counted=old.counted,
                        lane_id=old.lane_id,
                        entered_band=old.entered_band,
                        spawn_frame=old.spawn_frame,
                    )
                    if tid in self.lost:
                        del self.lost[tid]
                    used_dets.add(j)

        # 3) create new tracks for unmatched detections
        for j, bbox in enumerate(detections):
            if j in used_dets:
                continue
            c = det_centroids[j]

            # guard: don't create new-id if it overlaps an existing active track
            overlapped = any(bbox_iou(t.bbox, bbox) >= 0.10 for t in self.tracks.values())
            if overlapped:
                continue

            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(
                track_id=tid,
                centroid=c,
                prev_centroid=c,
                bbox=bbox,
                last_seen=frame_idx,
                missed=0,
                age=1,
                counted=False,
                lane_id=None,
                entered_band=False,
                spawn_frame=frame_idx,
            )

        # 4) move expired tracks to lost
        for tid in list(self.tracks.keys()):
            t = self.tracks[tid]
            if t.missed > self.max_missed:
                self.lost[tid] = t
                del self.tracks[tid]

        # 5) purge old lost
        for tid in list(self.lost.keys()):
            if frame_idx - self.lost[tid].last_seen > self.revive_max_age:
                del self.lost[tid]

        return list(self.tracks.values())


# -----------------------------
# Counting
# -----------------------------
def update_counts_for_tracks(
        tracks: List[Track],
        lane_model: LaneAtLineModel,
        margin: int,
        counts: Dict[int, int],
        counting_enabled: bool,
        post_count_px_above: int,
        frame_idx: int,
        counted_track_ids: set,
        counted_track_lane: Dict[int, int],
        lane_last_count_frame: Dict[int, int],
        post_min_age: int,
        post_min_up_px: float,
        lane_cooldown_frames: int,
):
    y_line = lane_model.line_y
    y_band_top = y_line - margin
    y_band_bot = y_line + margin
    y_post_top_above = y_line - post_count_px_above

    for i in range(len(tracks)):
        t = tracks[i]
        cx, cy = t.centroid
        _, py = t.prev_centroid

        # never count same track_id twice
        if t.track_id in counted_track_ids:
            if t.lane_id is None and t.track_id in counted_track_lane:
                t.lane_id = counted_track_lane[t.track_id]
            t.counted = True
            tracks[i] = t
            continue

        lane_guess = lane_id_at_crossing(cx, lane_model)
        if lane_guess is not None:
            t.lane_id = lane_guess

        # A) normal: first enter main band
        in_band = (y_band_top <= cy <= y_band_bot)
        if in_band and (not t.entered_band):
            t.entered_band = True
            if counting_enabled and (not t.counted) and lane_guess is not None:
                counts[lane_guess] += 1
                t.counted = True
                counted_track_ids.add(t.track_id)
                counted_track_lane[t.track_id] = lane_guess
                lane_last_count_frame[lane_guess] = frame_idx
                tracks[i] = t
                continue

        # B) early above (spawn frame only) with guards
        if counting_enabled and (not t.counted) and (t.spawn_frame == frame_idx):
            if (cy < y_band_top) and (cy >= y_post_top_above) and (lane_guess is not None):

                # lane cooldown
                if frame_idx - lane_last_count_frame[lane_guess] < lane_cooldown_frames:
                    tracks[i] = t
                    continue

                # min age (filters flicker)
                if t.age < post_min_age:
                    tracks[i] = t
                    continue

                # motion filter for lanes 1-2-3
                if lane_guess in (1, 2, 3):
                    up_motion = (py - cy)  # positive => moving up
                    if up_motion < post_min_up_px:
                        tracks[i] = t
                        continue

                counts[lane_guess] += 1
                t.counted = True
                t.entered_band = True
                counted_track_ids.add(t.track_id)
                counted_track_lane[t.track_id] = lane_guess
                lane_last_count_frame[lane_guess] = frame_idx

        tracks[i] = t


# -----------------------------
# Overlay (minimal)
# -----------------------------
def overlay_counting_line(frame, lane_model: LaneAtLineModel):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.line(out, (0, lane_model.line_y), (w - 1, lane_model.line_y), (0, 0, 255), 2)
    return out


def overlay_tracks_and_counts(frame, tracks: List[Track], counts: Dict[int, int]):
    out = frame.copy()

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
        cv2.putText(out, lane_txt, (x, max(20, y - 8)),
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

    lane_model = estimate_lane_boundaries_at_line(
        cap=cap,
        scale=args.scale,
        sample_secs=args.sample_secs,
        max_samples=args.max_samples,
        line_y_ratio=args.line_y_ratio,
        band_half_h=args.lane_band_h,
        min_prom=args.lane_peaks_min_prom,
        min_dist=args.lane_peaks_min_dist,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, tmp = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame.")
    tmp = resize_frame(tmp, args.scale)
    H, W = tmp.shape[:2]
    roi_poly = default_road_roi(H, W, top_pad=0.07, bottom_pad=0.001, y_top_ratio=0.28)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    warmup_frames = int(args.warmup_sec * fps)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (W, H))

    subtractor = build_bg_subtractor()
    tracker = CentroidTracker(max_missed=args.max_missed, match_dist=args.match_dist)
    counts = {i: 0 for i in range(1, 7)}

    lane_last_count_frame = {i: -10 ** 9 for i in range(1, 7)}
    counted_track_ids = set()
    counted_track_lane: Dict[int, int] = {}

    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_frame(frame, args.scale)
        frame_idx += 1

        det_bboxes = detect_vehicles(
            frame, roi_poly, subtractor,
            args.min_area, args.max_area,
            lane_model.line_y, args.side_margin_px
        )
        tracks = tracker.update(det_bboxes, frame_idx)

        counting_enabled = (frame_idx > warmup_frames)
        update_counts_for_tracks(
            tracks=tracks,
            lane_model=lane_model,
            margin=args.side_margin_px,
            counts=counts,
            counting_enabled=counting_enabled,
            post_count_px_above=args.post_count_px_above,
            frame_idx=frame_idx,
            counted_track_ids=counted_track_ids,
            counted_track_lane=counted_track_lane,
            lane_last_count_frame=lane_last_count_frame,
            post_min_age=args.post_min_age,
            post_min_up_px=args.post_min_up_px,
            lane_cooldown_frames=args.lane_cooldown_frames,
        )

        vis = overlay_counting_line(frame, lane_model)
        vis = overlay_tracks_and_counts(vis, tracks, counts)
        vis = draw_info_panel(vis, frame_idx, total_frames, fps, x=W - 350, y=20, alpha=0.55)

        if args.show:
            cv2.imshow("Vehicle Counting", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if writer is not None:
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    total = 0
    print("Final lane counts:")
    for i in range(1, 7):
        print(f"  Lane {i}: {counts[i]}")
        total += counts[i]
    print(f"Total: {total}")


if __name__ == "__main__":
    main()
