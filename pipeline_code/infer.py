#!/usr/bin/env python3
"""
solar_verifier.py

Rooftop PV verification pipeline:
- Reads CSV/XLSX with sample_id, latitude, longitude (WGS84)
- Fetches satellite image (Google Static Maps preferred, ESRI fallback)
- Runs segmentation inference (Ultralytics YOLO)
- Produces JSON records per site with required fields:
  { "sample_id":..., "lat":..., "lon":..., "has_solar":..., "confidence":..., "pv_area_sqm_est":...,
    "buffer_radius_sqft":..., "qc_status":..., "bbox_or_mask": <GeoJSON polygon or bbox>, "image_metadata": {...} }
- Saves audit overlay image (PNG/JPEG) per site and final_predictions.json
- Implements buffer logic: prefer 1200 sqft; if absent, use 2400 sqft
"""

import os
import sys
import math
import time
import json
import argparse
from io import BytesIO
from datetime import datetime

import requests
from PIL import Image, ImageStat
import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
from shapely.geometry import Polygon, Point, box, mapping
from shapely.ops import unary_union

# -----------------------
# CONFIG
# -----------------------
IMG_SIZE_PX = 1024
CONF_THRESHOLD = 0.25      # model raw conf threshold to consider detection
FINAL_CONF_MIN = 0.05      # floor for final calibrated conf
WP_PER_M2 = 190            # Wp per m² assumption
AVG_PANEL_M2 = 1.9         # typical module area estimate
BRIGHTNESS_CLOUDY_THRESHOLD = 220
DARK_THRESHOLD = 25
JITTER_METERS = 10         # search radius for jittering (10 m)
REQUEST_TIMEOUT = 30
DEFAULT_ZOOM = 20
IMG_SCALE = 1
MIN_POLY_AREA_PX = 30      # min polygon area in pixel-space to accept
CALIB_AREA_SATURATE_M2 = 5.0  # area where calibration saturates
QC_CONF_HIGH_THRESHOLD = 0.75  # if calibrated conf >= this, QC more forgiving

# -----------------------
# UTILITIES
# -----------------------
def meters_per_pixel(lat, zoom=DEFAULT_ZOOM, tile_size=256):
    """Ground resolution (meters per pixel) at given latitude and zoom."""
    R = 6378137.0
    return (math.cos(math.radians(lat)) * 2 * math.pi * R) / (tile_size * (2 ** zoom))

def meters_to_latlon_offset(lat, meters):
    """Approximate conversion: meters -> degrees lat, lon at given lat."""
    dlat = meters / 110574.0
    dlon = meters / (111320.0 * math.cos(math.radians(lat)))
    return dlat, dlon

def generate_jitter_points(lat, lon):
    dlat, dlon = meters_to_latlon_offset(lat, JITTER_METERS)
    return [
        (lat, lon),
        (lat + dlat, lon),
        (lat - dlat, lon),
        (lat, lon + dlon),
        (lat, lon - dlon)
    ]

def basic_qc(pil_img: Image.Image):
    w, h = pil_img.size
    mean_brightness = ImageStat.Stat(pil_img.convert("L")).mean[0]
    flags = []
    if w < 512 or h < 512:
        flags.append("LOW_RESOLUTION")
    if mean_brightness > BRIGHTNESS_CLOUDY_THRESHOLD:
        flags.append("POSSIBLE_CLOUD_GLARE")
    if mean_brightness < DARK_THRESHOLD:
        flags.append("POSSIBLE_SHADOW_DARK")
    return {"width": w, "height": h, "mean_brightness": mean_brightness, "qc_flags": flags}

def qc_badness_score(qc):
    # Lower is better
    score = 0.0
    score += 5.0 * len(qc["qc_flags"])
    score += abs(qc["mean_brightness"] - 120.0) / 60.0
    return score

# -----------------------
# IMAGE FETCHING (Google -> ESRI fallback)
# -----------------------
def fetch_image_google(lat, lon, api_key, zoom=DEFAULT_ZOOM, size_px=IMG_SIZE_PX, scale=IMG_SCALE):
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{size_px}x{size_px}",
        "scale": scale,
        "maptype": "satellite",
        "key": api_key
    }
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB"), {"source": "Google Static Maps"}

def fetch_image_esri(lat, lon, zoom=DEFAULT_ZOOM, size_px=IMG_SIZE_PX):
    # Compute half-width in meters: half_pixels * meters_per_pixel
    mpp = meters_per_pixel(lat, zoom)
    half_m = (size_px / 2.0) * mpp
    # convert half_m to degrees properly via meters_to_latlon_offset
    dlat, dlon = meters_to_latlon_offset(lat, half_m)
    bbox = f"{lon-dlon},{lat-dlat},{lon+dlon},{lat+dlat}"
    url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    params = {"bbox": bbox, "bboxSR": "4326", "imageSR": "4326", "size": f"{size_px},{size_px}", "format": "png", "f": "image"}
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB"), {"source": "ESRI World Imagery"}

def fetch_best_image(lat, lon, api_key=None):
    candidates = []
    for latj, lonj in generate_jitter_points(lat, lon):
        img = None
        meta = {"source": "FAILED"}
        try:
            if api_key:
                img, meta = fetch_image_google(latj, lonj, api_key)
            else:
                raise Exception("no api key")
        except Exception:
            try:
                img, meta = fetch_image_esri(latj, lonj)
            except Exception:
                img = None
        if img:
            qc = basic_qc(img)
            candidates.append({"img": img, "qc": qc, "score": qc_badness_score(qc), "lat": latj, "lon": lonj, "meta": meta})
    if not candidates:
        return None
    # choose lowest badness score
    candidates.sort(key=lambda x: x["score"])
    best = candidates[0]
    # add retrieval timestamp
    best["meta"]["retrieved_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return best

# -----------------------
# INFERENCE & MASK -> POLYGON
# -----------------------
def masks_from_ultralytics_result(result):
    """
    Return list of (binary_mask_uint8, conf) sized HxW where masks are aligned to result.orig_shape.
    """
    H, W = result.orig_shape[:2]
    polygons = []
    # Ultralytics stores masks in results.masks.data (N,H',W') or results.masks.data with float masks; handle generically
    # If masks exist:
    if hasattr(result, "masks") and result.masks is not None:
        # under ultralytics v8: result.masks.data is a tensor (n, h, w) relative to orig_shape
        try:
            mask_t = result.masks.data.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy() if hasattr(result, "boxes") and result.boxes is not None else np.ones(len(mask_t))
            for m, c in zip(mask_t, confs):
                # ensure size matches orig_shape; if not, resize
                if m.shape != (H, W):
                    m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    m_resized = m
                mask_bin = (m_resized > 0.5).astype(np.uint8)
                polygons.append((mask_bin, float(c)))
            return polygons
        except Exception:
            pass
    # Fallback to boxes if masks absent
    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes.xyxy) > 0:
        bxy = result.boxes.xyxy.cpu().numpy()  # n x 4
        confs = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(bxy, confs):
            mask = np.zeros((H, W), dtype=np.uint8)
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cv2.rectangle(mask, (max(0, x1i), max(0, y1i)), (min(W-1, x2i), min(H-1, y2i)), 1, -1)
            polygons.append((mask, float(c)))
    return polygons

def mask_to_shapely_polygon(mask):
    # expects uint8 binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        coords = cnt.squeeze()
        if coords.ndim != 2 or coords.shape[0] < 3:
            continue
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area >= MIN_POLY_AREA_PX:
            polys.append(poly)
    if not polys:
        return None
    return unary_union(polys)

# -----------------------
# PIXEL <-> LATLON mapping
# -----------------------
def pixel_to_latlon_poly(shapely_poly_px, center_lat, center_lon, img_w, img_h, mpp):
    """
    Convert a Shapely polygon in pixel coords (x,y) where origin (0,0) is top-left,
    x increases to right, y increases downward,
    and image center pixel = (img_w/2, img_h/2) corresponds to (center_lat, center_lon).
    Returns polygon in lon/lat as list of [lon,lat] coordinate pairs.
    """
    if shapely_poly_px is None:
        return None
    coords_px = list(shapely_poly_px.exterior.coords)
    pts_lonlat = []
    for x_px, y_px in coords_px:
        # pixel offsets from center
        dx_px = x_px - (img_w / 2.0)
        dy_px = y_px - (img_h / 2.0)
        # meters offsets (east positive, north positive)
        dx_m = dx_px * mpp
        dy_m = -dy_px * mpp  # because pixel y down -> north is negative dy_px
        # convert to degrees
        dlat = dy_m / 110574.0
        dlon = dx_m / (111320.0 * math.cos(math.radians(center_lat)))
        lat = center_lat + dlat
        lon = center_lon + dlon
        pts_lonlat.append([round(lon, 7), round(lat, 7)])
    return pts_lonlat

# -----------------------
# Draw overlay
# -----------------------
def draw_audit_overlay(img_rgb, polygon_px, buffer_circle_px, meta_text_lines):
    # img_rgb: HxW BGR or RGB? we'll ensure BGR for cv2 drawing
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if img_rgb.shape[2] == 3 else img_rgb
    overlay = img_bgr.copy()
    H, W = overlay.shape[:2]
    # draw buffer: buffer_circle_px is shapely geometry in pixel coords; we'll draw its boundary
    try:
        bx, by = buffer_circle_px.exterior.xy
        pts = np.vstack((np.array(bx), np.array(by))).T.astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 0, 255), 3)
    except Exception:
        pass
    # draw polygon
    if polygon_px is not None:
        try:
            coords = np.array(list(polygon_px.exterior.coords), dtype=np.int32)
            cv2.fillPoly(overlay, [coords], (0, 255, 0))
            cv2.polylines(overlay, [coords], True, (0, 128, 0), 2)
        except Exception:
            pass
    # footer
    footer_h = 120
    footer = np.zeros((footer_h, W, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30
    for i, line in enumerate(meta_text_lines):
        color = (200, 200, 200)
        if i == 1 and "VERIFIABLE" in "".join(meta_text_lines):
            color = (0, 255, 0)
        cv2.putText(footer, line, (10, y), font, 0.7, color, 2, cv2.LINE_AA)
        y += 30
    final = np.vstack([overlay, footer])
    return final

# -----------------------
# Confidence calibration & panel count / capacity
# -----------------------
def calibrate_confidence(raw_conf, pv_area_m2):
    # simple transparent calibration:
    # area_factor in [0,1] saturating at CALIB_AREA_SATURATE_M2
    area_factor = min(1.0, pv_area_m2 / CALIB_AREA_SATURATE_M2)
    # combine raw_conf and area_factor with weighted blend; preserve monotonicity
    calibrated = float(raw_conf) * (0.6 + 0.4 * area_factor)
    # clamp
    calibrated = min(max(calibrated, FINAL_CONF_MIN), 0.995)
    return round(calibrated, 3)

def estimate_panel_count(pv_area_m2):
    if pv_area_m2 <= 0:
        return 0
    return max(1, int(round(pv_area_m2 / AVG_PANEL_M2)))

def estimate_capacity_kw(pv_area_m2):
    kw = (pv_area_m2 * WP_PER_M2) / 1000.0
    return round(kw, 3)

# -----------------------
# MAIN Pipeline CLASS
# -----------------------
class SolarPipeline:
    def __init__(self, model_path, output_dir, api_key=None):
        self.output_dir = output_dir
        self.art_dir = os.path.join(output_dir, "artifacts")
        os.makedirs(self.art_dir, exist_ok=True)
        print(f"Loading model from {model_path} ...")
        self.model = YOLO(model_path)
        self.api_key = api_key

    def process_row(self, sample_id, lat, lon):
        # Fetch best image (with jitter)
        best = fetch_best_image(lat, lon, self.api_key)
        if best is None:
            return self._not_verifiable_record(sample_id, lat, lon, reason="image_acquisition_failed")

        pil_img = best["img"]
        img_meta = best["meta"]
        qc = best["qc"]
        effective_lat = best["lat"]
        effective_lon = best["lon"]

        # quick QC fail on low resolution
        if "LOW_RESOLUTION" in qc["qc_flags"]:
            return self._not_verifiable_record(sample_id, lat, lon, reason="low_resolution", image_meta=img_meta)

        # Save raw
        raw_path = os.path.join(self.art_dir, f"{sample_id}_raw.png")
        pil_img.save(raw_path)

        # run inference
        # Ultralytics accepts path string
        results = self.model(raw_path, imgsz=IMG_SIZE_PX, conf=0.1, verbose=False)
        if len(results) == 0:
            return self._not_verifiable_record(sample_id, lat, lon, reason="inference_failed", image_meta=img_meta)
        res = results[0]

        # get masks or fallback boxes
        polygons = masks_from_ultralytics_result(res)  # list of (mask_uint8, conf)
        H_img, W_img = res.orig_shape[:2]
        mpp = meters_per_pixel(effective_lat)  # meters per pixel for this image/zoom

        # convert masks -> shapely polygons in pixel-space with confidences
        panels = []
        for mask, conf in polygons:
            # mask may be float — ensure properly thresholded
            mask_bin = (mask > 0).astype(np.uint8)
            poly = mask_to_shapely_polygon(mask_bin)
            if poly is None:
                continue
            if poly.area < MIN_POLY_AREA_PX:
                continue
            panels.append({"poly_px": poly, "conf": float(conf)})

        # Build image center point and buffer pixels (for both sqft radii)
        center_px = Point(W_img/2.0, H_img/2.0)
        chosen_panel = None
        chosen_buffer_sqft = None
        buffer_poly_px = None

        for buf_sqft in [1200, 2400]:
            radius_m = math.sqrt((buf_sqft * 0.092903) / math.pi)  # m
            radius_px = radius_m / mpp
            buf_circle_px = center_px.buffer(radius_px)
            # check overlaps
            overlaps = []
            for p in panels:
                try:
                    inter_area = p["poly_px"].intersection(buf_circle_px).area
                except Exception:
                    inter_area = 0
                if inter_area > 0:
                    overlaps.append((inter_area, p))
            if overlaps:
                # choose panel with max overlap area
                overlaps.sort(key=lambda x: x[0], reverse=True)
                chosen_panel = overlaps[0][1]
                chosen_buffer_sqft = buf_sqft
                buffer_poly_px = buf_circle_px
                break

        # Decide report if found or not
        now_date = datetime.utcnow().strftime("%Y-%m-%d")
        base_image_meta = {"source": img_meta.get("source", "unknown"), "capture_date": img_meta.get("date", now_date), "retrieved_at": img_meta.get("retrieved_at")}
        # If panel found:
        if chosen_panel:
            pv_area_m2 = chosen_panel["poly_px"].area * (mpp ** 2)
            raw_conf = chosen_panel["conf"]
            calibrated_conf = calibrate_confidence(raw_conf, pv_area_m2)
            panel_count = estimate_panel_count(pv_area_m2)
            capacity_kw = estimate_capacity_kw(pv_area_m2)
            # generate reason codes (geometry)
            min_rect = chosen_panel["poly_px"].minimum_rotated_rectangle
            rectangularity = chosen_panel["poly_px"].area / min_rect.area if min_rect.area > 0 else 0.0
            reasons = []
            if rectangularity > 0.85:
                reasons.append("rectilinear_array_geometry")
            elif rectangularity > 0.60:
                reasons.append("complex_polygon_geometry")
            if calibrated_conf > 0.85:
                reasons.append("high_confidence_module_pattern")
            if pv_area_m2 > 2.0:
                reasons.append("area_consistent_with_pv")
            if not reasons:
                reasons = ["visual_anomaly_detected"]

            # QC decisions: if imagery has cloud/shadow flags AND confidence isn't very high, mark NOT_VERIFIABLE
            qc_flags = qc["qc_flags"]
            if (("POSSIBLE_CLOUD_GLARE" in qc_flags or "POSSIBLE_SHADOW_DARK" in qc_flags) and calibrated_conf < QC_CONF_HIGH_THRESHOLD):
                qc_status = "NOT_VERIFIABLE"
                # When NOT_VERIFIABLE we should not commit has_solar True/False — spec: NOT_VERIFIABLE indicates insufficient evidence.
                has_solar_report = None
            else:
                qc_status = "VERIFIABLE"
                has_solar_report = True

            # convert polygon pixel -> lon/lat for mask/bbox field
            polygon_lonlat = pixel_to_latlon_poly(chosen_panel["poly_px"], effective_lat, effective_lon, W_img, H_img, mpp)

            record = {
                "sample_id": sample_id,
                "lat": float(lat),
                "lon": float(lon),
                "effective_lat": float(effective_lat),
                "effective_lon": float(effective_lon),
                "has_solar": has_solar_report,
                "confidence": float(calibrated_conf),
                "pv_area_sqm_est": round(pv_area_m2, 3),
                "panel_count_est": int(panel_count),
                "capacity_kw_est": capacity_kw,
                "buffer_radius_sqft": int(chosen_buffer_sqft),
                "qc_status": qc_status,
                "reason_codes": reasons,
                "bbox_or_mask": {
                    "type": "Polygon",
                    "coordinates": [polygon_lonlat]
                },
                "image_metadata": base_image_meta
            }

            # draw overlay and save
            img_np = np.array(pil_img)
            overlay = draw_audit_overlay(img_np, chosen_panel["poly_px"], buffer_poly_px, [
                f"ID: {sample_id} | SRC: {base_image_meta['source']}",
                f"QC: {qc_status} | CONF: {record['confidence']:.3f} | AREA: {record['pv_area_sqm_est']} m2",
                f"BUFFER: {record['buffer_radius_sqft']} sqft | REASONS: {', '.join(reasons[:3])}"
            ])
            overlay_path = os.path.join(self.art_dir, f"{sample_id}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay)
            record["audit_image_path"] = overlay_path
            # helpful for auditors
            record["raw_image_path"] = raw_path
            return record

        # If no panel found
        # Decide QC: if image flags indicate poor quality, NOT_VERIFIABLE; else VERIFIABLE no_pv_in_buffer
        qc_flags = qc["qc_flags"]
        if ("POSSIBLE_CLOUD_GLARE" in qc_flags or "POSSIBLE_SHADOW_DARK" in qc_flags or "LOW_RESOLUTION" in qc_flags):
            qc_status = "NOT_VERIFIABLE"
            has_solar_report = None
        else:
            qc_status = "VERIFIABLE"
            has_solar_report = False

        chosen_buffer_sqft = 2400
        # construct empty buffer polygon px for overlay
        radius_m = math.sqrt((chosen_buffer_sqft * 0.092903) / math.pi)
        buf_px = center_px.buffer(radius_m / mpp)
        overlay_img = draw_audit_overlay(np.array(pil_img), None, buf_px, [
            f"ID: {sample_id} | SRC: {base_image_meta['source']}",
            f"QC: {qc_status} | CONF: 0.00 | AREA: 0.0 m2",
            f"BUFFER: {chosen_buffer_sqft} sqft | REASONS: no_pv_in_buffer"
        ])
        overlay_path = os.path.join(self.art_dir, f"{sample_id}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay_img)

        record = {
            "sample_id": sample_id,
            "lat": float(lat),
            "lon": float(lon),
            "effective_lat": float(effective_lat),
            "effective_lon": float(effective_lon),
            "has_solar": has_solar_report,
            "confidence": 0.0,
            "pv_area_sqm_est": 0.0,
            "panel_count_est": 0,
            "capacity_kw_est": 0.0,
            "buffer_radius_sqft": int(chosen_buffer_sqft),
            "qc_status": qc_status,
            "reason_codes": ["no_pv_in_buffer"],
            "bbox_or_mask": None,
            "image_metadata": base_image_meta,
            "audit_image_path": overlay_path,
            "raw_image_path": raw_path
        }
        return record

    def _not_verifiable_record(self, sample_id, lat, lon, reason="image_acquisition_failed", image_meta=None):
        rec = {
            "sample_id": sample_id,
            "lat": float(lat),
            "lon": float(lon),
            "has_solar": None,
            "confidence": 0.0,
            "pv_area_sqm_est": 0.0,
            "panel_count_est": 0,
            "capacity_kw_est": 0.0,
            "buffer_radius_sqft": None,
            "qc_status": "NOT_VERIFIABLE",
            "reason_codes": [reason],
            "bbox_or_mask": None,
            "image_metadata": image_meta or {},
            "audit_image_path": None
        }
        return rec

# -----------------------
# ENTRY POINT
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Rooftop PV verification pipeline")
    p.add_argument("--input_file", required=True, help="CSV or XLSX file with columns sample_id, latitude, longitude")
    p.add_argument("--output_dir", required=True, help="Directory to save outputs")
    p.add_argument("--model_path", required=True, help="Path to YOLO .pt segmentation model")
    p.add_argument("--api_key", required=False, help="Google Static Maps API key (optional; ESRI fallback used if not present)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = SolarPipeline(args.model_path, args.output_dir, api_key=args.api_key or os.getenv("GOOGLE_API_KEY"))

    # load input
    if args.input_file.lower().endswith(".xlsx") or args.input_file.lower().endswith(".xls"):
        df = pd.read_excel(args.input_file)
    else:
        df = pd.read_csv(args.input_file)
    # ensure columns present
    required = {"sample_id", "latitude", "longitude"}
    if not required.issubset(set(df.columns)):
        print(f"Input file must contain columns: {required}. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    results = []
    for idx, row in df.iterrows():
        sample_id = row.get("sample_id", f"row{idx}")
        lat = row["latitude"]
        lon = row["longitude"]
        print(f"[{idx+1}/{len(df)}] Processing sample_id={sample_id} lat={lat} lon={lon} ...")
        try:
            rec = pipeline.process_row(sample_id, lat, lon)
            results.append(rec)
        except Exception as e:
            print(f"Error processing {sample_id}: {e}", file=sys.stderr)
            results.append({
                "sample_id": sample_id,
                "lat": float(lat),
                "lon": float(lon),
                "has_solar": None,
                "confidence": 0.0,
                "pv_area_sqm_est": 0.0,
                "panel_count_est": 0,
                "capacity_kw_est": 0.0,
                "buffer_radius_sqft": None,
                "qc_status": "NOT_VERIFIABLE",
                "reason_codes": [f"exception:{str(e)}"],
                "bbox_or_mask": None,
                "image_metadata": {},
                "audit_image_path": None
            })

    # final save
    out_json = os.path.join(args.output_dir, "final_predictions.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Results written to {out_json}. Artifacts in {os.path.join(args.output_dir, 'artifacts')}")

if __name__ == "__main__":
    main()
