# Rooftop Solar PV Verification Pipeline

## Overview

This repository contains an end-to-end **Rooftop Solar Photovoltaic (PV) Verification Pipeline** implemented in Python.  
The pipeline uses satellite imagery and a YOLO-based segmentation model to detect rooftop solar panels, estimate their area and capacity, and generate auditable outputs.

The system is designed to be deterministic, explainable, and suitable for large-scale rooftop verification tasks.

---

## Key Features

- Reads rooftop coordinates from CSV or XLSX files
- Fetches satellite imagery using:
  - Google Static Maps (preferred)
- Applies spatial jittering to handle coordinate noise
- Performs image quality checks (clouds, shadows, resolution)
- Runs YOLO segmentation inference
- Applies rooftop buffer validation (1200 sqft, fallback to 2400 sqft)
- Estimates:
  - PV area (m²)
  - Panel count
  - Installed capacity (kW)
- Generates GeoJSON-compatible polygon outputs
- Saves visual audit overlays for verification

---

## Input Data Format

### Supported Formats
- `.xlsx`

### Required Columns

| Column Name | Description |
|------------|------------|
| `sample_id` | Unique identifier for the rooftop |
| `latitude` | Latitude (WGS84) |
| `longitude` | Longitude (WGS84) |




## Pipeline Workflow

### 1. Image Acquisition

- Downloads satellite imagery centered on the provided coordinates  
- Applies ±10 m spatial jitter to reduce geolocation error  
- Selects the best image using a quality scoring heuristic  

---

### 2. Image Quality Control

Images are flagged for:

- Low resolution  
- Excessive brightness (cloud glare)  
- Excessive darkness (shadowing)  

Poor-quality images are marked as **NOT_VERIFIABLE**.

---

### 3. Segmentation Inference

- Uses Ultralytics YOLO segmentation models  
- Extracts pixel-level segmentation masks  
- Falls back to bounding boxes if masks are unavailable  
- Filters detections using minimum area and confidence thresholds  

---

### 4. Rooftop Buffer Validation

PV detections are validated using concentric rooftop buffers:

- **1200 sqft buffer** (preferred)  
- **2400 sqft buffer** (fallback)  

---

### 5. Solar Estimation

For valid detections, the pipeline computes:

- Estimated PV area (m²)  
- Estimated number of panels  
- Estimated installed capacity (kW)  
- Calibrated confidence score  

---

## Running the Pipeline
### Command-Line Usage
python infer.py \
  --input_file data/input.csv \
  --output_dir outputs/run_01 \
  --model_path models/yolo_seg.pt \
  --api_key YOUR_GOOGLE_MAPS_API_KEY
### Arguments
Argument	Description<br><br>
--input_file	CSV/XLSX with rooftop coordinates<br>
--output_dir	Directory to store outputs<br>
--model_path	Path to YOLO .pt segmentation model<br>
--api_key	Google Static Maps API key (optional)<br>

## Failure Handling

The pipeline robustly handles:<br>

-Image acquisition failures<br>

-Poor image quality<br>

-Inference errors<br>

Such cases are explicitly marked as NOT_VERIFIABLE with reason codes.<br>

## Design Principles

-Conservative decision-making under uncertainty<br>

-Deterministic and reproducible inference<br>

-Explicit confidence calibration<br>

-Geometry-aware validation<br>

-Audit-friendly outputs<br>
## Outputs

###  Prediction File

The pipeline generates a consolidated prediction file:

Each record includes:

```json
{
  "sample_id": "ID_001",
  "lat": 12.9716,
  "lon": 77.5946,
  "has_solar": true,
  "confidence": 0.82,
  "pv_area_sqm_est": 18.4,
  "panel_count_est": 10,
  "capacity_kw_est": 3.5,
  "buffer_radius_sqft": 1200,
  "qc_status": "VERIFIABLE",
  "bbox_or_mask": {
    "type": "Polygon",
    "coordinates": [...]
  },
  "image_metadata": {
    "source": "Google Static Maps",
    "retrieved_at": "2025-01-01T12:00:00Z"
  }
}

```
## License
This project is intended for academic, research, and evaluation use.




