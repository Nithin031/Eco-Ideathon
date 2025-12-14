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
## API Key Configuration (Required)

Satellite imagery is fetched using **Google Static Maps**, which requires a valid API key with billing enabled.

**API keys are not included in this repository.**

### Step 1: Create a Google Static Maps API Key

1. Go to Google Cloud Console  
2. Create or select a project  
3. Enable **Maps Static API**  
4. Enable billing  
5. Generate an API key  
6. (Recommended) Restrict the key:
   - Limit to Maps Static API
   - Apply a billing or usage cap

The pipeline can be executed with or without explicitly passing the Google Maps API key, depending on how the key is provided.

---

### Step 2: Provide the API Key (Two Supported Methods)

#### Option A: Environment Variable (Recommended)

This is the preferred and secure method, especially for server execution.

**Linux / Server**
```bash
export GOOGLE_API_KEY="YOUR_GOOGLE_MAPS_API_KEY"
```
### Case 1: API Key Passed via Command-Line Argument

Use this method if the API key is **not set as an environment variable**.

```bash
python infer.py \
  --input_file data/input.csv \
  --output_dir outputs/run_01 \
  --model_path models/yolo_seg.pt \
  --api_key YOUR_GOOGLE_MAPS_API_KEY
```
Use this method if the API key is already set in the environment as GOOGLE_API_KEY

```bash
python infer.py \
  --input_file data/input.csv \
  --output_dir outputs/run_01 \
  --model_path models/yolo_seg.pt
```

### Arguments
### Command-Line Arguments

| Argument | Description |
|--------|------------|
| `--input_file` | Path to the input XLSX/CSV file containing `sample_id`, `latitude`, and `longitude` |
| `--output_dir` | Directory where prediction JSON and audit artifacts will be saved |
| `--model_path` | Path to the provided YOLO segmentation model (`.pt`) included in this repository |
| `--api_key` | Google Static Maps API key (optional; can also be provided via environment variable) |

**Note:**  
A trained YOLO segmentation model (`.pt`) is included as part of the submission.  
No model training is required to run inference.

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




