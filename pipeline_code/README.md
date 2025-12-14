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
  - ESRI World Imagery (fallback)
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
- `.csv`
- `.xlsx`

### Required Columns

| Column Name | Description |
|------------|------------|
| `sample_id` | Unique identifier for the rooftop |
| `latitude` | Latitude (WGS84) |
| `longitude` | Longitude (WGS84) |

### Example

```csv
sample_id,latitude,longitude
ID_001,12.9716,77.5946
ID_002,19.0760,72.8777

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

## Outputs

### 1. Prediction File

The pipeline generates a consolidated prediction file:


