# Eco-Ideathon  
## AI-Powered Rooftop Solar PV Detection & Area Estimation

This project implements an end-to-end **AI-based verification pipeline** for detecting rooftop solar photovoltaic (PV) installations using satellite imagery.  
It was developed for **EcoInnovators Ideathon 2026** under the theme of governance-ready solar verification.

---

## Problem Statement

Government solar subsidy programs require **reliable, scalable verification** of rooftop solar installations.  
Manual inspections are slow, expensive, and inconsistent.

This system answers:

> **Given a latitude and longitude, is a rooftop solar PV system installed at that location?**

If present, the system estimates the **total PV panel area** and generates **audit-friendly evidence**.

---

## Core Capabilities

- Satellite image retrieval using **Google Static Maps API**
- Solar panel detection using **YOLO segmentation**
- Buffer-based verification logic:
  - **1200 sq.ft** primary buffer
  - **2400 sq.ft** fallback buffer
- Selection of the panel with **maximum overlap** inside the buffer
- Estimation of:
  - PV area (m²)
  - Panel count
  - Capacity (kW)
- Quality Control (QC) classification:
  - `VERIFIABLE`
  - `NOT_VERIFIABLE`
- Explainable outputs with polygon masks and audit overlays

---

## Pipeline Overview

1. **Fetch**
   - Retrieve high-resolution satellite imagery for each `(lat, lon)`
   - Apply small coordinate jitters for robustness

2. **Classify**
   - Run YOLO segmentation to detect solar panels
   - Extract polygon masks for each detected instance

3. **Quantify**
   - Apply circular buffer search (1200 → 2400 sq.ft)
   - Select panel with **largest overlap**
   - Report **full area of selected panel (m²)**

4. **Explain & QC**
   - Generate reason codes
   - Assign QC status based on image quality and confidence

5. **Store**
   - Save results as structured JSON
   - Export visual audit artifacts (PNG/JPG)

---

## Input Format

Input file (`.csv` or `.xlsx`) must contain:

```text
sample_id, latitude, longitude
```

---

## Output Format (Per Sample)

```json
{
  "sample_id": 123,
  "lat": 12.9716,
  "lon": 77.5946,
  "has_solar": true,
  "confidence": 0.92,
  "pv_area_sqm_est": 23.5,
  "panel_count_est": 8,
  "capacity_kw_est": 4.4,
  "buffer_radius_sqft": 1200,
  "qc_status": "VERIFIABLE",
  "bbox_or_mask": "<GeoJSON polygon>",
  "image_metadata": {
    "source": "Google Static Maps",
    "capture_date": "YYYY-MM-DD"
  }
}
```

---

## Repository Structure

```
Eco-Ideathon/
│
├── pipeline/
│   └── infer.py                 # Main inference pipeline
│
├── environment/
│   ├── requirements.txt
│   ├── environment.yml
│   └── python_version.txt
│
├── model/
│   └── best.pt                  # Trained YOLO model
│
├── artifacts/
│   ├── *_raw.png
│   ├── *_qc.png
│   ├── *_mask.png
│   └── *_audit_overlay.jpg
│
├── training_logs/
│   └── training_metrics.csv
│
├── model_card/
│   └── model_card.pdf
│
└── README.md
```

---

## Running the Pipeline

```bash
python solar.py \
  --input_file input.csv \
  --output_dir outputs/ \
  --model_path model/best.pt \
  --api_key YOUR_GOOGLE_API_KEY
```

> **Note:** The API key is passed at runtime or via environment variables.  
> It is **never committed to the repository**.

---

## Assumptions & Notes

- PV area reported corresponds to the **full area of the selected panel**, not only the overlapping region.
- Capacity is estimated using a transparent assumption of **190 Wp/m²**.
- Image capture date is approximated by processing date due to API limitations.
- Designed for **auditability, not black-box inference**.

---

## License

This project uses open-source libraries and publicly permitted imagery sources.  
All datasets and tools are cited in the model card.

## Authors

- **Nithin**
- **Sam**
