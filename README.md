od-parse-fork: Intelligent Drawing Parser

This project extends the base `od-parse` library to create an intelligent, multi-pipeline system for parsing engineering drawings and other documents.

It features a smart **triage router** that analyzes input files (`.pdf`, `.jpg`, `.png`, `.dxf`, `.dwg`, etc.) and routes them to the most efficient and accurate parsing pipeline available. This fork introduces a specialized, high-accuracy pipeline for mechanical drawings, activated with the `--mech` flag.

## Features

- **Intelligent File Routing**: Automatically handles `.pdf`, `.jpg`, `.png`, `.dxf`, and `.dwg` files.
- **Smart PDF Triage**: Uses PyMuPDF (`fitz`) to detect if a PDF is Vector (text-based) or Raster (scanned image) and routes accordingly.
- **High-Accuracy Mechanical Pipeline (`--mech`)**:
  - **Vector Path**: Fast, cheap, and highly accurate text-only pipeline using PyMuPDF and Gemini text batch processing.
  - **Raster Path**: Robust, 3-stage image pipeline (Roboflow → Gemini batch image parse → Gemini Stage 3 “Safety Net”) to ensure maximum accuracy and find missed values.
- **Default Parser Integration**: Falls back to the original `od-parse` single-stage LLM parser when the `--mech` flag is not used.
- **Multi-Page PDF Support**: Processes all pages in a PDF, not just the first.
- **Resilient API Calls**: Built-in batching, retries, and backoff logic to handle API rate limits (429) and server errors (503).

## How to Use

The main entry point is `parse_pdf.py`.

### Prerequisites

- **Python dependencies**:

  ```bash
  pip install -r requirements.txt
  # Key libraries: PyMuPDF, pdf2image, roboflow, requests, google-generativeai, python-dotenv
  ```

- **Poppler (for `pdf2image`)**  
  This is required for the Raster PDF pipeline.

  - macOS:

    ```bash
    brew install poppler
    ```

  - Linux:

    ```bash
    sudo apt-get install poppler-utils
    ```

  - Windows: Download the Poppler binaries, unzip, and either:
    - Add the `bin` folder to your system `PATH`, or
    - Set the `POPPLER_PATH` environment variable, for example:

      ```text
      POPPLER_PATH=C:\Poppler\poppler-25.07.0\Library\bin
      ```

### Environment Variables

Set these in your shell or in a `.env` file in the project root:

- `GOOGLE_API_KEY`: Google Gemini API key (used for all LLM calls).
- `ROBOFLOW_API_KEY`: Roboflow API key (used when local inference server is not available).
- `POPPLER_PATH` (Windows only, optional if Poppler is on `PATH`): Path to the Poppler `bin` directory.

Example `.env`:

```env
GOOGLE_API_KEY=your-google-api-key
ROBOFLOW_API_KEY=your-roboflow-api-key
POPPLER_PATH=C:\Poppler\poppler-25.07.0\Library\bin
```

### Roboflow Server (Optional but Recommended)

For the fastest raster detection, run the local Roboflow server:

```bash
pip install inference-cli
inference server start
```

If the server is not running, the pipeline will automatically fall back to using the (slower) Roboflow cloud API.

## Command-Line Usage

Basic command:

```bash
python parse_pdf.py <path_to_file> [flags]
```

### Examples

- **Parse a mechanical drawing (Vector PDF)**  
  Fastest, most accurate path:

  ```bash
  python parse_pdf.py "C:\Drawings\sample_vector.pdf" --mech --output "output\sample_vector.json"
  ```

- **Parse a mechanical drawing (scanned image/PDF)**  
  Uses the 3-stage Roboflow + Gemini pipeline:

  ```bash
  python parse_pdf.py "C:\Drawings\scanned_drawing.pdf" --mech
  ```

- **Parse a standard (non-mech) PDF**  
  Uses the default `od-parse` single-stage LLM parser:

  ```bash
  python parse_pdf.py "C:\Documents\standard_doc.pdf"
  ```

- **Parse a DXF/DWG file (simulation)**:

  ```bash
  python parse_pdf.py "C:\Drawings\cad_file.dxf"
  python parse_pdf.py "C:\Drawings\cad_file.dwg"
  ```

## Architecture: Intelligent Triage Pipeline

The `parse_pdf.py` script acts as a master controller. High-level decision logic:

1. **Check file extension**:
   - `.dxf` / `.dwg` → Vector pipeline (simulated).
   - `.jpg` / `.png` → Raster pipeline.
   - `.pdf` → PDF triage pipeline.

### PDF Triage Pipeline (`runPdfPipeline`)

1. Open PDF with PyMuPDF (`fitz`).
2. Check if the document contains vector text.
3. Routing:
   - If **vector text is found** and `--mech` is `True` → `runVectorPdfPipeline` (Path A).
   - Else (raster PDF or no `--mech` flag) → `runRasterPdfPipeline` (Path B).

### Raster Pipeline (`runRasterPdfPipeline` or direct image)

1. Convert all PDF pages to temporary PNGs (if starting from PDF).
2. Loop through each image page:
   - If `--mech` is `True`:
     - Call `runFullHybridPipeline` (3-stage mechanical pipeline).
   - If `--mech` is `False`:
     - Call the default single-stage LLM pipeline (original `od-parse` parser).

## Mechanical Drawing Pipelines (`--mech`)

This is the core of the new functionality, providing two distinct paths for maximum accuracy.

### Path A: Vector PDF Pipeline (Fast & Accurate)

Used for digital-born PDFs that have selectable text. It **skips** `pdf2image` and **skips** Roboflow entirely.

Per-page flow (`runVectorPdfPipeline`):

1. **Extract text**: PyMuPDF (`fitz`) reads all text strings (e.g., `R10`, `36.5`) directly from each PDF page.
2. **Batch parse**: All text strings for a page are sent in a single API call to `stage2_runBatchTextParsing` (Gemini text-only).
3. **Aggregate**: The resulting JSON annotations are merged into the final output.

### Path B: Raster PDF/Image Pipeline (Robust 3-Stage)

Used for scanned PDFs or flat images (`.jpg`, `.png`). It is designed to be highly robust and find all annotations, even if the detector (Roboflow) is imperfect.

Per-page flow (`runFullHybridPipeline`):

1. **Stage 1 (Detect): `stage1_runRoboflowDetection`**
   - The full image is sent to the Roboflow `eng-drawing-ukrvj/3` model.
   - It returns a list of potential annotations (e.g., “dimension”, “radius”) with a `0.05` confidence threshold.

2. **Stage 2 (Parse): `stage2_runBatchVerification`**
   - The pipeline crops all the patches found in Stage 1.
   - All image patches are sent in a single batch API call to Gemini (multimodal).
   - Gemini verifies each patch (is it a false positive?) and parses the value.

3. **Stage 3 (Safety Net): `stage3_runMissingItemScan`**
   - The pipeline sends the full image **and** the list of items found in Stage 2 to Gemini.
   - It asks Gemini to find any annotations that Roboflow missed.
   - This catches the “false negatives” from Stage 1.

4. **Aggregate**: The results from Stage 2 and Stage 3 are combined into the final JSON for that page.