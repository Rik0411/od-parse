od-parse-fork: Intelligent Drawing Parser

This project extends the base od-parse library to create an intelligent, multi-pipeline system for parsing engineering drawings and other documents.

It features a smart "triage" router that analyzes input files (.pdf, .jpg, .dxf, etc.) and routes them to the most efficient and accurate parsing pipeline available. This fork introduces a specialized, high-accuracy pipeline for mechanical drawings, activated with the --mech flag.

Features

Intelligent File Routing: Automatically handles .pdf, .jpg, .png, .dxf, and .dwg files.

Smart PDF Triage: Uses PyMuPDF (fitz) to detect if a PDF is Vector (text-based) or Raster (scanned image) and routes accordingly.

High-Accuracy Mechanical Pipeline (--mech):

Vector Path: A fast, cheap, and highly accurate text-only pipeline using PyMuPDF and Gemini Text Batch processing.

Raster Path: A robust, 3-stage image pipeline (Roboflow -> Gemini Batch Image Parse -> Gemini Stage 3 "Safety Net") to ensure maximum accuracy and find missed values.

Default Parser Integration: Falls back to the original od-parse single-stage LLM parser when the --mech flag is not used.

Multi-Page PDF Support: Processes all pages in a PDF, not just the first.

Resilient API Calls: Built-in batching, retries, and backoff logic to handle API rate limits (429) and server errors (503).

How to Use

The main entry point is parse_pdf.py.

Prerequisites

Python Dependencies:

pip install -r requirements.txt
# Key libraries: PyMuPDF, pdf2image, roboflow, requests, google-generativeai


Poppler (for pdf2image):
This is required for the Raster PDF Pipeline.

macOS: brew install poppler

Linux: sudo apt-get install poppler-utils

Windows: Download the Poppler binaries, unzip, and either:

- Add the `bin` folder to your system's PATH, or
- Set the `POPPLER_PATH` environment variable, for example:

  `POPPLER_PATH=C:\Poppler\poppler-25.07.0\Library\bin`


Environment Variables:

- `GOOGLE_API_KEY`: Google Gemini API key (used for all LLM calls).
- `ROBOFLOW_API_KEY`: Roboflow API key (used when local inference server is not available).
- `POPPLER_PATH` (Windows only, optional if Poppler is on PATH): Path to the Poppler `bin` directory.

These can be set in your shell environment or via a `.env` file in the project root (loaded with `python-dotenv`).

Roboflow Server (Optional but Recommended):
For the fastest raster detection, run the local Roboflow server.

pip install inference-cli
inference server start


If the server is not running, the pipeline will automatically fall back to using the (slower) Roboflow cloud API.

Command-Line Usage

Basic Command:

python parse_pdf.py <path_to_file> [flags]


Examples:

Parse a Mechanical Drawing (Vector PDF):
This is the fastest, most accurate path.

python parse_pdf.py "C:\Drawings\sample_vector.pdf" --mech --output "output\sample_vector.json"


Parse a Mechanical Drawing (Scanned Image/PDF):
This will use the 3-stage Roboflow + Gemini pipeline.

python parse_pdf.py "C:\Drawings\scanned_drawing.pdf" --mech


Parse a standard (non-mech) PDF:
This will use the default od-parse single-stage LLM parser.

python parse_pdf.py "C:\Documents\standard_doc.pdf"


Parse a DXF/DWG file (Simulation):

python parse_pdf.py "C:\Drawings\cad_file.dxf"
python parse_pdf.py "C:\Drawings\cad_file.dwg"


Architecture: The Intelligent Triage Pipeline

The parse_pdf.py script acts as a master controller. Here is the decision logic:

Check File Extension:

.dxf / .dwg -> Vector Pipeline (Simulated)

.jpg / .png -> Raster Pipeline

.pdf -> PDF Triage Pipeline

PDF Triage Pipeline (runPdfPipeline):

Open PDF with PyMuPDF (fitz).

Check if the document contains vector text.

IF Vector Text is found AND --mech is True:

Route to runVectorPdfPipeline (Path A).

ELSE (Raster PDF or no --mech flag):

Route to runRasterPdfPipeline (Path B).

Raster Pipeline (runRasterPdfPipeline or direct image):

Convert all PDF pages to temporary PNGs (if needed).

Loop through each image page:

IF --mech is True:

Call runFullHybridPipeline (Path B).

IF --mech is False:

Call run_single_stage_llm_pipeline (Default od-parse parser).

Mechanical Drawing Pipelines (--mech)

This is the core of the new functionality, providing two distinct paths for maximum accuracy.

Path A: Vector PDF Pipeline (Fast & Accurate)

This pipeline is used for digital-born PDFs that have selectable text. It SKIPS pdf2image and SKIPS Roboflow entirely.

Page Loop (runVectorPdfPipeline):

Extract Text: PyMuPDF (fitz) reads all text strings (e.g., "R10", "36.5") directly from the PDF page.

Batch Parse: All text strings are sent in a single API call to stage2_runBatchTextParsing (Gemini Text-only).

Aggregate: The resulting JSON annotations are added to the final output.

Path B: Raster PDF/Image Pipeline (Robust 3-Stage)

This pipeline is used for scanned PDFs or flat images (.jpg, .png). It is designed to be highly robust and find all annotations, even if the detector (Roboflow) is imperfect.

Page Loop (runFullHybridPipeline):

Stage 1 (Detect): stage1_runRoboflowDetection

The full image is sent to the Roboflow eng-drawing-ukrvj/3 model.

It returns a list of potential annotations (e.g., "dimension", "radius") with a 0.05 confidence.

Stage 2 (Parse): stage2_runBatchVerification

The pipeline crops all the patches found in Stage 1.

All image patches are sent in a single batch API call to Gemini (multimodal).

Gemini verifies each patch (is it a false positive?) and parses the value.

Stage 3 (Safety Net): stage3_runMissingItemScan

The pipeline sends the full image AND the list of items found in Stage 2 to Gemini.

It asks Gemini to find any annotations that Roboflow missed.

This catches the "false negatives" from Stage 1.

Aggregate: The results from Stage 2 and Stage 3 are combined into the final JSON for that page.