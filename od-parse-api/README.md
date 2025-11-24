# Intelligent File Parser API

FastAPI REST API for intelligent file parsing supporting PDF, images, Excel, and vector files.

## Features

- **Multi-format Support**: PDF, PNG, JPG, JPEG, Excel (.xlsx, .xls), and vector files (.dxf, .dwg)
- **Intelligent Routing**: Automatically routes files to appropriate parsing pipelines
- **PDF Triage**: Automatically detects vector vs raster PDFs and uses optimal pipeline
- **Mechanical Drawing Pipeline**: Specialized pipeline for technical drawings (optional)
- **Excel BOM Mapping**: Intelligent Bill of Materials mapping using Gemini AI

## Quick Start

### Prerequisites

- Python 3.11+
- Poppler (for PDF processing)
  - macOS: `brew install poppler`
  - Linux: `sudo apt-get install poppler-utils`
  - Windows: Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases)

### Installation

1. Clone the repository and navigate to the API directory:
```bash
cd od-parse-api
```

2. Install dependencies:

**Option A: Using setup script (Recommended)**
- **Windows**: Run `setup.bat`
- **Linux/Mac**: Run `chmod +x setup.sh && ./setup.sh`

**Option B: Manual installation**
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```bash
cp .env.example .env
```

4. Edit `.env` and add your API keys:
```
GOOGLE_API_KEY=your_google_api_key_here
ROBOFLOW_API_KEY=your_roboflow_api_key_here  # Optional
```

5. Run the API:

**Option A: Using run script (Recommended)**
- **Windows**: Run `run.bat`
- **Linux/Mac**: Run `chmod +x run.sh && ./run.sh`

**Option B: Manual run**
```bash
python -m uvicorn app.main:app --reload
```

**Note**: If `uvicorn` command is not found, use `python -m uvicorn` instead.

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /api/parse

Parse an uploaded file.

**Parameters:**
- `file` (form-data): File to parse (required)
- `mech_mode` (query, bool): Use mechanical drawing pipeline (default: false)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/parse?mech_mode=false" \
  -F "file=@sample.pdf"
```

**Response:**
```json
{
  "text": "...",
  "tables": [...],
  "images": [...],
  "metadata": {...}
}
```

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### GET /api/info

Get API information and supported file types.

**Response:**
```json
{
  "app_name": "Intelligent File Parser API",
  "version": "1.0.0",
  "supported_extensions": [".pdf", ".png", ".jpg", ".jpeg", ".xlsx", ".xls", ".dxf", ".dwg"],
  "max_file_size_mb": 100
}
```

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t od-parse-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_key \
  -e ROBOFLOW_API_KEY=your_key \
  od-parse-api
```

## Configuration

Environment variables (see `.env.example`):

- `GOOGLE_API_KEY`: Required - Google Gemini API key
- `ROBOFLOW_API_KEY`: Optional - Roboflow API key for mechanical drawing pipeline
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 100)
- `TEMP_DIR`: Temporary directory for file uploads (default: /tmp/od-parse)
- `LOG_LEVEL`: Logging level (default: INFO)

## Supported File Types

- **PDF** (.pdf): Intelligent triage between vector and raster pipelines
- **Images** (.png, .jpg, .jpeg): Raster pipeline with optional mechanical drawing mode
- **Excel** (.xlsx, .xls): DuckDB + Gemini BOM mapping (with `mech_mode=true`)
- **Vector** (.dxf, .dwg): Vector file parsing (simulation)

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid file, unsupported format)
- `413`: Payload Too Large (file exceeds max size)
- `500`: Internal Server Error (processing failure)
- `503`: Service Unavailable (API keys missing, external service down)

## Development

### Project Structure

```
od-parse-api/
├── app/
│   ├── main.py              # FastAPI app initialization
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   └── dependencies.py  # Dependency injection
│   └── core/
│       ├── config.py        # Configuration management
│       └── router.py        # File routing logic
├── requirements.txt
├── Dockerfile
└── README.md
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

## License

See parent repository for license information.

