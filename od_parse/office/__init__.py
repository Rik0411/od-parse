"""
Office document parsing pipelines for Word (.docx) and PowerPoint (.pptx) files.
"""

from .docx_pipeline import runDocxPipeline
from .pptx_pipeline import runPptxPipeline

__all__ = ["runDocxPipeline", "runPptxPipeline"]

