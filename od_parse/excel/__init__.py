"""
Excel parsing pipeline module.

This module provides Excel file parsing capabilities using DuckDB and Gemini
for intelligent Bill of Materials (BOM) mapping.
"""

from od_parse.excel.pipeline import runExcelPipeline
from od_parse.excel.gemini_client import generate_duckdb_sql_for_bom, generate_excel_image_description

__all__ = ['runExcelPipeline', 'generate_duckdb_sql_for_bom', 'generate_excel_image_description']

