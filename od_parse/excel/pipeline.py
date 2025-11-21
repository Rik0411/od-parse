"""
Excel Pipeline for parsing .xlsx and .xls files.

Uses DuckDB as an in-memory SQL engine and Gemini for intelligent
Bill of Materials (BOM) mapping when --mech flag is used.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import duckdb

from od_parse.excel.gemini_client import generate_duckdb_sql_for_bom
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


def runExcelPipeline(file_path: str, args) -> Dict[str, Any]:
    """
    Parse an Excel file (.xlsx or .xls) and extract structured data.
    
    If --mech flag is set, uses Gemini to map columns to standard BOM schema.
    Otherwise, returns raw data from all sheets.
    
    Args:
        file_path: Path to the Excel file
        args: argparse.Namespace object with .mech attribute
        
    Returns:
        Dictionary with structure:
        {
            "file_type": "excel",
            "sheets": [
                {
                    "sheet_name": "Sheet1",
                    "type": "BOM" or "Raw",
                    "rows": [ ... data ... ]
                }
            ]
        }
    """
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if args.mech and not api_key:
        logger.warning("--mech flag is set but GOOGLE_API_KEY not found. Falling back to raw data.")
        use_bom_mapping = False
    else:
        use_bom_mapping = args.mech
    
    # Load all sheets from Excel file
    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    except Exception as e:
        # Try with xlrd for .xls files
        try:
            all_sheets = pd.read_excel(file_path, sheet_name=None, engine='xlrd')
        except Exception as e2:
            logger.error(f"Failed to read Excel file: {e2}")
            raise ValueError(f"Cannot read Excel file {file_path}: {e2}")
    
    if not all_sheets:
        raise ValueError(f"No sheets found in Excel file: {file_path}")
    
    # Initialize DuckDB connection
    con = duckdb.connect(database=':memory:')
    
    sheets_result = []
    
    try:
        # Process each sheet
        for sheet_name, dataframe in all_sheets.items():
            try:
                # Skip empty sheets
                if dataframe.empty:
                    logger.warning(f"Sheet '{sheet_name}' is empty, skipping...")
                    sheets_result.append({
                        "sheet_name": sheet_name,
                        "type": "Raw",
                        "rows": []
                    })
                    continue
                
                # Register dataframe in DuckDB
                con.register('raw_data', dataframe)
                
                if use_bom_mapping:
                    # BOM mapping mode: use Gemini to generate SQL query
                    try:
                        columns = dataframe.columns.tolist()
                        logger.info(f"Generating BOM SQL query for sheet '{sheet_name}' with columns: {columns}")
                        
                        sql_query = generate_duckdb_sql_for_bom(columns, api_key)
                        logger.info(f"Generated SQL query: {sql_query}")
                        
                        # Execute SQL query
                        result_df = con.execute(sql_query).df()
                        
                        # Replace NaN/NaT with None so it serializes to valid JSON 'null'
                        result_df = result_df.where(pd.notnull(result_df), None)
                        
                        # Convert to list of dicts
                        rows = result_df.to_dict(orient='records')
                        
                        sheets_result.append({
                            "sheet_name": sheet_name,
                            "type": "BOM",
                            "rows": rows
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate BOM mapping for sheet '{sheet_name}': {e}. Falling back to raw data.")
                        # Fall back to raw data
                        # Replace NaN/NaT with None so it serializes to valid JSON 'null'
                        result_df = dataframe.where(pd.notnull(dataframe), None)
                        rows = result_df.to_dict(orient='records')
                        sheets_result.append({
                            "sheet_name": sheet_name,
                            "type": "Raw",
                            "rows": rows
                        })
                else:
                    # Raw mode: just dump the data
                    # Replace NaN/NaT with None so it serializes to valid JSON 'null'
                    result_df = dataframe.where(pd.notnull(dataframe), None)
                    rows = result_df.to_dict(orient='records')
                    sheets_result.append({
                        "sheet_name": sheet_name,
                        "type": "Raw",
                        "rows": rows
                    })
                
                # Unregister the table for next iteration
                con.unregister('raw_data')
                
            except Exception as e:
                logger.error(f"Error processing sheet '{sheet_name}': {e}", exc_info=True)
                # Add error sheet to results
                sheets_result.append({
                    "sheet_name": sheet_name,
                    "type": "Raw",
                    "rows": [],
                    "error": str(e)
                })
                # Try to unregister in case of error
                try:
                    con.unregister('raw_data')
                except:
                    pass
                continue
    
    finally:
        # Always close DuckDB connection
        con.close()
    
    # Build final result structure
    result = {
        "file_type": "excel",
        "sheets": sheets_result
    }
    
    return result

