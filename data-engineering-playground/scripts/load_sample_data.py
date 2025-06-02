"""
Script to load sample financial data into the database
"""
from src.utils.logger import setup_logger
from src.etl.loaders import StockPriceLoader, CompanyLoader
from src.etl.transformers import StockPriceTransformer, CompanyDataTransformer
from src.etl.extractors import CSVExtractor, ParquetExtractor, ExcelExtractor
from src.database.connection import DatabaseConnection
import os
import sys
import pandas as pd
from pathlib import Path
import argparse

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))


logger = setup_logger("load_sample_data")


def load_sample_data(data_path: str, data_type: str, file_format: str):
    """
    Load sample data into the database

    Args:
        data_path (str): Path to the data file
        data_type (str): Type of data ('stock' or 'company')
        file_format (str): Format of the file ('csv', 'parquet', 'excel')
    """
    try:
        logger.info(f"Loading sample {data_type} data from {data_path}")

        # 1. Extract data
        if file_format == 'csv':
            extractor = CSVExtractor()
        elif file_format == 'parquet':
            extractor = ParquetExtractor()
        elif file_format == 'excel':
            extractor = ExcelExtractor()
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        df = extractor.extract(data_path)
        # Handle case where extractor returns a dict of DataFrames (e.g., Excel with multiple sheets)
        if isinstance(df, dict):
            logger.info(f"Extracted {sum(len(sheet) for sheet in df.values())} rows of data from {len(df)} sheets")
            # For simplicity, concatenate all sheets into one DataFrame
            df = pd.concat(df.values(), ignore_index=True)
        else:
            logger.info(f"Extracted {len(df)} rows of data")

        # 2. Transform data
        if data_type == 'stock':
            transformer = StockPriceTransformer()
        elif data_type == 'company':
            transformer = CompanyDataTransformer()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        transformed_df = transformer.transform(df)
        logger.info(f"Transformed data: {len(transformed_df)} rows")

        # 3. Load data
        db_conn = DatabaseConnection()
        with db_conn:
            if data_type == 'stock':
                loader = StockPriceLoader(db_conn)
                rows_loaded = loader.load(transformed_df)
            elif data_type == 'company':
                loader = CompanyLoader(db_conn)
                rows_loaded = loader.load(transformed_df)

        logger.info(f"Successfully loaded {rows_loaded} rows into the database")
        return True
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load sample financial data into the database')
    parser.add_argument('--data-path', required=True, help='Path to the data file')
    parser.add_argument('--data-type', required=True, choices=['stock', 'company'],
                        help='Type of data: stock or company')
    parser.add_argument('--file-format', required=True, choices=['csv', 'parquet', 'excel'],
                        help='Format of the data file: csv, parquet, or excel')

    args = parser.parse_args()

    result = load_sample_data(args.data_path, args.data_type, args.file_format)
    sys.exit(0 if result else 1)
