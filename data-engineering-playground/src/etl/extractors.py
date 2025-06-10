"""
Data extractors for financial data from various sources
"""
import requests
import os
import json
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Literal

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataExtractor:
    """
    Base class for data extractors
    """

    def extract(self, source_path: Union[str, Path]) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Extract data from the source

        Args:
            source_path (Union[str, Path]): Path to the data source

        Returns:
            pd.DataFrame | dict[str, pd.DataFrame]: Extracted data as a pandas DataFrame or a dictionary of DataFrames
        """
        raise NotImplementedError("Subclasses must implement extract method")


class CSVExtractor(DataExtractor):
    """
    Extractor for CSV files
    """

    def extract(self, source_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Extract data from CSV file

        Args:
            source_path (Union[str, Path]): Path to the CSV file
            **kwargs: Additional arguments to pass to pandas.read_csv

        Returns:
            pd.DataFrame: Extracted data as a pandas DataFrame
        """
        try:
            logger.info(f"Extracting data from CSV file: {source_path}")
            df = pd.read_csv(source_path, **kwargs)
            logger.info(f"Successfully extracted {len(df)} rows from {source_path}")
            return df
        except Exception as e:
            logger.error(f"Error extracting data from CSV file: {e}")
            raise


class ParquetExtractor(DataExtractor):
    """
    Extractor for Parquet files
    """

    def extract(self, source_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Extract data from Parquet file

        Args:
            source_path (Union[str, Path]): Path to the Parquet file
            **kwargs: Additional arguments to pass to pandas.read_parquet

        Returns:
            pd.DataFrame: Extracted data as a pandas DataFrame
        """
        try:
            logger.info(f"Extracting data from Parquet file: {source_path}")
            df = pd.read_parquet(source_path, **kwargs)
            logger.info(f"Successfully extracted {len(df)} rows from {source_path}")
            return df
        except Exception as e:
            logger.error(f"Error extracting data from Parquet file: {e}")
            raise


class ExcelExtractor(DataExtractor):
    """
    Extractor for Excel files
    """

    def extract(self, source_path: Union[str, Path], sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Extract data from Excel file

        Args:
            source_path (Union[str, Path]): Path to the Excel file
            sheet_name (Optional[str], optional): Name of the sheet to extract. Defaults to None.
            **kwargs: Additional arguments to pass to pandas.read_excel

        Returns:
            pd.DataFrame | dict[str, pd.DataFrame]: Extracted data as a pandas DataFrame or a dictionary of DataFrames
        """
        try:
            logger.info(f"Extracting data from Excel file: {source_path}")
            df = pd.read_excel(source_path, sheet_name=sheet_name, **kwargs)
            logger.info(f"Successfully extracted {len(df)} rows from {source_path}")
            return df
        except Exception as e:
            logger.error(f"Error extracting data from Excel file: {e}")
            raise


class JSONExtractor(DataExtractor):
    """
    Extractor for JSON files
    """

    def extract(self, source_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Extract data from JSON file

        Args:
            source_path (Union[str, Path]): Path to the JSON file
            **kwargs: Additional arguments to pass to pandas.read_json

        Returns:
            pd.DataFrame: Extracted data as a pandas DataFrame
        """
        try:
            logger.info(f"Extracting data from JSON file: {source_path}")
            df = pd.read_json(source_path, **kwargs)
            logger.info(f"Successfully extracted {len(df)} rows from {source_path}")
            return df
        except Exception as e:
            logger.error(f"Error extracting data from JSON file: {e}")
            raise


class AlphaVantageExtractor(DataExtractor):
    """
    Extractor for Alpha Vantage API data
    """

    def extract(self, api_key: str, symbol: str, function: str | Literal['EARNINGS', 'CASH_FLOW', 'BALANCE_SHEET', 'INCOME_STATEMENT', 'SPLITS', 'DIVIDENDS'] = 'CASH_FLOW', **kwargs) -> None:
        """
        Extract data from Alpha Vantage API and save it in json format to datafolder

        Args:
            api_key (str): Your Alpha Vantage API key
            symbol (str): Stock symbol to query
            function (str, optional): API function to call. Defaults to 'CASH_FLOW'.
            **kwargs: Additional arguments for the API request

        Returns:
            pd.DataFrame: Extracted data as a pandas DataFrame
        """
        try:
            logger.info(f"Extracting data from Alpha Vantage for symbol: {symbol}")
            url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}"
            response = requests.get(url, **kwargs)
            response.raise_for_status()
            data = response.json()
            with open(f"data/raw/alpha_vantage_{function}_{symbol}.json", "w") as f:
                json.dump(data, f)
            logger.info(f"Successfully extracted data from Alpha Vantage for symbol {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error extracting data from Alpha Vantage: {e}")
            raise
