"""
Data transformers for financial data
"""
import pandas as pd
from typing import Dict, Any, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataTransformer:
    """
    Base class for data transformers
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Transform the data

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame | dict[str, pd.DataFrame]: Transformed DataFrame or a dictionary of DataFrames
        """
        raise NotImplementedError("Subclasses must implement transform method")


class StockPriceTransformer(DataTransformer):
    """
    Transformer for stock price data
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform stock price data

        Args:
            df (pd.DataFrame): Raw stock price DataFrame

        Returns:
            pd.DataFrame: Transformed stock price DataFrame
        """
        try:
            logger.info("Transforming stock price data")

            # Make a copy to avoid modifying the original
            result_df = df.copy()

            # Ensure column names are standardized
            column_mapping = {
                'Symbol': 'symbol',
                'Date': 'date',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Adj Close': 'adjusted_close',
                'Volume': 'volume'
            }

            # Rename columns that exist in the DataFrame
            existing_columns = {k: v for k, v in column_mapping.items() if k in result_df.columns}
            if existing_columns:
                result_df = result_df.rename(columns=existing_columns)

            # Ensure date is in correct format
            if 'date' in result_df.columns and not pd.api.types.is_datetime64_any_dtype(result_df['date']):
                result_df['date'] = pd.to_datetime(result_df['date'])

            # Sort by symbol and date
            if 'symbol' in result_df.columns and 'date' in result_df.columns:
                result_df = result_df.sort_values(['symbol', 'date'])

            # Ensure numeric columns are of correct type
            numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'adjusted_close', 'volume']
            for col in [c for c in numeric_columns if c in result_df.columns]:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

            logger.info(f"Successfully transformed {len(result_df)} rows of stock price data")
            return result_df
        except Exception as e:
            logger.error(f"Error transforming stock price data: {e}")
            raise


class CompanyDataTransformer(DataTransformer):
    """
    Transformer for company data
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform company data

        Args:
            df (pd.DataFrame): Raw company DataFrame

        Returns:
            pd.DataFrame: Transformed company DataFrame
        """
        try:
            logger.info("Transforming company data")

            # Make a copy to avoid modifying the original
            result_df = df.copy()

            # Ensure column names are standardized
            column_mapping = {
                'Symbol': 'symbol',
                'Name': 'company_name',
                'Company': 'company_name',
                'Industry': 'industry',
                'Sector': 'sector',
                'Market Cap': 'market_cap',
                'MarketCap': 'market_cap'
            }

            # Rename columns that exist in the DataFrame
            existing_columns = {k: v for k, v in column_mapping.items() if k in result_df.columns}
            if existing_columns:
                result_df = result_df.rename(columns=existing_columns)

            # Ensure symbol is uppercase
            if 'symbol' in result_df.columns:
                result_df['symbol'] = result_df['symbol'].str.upper()

            # Ensure market_cap is numeric
            if 'market_cap' in result_df.columns:
                result_df['market_cap'] = pd.to_numeric(result_df['market_cap'], errors='coerce')

            # Drop duplicates based on symbol
            if 'symbol' in result_df.columns:
                result_df = result_df.drop_duplicates(subset=['symbol'])

            logger.info(f"Successfully transformed {len(result_df)} rows of company data")
            return result_df
        except Exception as e:
            logger.error(f"Error transforming company data: {e}")
            raise
