"""
Data loaders for loading processed data into the database
"""
import pandas as pd
from typing import Optional, List, Dict, Any, Literal
from sqlalchemy import create_engine
from src.database.connection import DatabaseConnection
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """
    Base class for data loaders
    """

    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize DataLoader with database connection

        Args:
            db_connection (DatabaseConnection): Database connection object
        """
        self.db = db_connection

    def load(self, df: pd.DataFrame, table_name: str, schema: str = "financial_data") -> int:
        """
        Load data into the database

        Args:
            df (pd.DataFrame): DataFrame to load
            table_name (str): Target table name
            schema (str, optional): Schema name. Defaults to "financial_data".

        Returns:
            int: Number of rows loaded
        """
        raise NotImplementedError("Subclasses must implement load method")


class PostgresLoader(DataLoader):
    """
    Loader for PostgreSQL database
    """

    def load(self, df: pd.DataFrame, table_name: str, schema: str = "financial_data",
             if_exists: Literal["append", "replace", "fail"] = "append", chunk_size: int = 1000) -> int:
        """
        Load data into PostgreSQL table

        Args:
            df (pd.DataFrame): DataFrame to load
            table_name (str): Target table name
            schema (str, optional): Schema name. Defaults to "financial_data".
            if_exists (Literal["append", "replace", "fail"], optional): Action if table exists. Defaults to "append".
            chunk_size (int, optional): Size of chunks when loading. Defaults to 1000.

        Returns:
            int: Number of rows loaded
        """
        try:
            logger.info(f"Loading {len(df)} rows into {schema}.{table_name}")

            # For the first implementation, we'll use pandas to_sql for simplicity
            # In a more advanced version, this could be replaced with more efficient bulk loading

            # Create SQLAlchemy engine from connection parameters
            conn_string = f"postgresql://{self.db.config['user']}:{self.db.config['password']}@"\
                f"{self.db.config['host']}:{self.db.config['port']}/{self.db.config['database']}"
            engine = create_engine(conn_string)

            # Load data to table
            df.to_sql(
                name=table_name,
                schema=schema,
                con=engine,
                if_exists=if_exists,
                index=False,
                chunksize=chunk_size
            )

            logger.info(f"Successfully loaded {len(df)} rows into {schema}.{table_name}")
            return len(df)
        except Exception as e:
            logger.error(f"Error loading data into {schema}.{table_name}: {e}")
            raise


class StockPriceLoader(PostgresLoader):
    """
    Specialized loader for stock price data
    """

    def load(self, df: pd.DataFrame, table_name: str = "stock_prices",
             schema: str = "financial_data", if_exists: Literal["append", "replace", "fail"] = "append",
             chunk_size: int = 1000) -> int:
        """
        Load stock price data into the database

        Args:
            df (pd.DataFrame): DataFrame with stock price data
            table_name (str, optional): Target table name. Defaults to "stock_prices".
            schema (str, optional): Schema name. Defaults to "financial_data".
            if_exists (Literal["append", "replace", "fail"], optional): Action if table exists. Defaults to "append".
            chunk_size (int, optional): Size of chunks when loading. Defaults to 1000.

        Returns:
            int: Number of rows loaded
        """
        # Ensure required columns are present
        required_columns = ['symbol', 'date', 'open_price', 'high_price', 'low_price',
                            'close_price', 'adjusted_close', 'volume']

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return super().load(df, table_name, schema, if_exists, chunk_size)


class CompanyLoader(PostgresLoader):
    """
    Specialized loader for company data
    """

    def load(self, df: pd.DataFrame, table_name: str = "companies",
             schema: str = "financial_data", if_exists: Literal["append", "replace", "fail"] = "append",
             chunk_size: int = 1000) -> int:
        """
        Load company data into the database

        Args:
            df (pd.DataFrame): DataFrame with company data
            table_name (str, optional): Target table name. Defaults to "companies".
            schema (str, optional): Schema name. Defaults to "financial_data".
            if_exists (Literal["append", "replace", "fail"], optional): Action if table exists. Defaults to "append".
            chunk_size (int, optional): Size of chunks when loading. Defaults to 1000.

        Returns:
            int: Number of rows loaded
        """
        # Ensure required columns are present
        required_columns = ['symbol', 'company_name']

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return super().load(df, table_name, schema, if_exists, chunk_size)
