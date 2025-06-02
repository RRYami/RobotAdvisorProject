"""
Database schema definitions for the financial data playground
"""
from src.utils.logger import setup_logger
from src.database.connection import DatabaseConnection

logger = setup_logger(__name__)


class Schema:
    """
    Class to manage database schema creation and initialization
    """

    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize Schema with database connection

        Args:
            db_connection (DatabaseConnection): Database connection object
        """
        self.db = db_connection

    def create_schema(self) -> None:
        """
        Create the database schema if it doesn't exist
        """
        try:
            # Create financial_data schema
            self.db.execute_query(
                "CREATE SCHEMA IF NOT EXISTS financial_data;", params=None
            )
            logger.info("Schema 'financial_data' created successfully")
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            raise

    def create_tables(self) -> None:
        """
        Create all tables for the financial data playground
        """
        try:
            # Create stock_prices table
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS financial_data.stock_prices (
                    id SERIAL PRIMARY KEY,
                    ts_event TIMESTAMP,
                    rtype VARCHAR(50),
                    publisher_id INTEGER,
                    instrument_id INTEGER,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                    volume BIGINT,
                    symbol VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_financial_data_stock_prices_symbol ON financial_data.stock_prices(symbol);
                CREATE INDEX IF NOT EXISTS idx_financial_data_stock_prices_ts_event ON financial_data.stock_prices(ts_event);
                CREATE INDEX IF NOT EXISTS idx_financial_data_stock_prices_instrument_id ON financial_data.stock_prices(instrument_id);
            """, params=None)

            # Create companies table
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS financial_data.companies (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL UNIQUE,
                    company_name VARCHAR(255) NOT NULL,
                    industry VARCHAR(100),
                    sector VARCHAR(100),
                    market_cap DECIMAL(20, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """, params=None)

            # Create financial_metrics table
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS financial_data.financial_metrics (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    reporting_date DATE NOT NULL,
                    revenue DECIMAL(20, 2),
                    net_income DECIMAL(20, 2),
                    eps DECIMAL(10, 4),
                    pe_ratio DECIMAL(10, 2),
                    dividend_yield DECIMAL(5, 4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, reporting_date),
                    FOREIGN KEY (symbol) REFERENCES financial_data.companies(symbol)
                );
            """, params=None)

            logger.info("All tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
