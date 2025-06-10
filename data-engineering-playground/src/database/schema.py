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
        Create all tables for the financial data playground based on Alpha Vantage API data
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
                    asset_type VARCHAR(50),
                    cik VARCHAR(20),
                    name VARCHAR(255),
                    exchange VARCHAR(10),
                    country VARCHAR(100),
                    currency VARCHAR(10),
                    sector VARCHAR(100),
                    industry VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """, params=None)

            # # Create stock_prices table
            # self.db.execute_query("""
            #     CREATE TABLE IF NOT EXISTS financial_data.stock_prices (
            #         id SERIAL PRIMARY KEY,
            #         symbol VARCHAR(10) NOT NULL,
            #         date DATE NOT NULL,
            #         open NUMERIC(15, 4),
            #         high NUMERIC(15, 4),
            #         low NUMERIC(15, 4),
            #         close NUMERIC(15, 4),
            #         adjusted_close NUMERIC(15, 4),
            #         volume BIGINT,
            #         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            #         UNIQUE(symbol, date),
            #         FOREIGN KEY (symbol) REFERENCES financial_data.companies(symbol)
            #     );

            #     CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON financial_data.stock_prices(symbol);
            #     CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON financial_data.stock_prices(date);
            # """, params=None)

            # Create balance_sheet table based on Alpha Vantage API
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS financial_data.balance_sheets (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    fiscal_date_ending DATE NOT NULL,
                    reported_currency VARCHAR(10),
                    total_assets NUMERIC(20, 2),
                    total_current_assets NUMERIC(20, 2),
                    cash_and_equivalents NUMERIC(20, 2),
                    cash_and_short_term_investments NUMERIC(20, 2),
                    inventory NUMERIC(20, 2),
                    current_net_receivables NUMERIC(20, 2),
                    total_non_current_assets NUMERIC(20, 2),
                    property_plant_equipment NUMERIC(20, 2),
                    long_term_investments NUMERIC(20, 2),
                    short_term_investments NUMERIC(20, 2),
                    other_current_assets NUMERIC(20, 2),
                    total_liabilities NUMERIC(20, 2),
                    total_current_liabilities NUMERIC(20, 2),
                    current_accounts_payable NUMERIC(20, 2),
                    short_term_debt NUMERIC(20, 2),
                    total_non_current_liabilities NUMERIC(20, 2),
                    long_term_debt NUMERIC(20, 2),
                    current_long_term_debt NUMERIC(20, 2),
                    short_long_term_debt_total NUMERIC(20, 2),
                    other_current_liabilities NUMERIC(20, 2),
                    other_non_current_liabilities NUMERIC(20, 2),
                    total_shareholder_equity NUMERIC(20, 2),
                    retained_earnings NUMERIC(20, 2),
                    common_stock NUMERIC(20, 2),
                    common_stock_shares_outstanding NUMERIC(20, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, fiscal_date_ending),
                    FOREIGN KEY (symbol) REFERENCES financial_data.companies(symbol)
                );
                
                CREATE INDEX IF NOT EXISTS idx_balance_sheets_symbol ON financial_data.balance_sheets(symbol);
                CREATE INDEX IF NOT EXISTS idx_balance_sheets_fiscal_date ON financial_data.balance_sheets(fiscal_date_ending);
            """, params=None)

            # Create income_statements table based on Alpha Vantage API
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS financial_data.income_statements (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    fiscal_date_ending DATE NOT NULL,
                    reported_currency VARCHAR(10),
                    gross_profit NUMERIC(20, 2),
                    total_revenue NUMERIC(20, 2),
                    cost_of_revenue NUMERIC(20, 2),
                    cost_of_goods_and_services_sold NUMERIC(20, 2),
                    operating_income NUMERIC(20, 2),
                    selling_general_and_administrative NUMERIC(20, 2),
                    research_and_development NUMERIC(20, 2),
                    operating_expenses NUMERIC(20, 2),
                    interest_income NUMERIC(20, 2),
                    interest_expense NUMERIC(20, 2),
                    depreciation_and_amortization NUMERIC(20, 2),
                    income_before_tax NUMERIC(20, 2),
                    income_tax_expense NUMERIC(20, 2),
                    net_income_from_continuing_operations NUMERIC(20, 2),
                    ebit NUMERIC(20, 2),
                    ebitda NUMERIC(20, 2),
                    net_income NUMERIC(20, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, fiscal_date_ending),
                    FOREIGN KEY (symbol) REFERENCES financial_data.companies(symbol)
                );
                
                CREATE INDEX IF NOT EXISTS idx_income_statements_symbol ON financial_data.income_statements(symbol);
                CREATE INDEX IF NOT EXISTS idx_income_statements_fiscal_date ON financial_data.income_statements(fiscal_date_ending);
            """, params=None)

            # Create cash_flows table based on Alpha Vantage API
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS financial_data.cash_flows (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    fiscal_date_ending DATE NOT NULL,
                    reported_currency VARCHAR(10),
                    operating_cashflow NUMERIC(20, 2),
                    depreciation_depletion_and_amortization NUMERIC(20, 2),
                    capital_expenditures NUMERIC(20, 2),
                    change_in_inventory NUMERIC(20, 2),
                    cashflow_from_investment NUMERIC(20, 2),
                    cashflow_from_financing NUMERIC(20, 2),
                    dividend_payout NUMERIC(20, 2),
                    dividend_payout_common_stock NUMERIC(20, 2),
                    proceeds_from_repurchase_of_equity NUMERIC(20, 2),
                    net_income NUMERIC(20, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, fiscal_date_ending),
                    FOREIGN KEY (symbol) REFERENCES financial_data.companies(symbol)
                );
                
                CREATE INDEX IF NOT EXISTS idx_cash_flows_symbol ON financial_data.cash_flows(symbol);
                CREATE INDEX IF NOT EXISTS idx_cash_flows_fiscal_date ON financial_data.cash_flows(fiscal_date_ending);
            """, params=None)

            # Create earnings table based on Alpha Vantage API
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS financial_data.earnings (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    fiscal_date_ending DATE NOT NULL,
                    reported_eps NUMERIC(10, 4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, fiscal_date_ending),
                    FOREIGN KEY (symbol) REFERENCES financial_data.companies(symbol)
                );
                
                CREATE INDEX IF NOT EXISTS idx_earnings_symbol ON financial_data.earnings(symbol);
                CREATE INDEX IF NOT EXISTS idx_earnings_fiscal_date ON financial_data.earnings(fiscal_date_ending);
            """, params=None)

            # Create dividends table based on Alpha Vantage API
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS financial_data.dividends (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    ex_dividend_date DATE NOT NULL,
                    declaration_date DATE,
                    record_date DATE,
                    payment_date DATE,
                    amount NUMERIC(10, 4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, ex_dividend_date),
                    FOREIGN KEY (symbol) REFERENCES financial_data.companies(symbol)
                );
                
                CREATE INDEX IF NOT EXISTS idx_dividends_symbol ON financial_data.dividends(symbol);
                CREATE INDEX IF NOT EXISTS idx_dividends_ex_date ON financial_data.dividends(ex_dividend_date);
            """, params=None)

            # Create stock splits table based on Alpha Vantage API
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS financial_data.stock_splits (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    effective_date DATE NOT NULL,
                    split_factor VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, effective_date),
                    FOREIGN KEY (symbol) REFERENCES financial_data.companies(symbol)
                );
                
                CREATE INDEX IF NOT EXISTS idx_stock_splits_symbol ON financial_data.stock_splits(symbol);
                CREATE INDEX IF NOT EXISTS idx_stock_splits_effective_date ON financial_data.stock_splits(effective_date);
            """, params=None)

            logger.info("All financial data tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
