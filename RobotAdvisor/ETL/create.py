import sqlite3
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("./logs/logs.log"), logging.StreamHandler()],
)

con = sqlite3.connect("./db/database.db")
cur = con.cursor()


def create_price_table():
    logging.info("Creating markerData table")
    try:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS marketData(id INTEGER PRIMARY KEY, Symbol TEXT, Date TEXT, Open FLOAT, High FLOAT, Low FLOAT, Close FLOAT, Adjusted_close FLOAT, Volume FLOAT)"
        )
        con.commit()
        logging.info("marketData table created successfully")
    except Exception as e:
        logging.error(f"Error creating marketData table: {e}")


def create_income_table():
    logging.info("Creating incomeData table")
    try:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS incomeData(id INTEGER PRIMARY KEY, Symbol TEXT, Date TEXT, Currency TEXT, Revenue FLOAT, Gross_profit FLOAT, Total_revenue FLOAT, Cost_of_revenue FLOAT, Cost_of_goods_and_services_sold FLOAT, Operating_income FLOAT, Selling_general_and_administrative FLOAT, Research_and_development FLOAT, Operating_expenses FLOAT, Investment_income_net FLOAT, Net_interest_income FLOAT, Interest_income FLOAT, Interest_expense FLOAT, Non_interest_income FLOAT, Other_non_operating_income FLOAT, Depreciation FLOAT, Depreciation_and_amortization FLOAT, Income_before_tax FLOAT, Income_tax_expense FLOAT, Interest_and_debt_expense FLOAT, Net_income_from_continuing_operations FLOAT, Comprehensive_income_net_of_tax FLOAT, EBIT FLOAT, EBITDA FLOAT, Net_income FLOAT)"
        )
        con.commit()
        logging.info("incomeData table created successfully")
    except Exception as e:
        logging.error(f"Error creating incomeData table: {e}")
