import sqlite3
import pandas as pd
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('./logs/logs.log'), logging.StreamHandler()])

con = sqlite3.connect('./db/database.db')
cur = con.cursor()


def insert_price_data(data: pd.DataFrame | None):
    if data is None:
        logging.error('Data is None')
        raise ValueError('Data is None')

    try:
        with con:
            logging.info('Starting data insertion into the database')
            cur.executemany(
                'INSERT INTO marketData(Symbol, Date, Open, High, Low, Close, Adjusted_close, Volume) VALUES(?,?,?,?,?,?,?,?)',
                data.values.tolist())
            con.commit()
            logging.info('Data insertion successful')
    except sqlite3.IntegrityError as e:
        logging.error(f'SQLite IntegrityError: {e}')
    except sqlite3.Error as e:
        logging.error(f'SQLite Error: {e}')
    except Exception as e:
        logging.error(f'Error inserting data into the database: {e}')
        raise e

    time.sleep(1)


def insert_income_data(data: pd.DataFrame | None):
    if data is None:
        logging.error('Data is None')
        raise ValueError('Data is None')

    try:
        with con:
            logging.info('Starting data insertion into the database')
            cur.executemany(
                'INSERT INTO incomeData(Symbol, Date, Currency, Revenue, Gross_profit, Total_revenue, Cost_of_revenue, Cost_of_goods_and_services_sold, Operating_income, Selling_general_and_administrative, Research_and_development, Operating_expenses, Investment_income_net, Net_interest_income, Interest_income, Interest_expense, Non_interest_income, Other_non_operating_income, Depreciation, Depreciation_and_amortization, Income_before_tax, Income_tax_expense, Interest_and_debt_expense, Net_income_from_continuing_operations, Comprehensive_income_net_of_tax, EBIT, EBITDA, Net_income) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                data.values.tolist())
            con.commit()
            logging.info('Data insertion successful')
    except sqlite3.IntegrityError as e:
        logging.error(f'SQLite IntegrityError: {e}')
    except sqlite3.Error as e:
        logging.error(f'SQLite Error: {e}')
    except Exception as e:
        logging.error(f'Error inserting data into the database: {e}')
        raise e

    time.sleep(1)
