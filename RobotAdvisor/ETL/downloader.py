import logging
import pandas as pd
import yfinance as yf
import collections.abc as c
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("./logs/logs.log"), logging.StreamHandler()],
)

# Alphavantage API key
api_key = open("./keys/AlphaVantage.txt").read().strip()


def get_price_data(ticker: str) -> pd.DataFrame | None:
    logging.info(f"Starting data download for ticker: {ticker}")
    try:
        data = yf.download(tickers=ticker)
        logging.info(f"Data download successful for ticker: {ticker}")
    except Exception as e:
        logging.error(f"Error downloading data for ticker {ticker}: {e}")
        return None

    data["Symbol"] = ticker
    data.reset_index(inplace=True)
    data = data[["Symbol", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    data["Date"] = data["Date"].astype(str)

    logging.info(f"Data processing complete for ticker: {ticker}")
    return data


def get_income_data(ticker: str) -> pd.DataFrame | None:

    logging.info(f"Starting income data download for ticker: {ticker}")
    try:
        response = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}"
        )
        data = response.json()
        data = pd.DataFrame(data["quarterlyReports"])
        logging.info(f"Income Data download successful for ticker: {ticker}")
    except Exception as e:
        logging.error(f"Error downloading data for ticker {ticker}: {e}")

    data["Symbol"] = ticker
    data = data[
        [
            "Symbol",
            "fiscalDateEnding",
            "reportedCurrency",
            "totalRevenue",
            "grossProfit",
            "totalRevenue",
            "costOfRevenue",
            "costofGoodsAndServicesSold",
            "operatingIncome",
            "sellingGeneralAndAdministrative",
            "researchAndDevelopment",
            "operatingExpenses",
            "investmentIncomeNet",
            "netInterestIncome",
            "interestIncome",
            "interestExpense",
            "nonInterestIncome",
            "otherNonOperatingIncome",
            "depreciation",
            "depreciationAndAmortization",
            "incomeBeforeTax",
            "incomeTaxExpense",
            "interestAndDebtExpense",
            "netIncomeFromContinuingOperations",
            "comprehensiveIncomeNetOfTax",
            "ebit",
            "ebitda",
            "netIncome",
        ]
    ]
    data.columns = [
        "Symbol",
        "Date",
        "Currency",
        "Revenue",
        "Gross_profit",
        "Total_revenue",
        "Cost_of_revenue",
        "Cost_of_goods_and_services_sold",
        "Operating_income",
        "Selling_general_and_administrative",
        "Research_and_development",
        "Operating_expenses",
        "Investment_income_net",
        "Net_interest_income",
        "Interest_income",
        "Interest_expense",
        "Non_interest_income",
        "Other_non_operating_income",
        "Depreciation",
        "Depreciation_and_amortization",
        "Income_before_tax",
        "Income_tax_expense",
        "Interest_and_debt_expense",
        "Net_income_from_continuing_operations",
        "Comprehensive_income_net_of_tax",
        "EBIT",
        "EBITDA",
        "Net_income",
    ]
    data["Date"] = data["Date"].astype(str)
    logging.info(f"Data processing complete for ticker: {ticker}")
    return data
