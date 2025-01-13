from ETL.load import insert_price_data, insert_income_data
from ETL.downloader import get_price_data, get_income_data
from ETL.create import create_price_table, create_income_table


def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'SNOW', 'PLTR', 'ARM', 'AMD', 'QCOM']
    create_price_table()
    create_income_table()
    for ticker in tickers:
        data = get_price_data(ticker)
        income_data = get_income_data(ticker)
        if data is not None:
            insert_price_data(data)
            insert_income_data(income_data)


if __name__ == "__main__":
    main()
