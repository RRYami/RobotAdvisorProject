import requests
import pandas as pd
import duckdb
import json
from datetime import datetime, timedelta
import time

# Get API key


def get_api_key():
    with open(r"C:\Users\renar\OneDrive\Bureau\AlphaVantage.txt", 'r') as file:
        api_key = file.read().strip()
    return api_key


def find_valid_trading_day(date_str=None, max_attempts=10):
    """Find a valid trading day on or before the given date."""
    if date_str is None:
        date = datetime.now() - timedelta(days=1)
    else:
        date = datetime.strptime(date_str, "%Y-%m-%d")

    attempts = 0
    while attempts < max_attempts:
        current_date_str = date.strftime("%Y-%m-%d")
        if date.weekday() >= 5:  # Weekend
            date -= timedelta(days=1)
        else:
            return current_date_str
        attempts += 1

    return date.strftime("%Y-%m-%d")


class HistoricalOptionsLoader:
    """Load historical options data from Alpha Vantage and save to DuckDB"""

    def __init__(self, api_key, db_path="my_database.duckdb"):
        self.api_key = api_key
        self.db_path = db_path
        self.base_url = "https://www.alphavantage.co/query"
        self.conn = duckdb.connect(db_path)

    def get_historical_options(self, symbol, date=None):
        """Fetch historical options data for a given symbol and date"""
        if date is None:
            date = find_valid_trading_day()

        print(f"Fetching historical options for {symbol} on {date}")

        params = {
            "function": "HISTORICAL_OPTIONS",
            "symbol": symbol,
            "date": date,
            "apikey": self.api_key
        }

        response = requests.get(self.base_url, params=params)

        if response.status_code != 200:
            print(f"Error: API request failed with status {response.status_code}")
            print(response.text)
            return None

        data = response.json()

        # Save raw response for debugging
        with open(f"{symbol}_options_raw.json", "w") as f:
            json.dump(data, f, indent=2)

        return data

    def process_and_save_options(self, symbol, date=None):
        """Process and save options data to the database"""
        data = self.get_historical_options(symbol, date)

        if not data or "data" not in data or not isinstance(data["data"], list):
            print("No valid options data found")
            return False

        options_data = data["data"]
        print(f"Found {len(options_data)} option contracts")

        # Process each option contract
        timestamp = datetime.now()
        contracts_saved = 0

        # Keep track of unique expiration dates to create option_chains entries
        expirations = set()

        for contract in options_data:
            try:
                # Get expiration date
                exp_date_str = contract.get("expiration", "")
                if not exp_date_str:
                    continue

                exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
                expirations.add(exp_date)

                # Determine contract type
                option_type = contract.get("type", "").upper()
                if option_type not in ["CALL", "PUT"]:
                    continue

                contract_symbol = contract.get("symbol", "")
                if not contract_symbol:
                    continue

                # Extract and convert data
                strike_price = float(contract.get("strike", 0))

                # Find the last trading date
                last_trading_day = datetime.now().date()
                if "date" in contract:
                    try:
                        last_trading_day = datetime.strptime(contract["date"], "%Y-%m-%d").date()
                    except ValueError:
                        pass

                # Extract pricing information
                last_price = float(contract.get("last", 0))
                mark_price = last_price  # Alpha Vantage doesn't provide mark price directly
                bid = float(contract.get("bid", 0))
                ask = float(contract.get("ask", 0))

                # Some APIs use "change" field directly
                change = 0
                if "change" in contract:
                    change = float(contract["change"])

                percent_change = 0
                if "change_percentage" in contract:
                    # Remove % sign if present
                    pct_str = contract["change_percentage"]
                    if isinstance(pct_str, str) and "%" in pct_str:
                        pct_str = pct_str.replace("%", "")
                    try:
                        percent_change = float(pct_str)
                    except (ValueError, TypeError):
                        pass

                # Volume and open interest
                volume = int(contract.get("volume", 0) or 0)
                open_interest = int(contract.get("open_interest", 0) or 0)

                # Implied volatility - sometimes provided as decimal (0.25), sometimes as percentage (25%)
                implied_volatility = 0
                if "implied_volatility" in contract:
                    iv_value = contract["implied_volatility"]
                    if isinstance(iv_value, str) and "%" in iv_value:
                        iv_value = iv_value.replace("%", "")
                        try:
                            implied_volatility = float(iv_value) / 100.0  # Convert percentage to decimal
                        except (ValueError, TypeError):
                            pass
                    else:
                        try:
                            implied_volatility = float(iv_value or 0)
                        except (ValueError, TypeError):
                            pass

                # Greeks - these are often not provided in all APIs
                delta = float(contract.get("delta", 0) or 0)
                gamma = float(contract.get("gamma", 0) or 0)
                theta = float(contract.get("theta", 0) or 0)
                vega = float(contract.get("vega", 0) or 0)
                rho = float(contract.get("rho", 0) or 0)

                # Save to database
                self.save_contract(
                    symbol=symbol,
                    expiration_date=exp_date,
                    contract_type=option_type,
                    contract_name=contract_symbol,
                    strike_price=strike_price,
                    last_trading_day=last_trading_day,
                    last_price=last_price,
                    mark_price=mark_price,
                    bid=bid,
                    ask=ask,
                    change=change,
                    percent_change=percent_change,
                    volume=volume,
                    open_interest=open_interest,
                    implied_volatility=implied_volatility,
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    vega=vega,
                    rho=rho,
                    timestamp=timestamp
                )

                contracts_saved += 1

            except Exception as e:
                print(f"Error processing contract: {e}")
                print(f"Contract data: {contract}")
                continue

        # Save expiration dates to option_chains table
        for exp_date in expirations:
            self.conn.execute("""
                INSERT OR REPLACE INTO option_chains (symbol, expiration_date, last_updated)
                VALUES (?, ?, ?)
            """, (symbol, exp_date, timestamp))

        print(f"Processed {len(options_data)} contracts, saved {contracts_saved} to database")
        print(f"Found {len(expirations)} unique expiration dates")

        return contracts_saved > 0

    def save_contract(self, symbol, expiration_date, contract_type, contract_name,
                      strike_price, last_trading_day, last_price, mark_price,
                      bid, ask, change, percent_change, volume, open_interest,
                      implied_volatility, delta, gamma, theta, vega, rho, timestamp):
        """Save a single option contract to the database"""
        try:
            # Check if the table has a rho column
            has_rho = False
            columns = self.conn.execute("PRAGMA table_info(option_contracts)").fetchall()
            column_names = [col[1] for col in columns]

            if "rho" in column_names:
                has_rho = True

            # Construct the SQL query based on available columns
            if "mark_price" in column_names:
                if has_rho:
                    # Full schema with all columns
                    self.conn.execute("""
                        INSERT OR REPLACE INTO option_contracts (
                            symbol, expiration_date, contract_type, strike_price, contract_name,
                            last_trading_day, last_price, mark_price, bid, ask, change, percent_change,
                            volume, open_interest, implied_volatility, delta, gamma, theta, vega, rho,
                            inserted_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, expiration_date, contract_type, strike_price, contract_name,
                        last_trading_day, last_price, mark_price, bid, ask, change, percent_change,
                        volume, open_interest, implied_volatility, delta, gamma, theta, vega, rho,
                        timestamp
                    ))
                else:
                    # Schema with mark_price but no rho
                    self.conn.execute("""
                        INSERT OR REPLACE INTO option_contracts (
                            symbol, expiration_date, contract_type, strike_price, contract_name,
                            last_trading_day, last_price, mark_price, bid, ask, change, percent_change,
                            volume, open_interest, implied_volatility, delta, gamma, theta, vega,
                            inserted_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, expiration_date, contract_type, strike_price, contract_name,
                        last_trading_day, last_price, mark_price, bid, ask, change, percent_change,
                        volume, open_interest, implied_volatility, delta, gamma, theta, vega,
                        timestamp
                    ))
            else:
                # Original schema without mark_price
                self.conn.execute("""
                    INSERT OR REPLACE INTO option_contracts (
                        symbol, expiration_date, contract_type, strike_price, contract_name,
                        last_trading_day, last_price, bid, ask, change, percent_change,
                        volume, open_interest, implied_volatility, delta, gamma, theta, vega,
                        inserted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, expiration_date, contract_type, strike_price, contract_name,
                    last_trading_day, last_price, bid, ask, change, percent_change,
                    volume, open_interest, implied_volatility, delta, gamma, theta, vega,
                    timestamp
                ))

        except Exception as e:
            print(f"Error saving contract {contract_name}: {e}")

    def close(self):
        """Close the database connection"""
        self.conn.close()


if __name__ == "__main__":
    # Get API key
    api_key = get_api_key()

    print("=" * 60)
    print("Alpha Vantage HISTORICAL_OPTIONS Data Loader")
    print("=" * 60)

    # Initialize the loader
    loader = HistoricalOptionsLoader(api_key)

    # Define symbols to process
    symbols = ["AAPL", "MSFT", "NVDA"]

    # Process each symbol
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        loader.process_and_save_options(symbol)

        # Respect API rate limits
        if symbol != symbols[-1]:
            print("Waiting 15 seconds for rate limits...")
            time.sleep(15)

    print("\nAll processing complete. Closing database connection.")
    loader.close()
