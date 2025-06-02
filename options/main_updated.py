import requests
import pandas as pd
import duckdb
import os
from datetime import datetime, timedelta
import time

# get the api key from txt file


def get_api_key():
    with open(r"C:\Users\renar\OneDrive\Bureau\AlphaVantage.txt", 'r') as file:
        api_key = file.read().strip()
    return api_key


api_key = get_api_key()


class AlphaVantageOptionDataLoader:
    """
    Class to download option data from Alpha Vantage and store it in DuckDB
    """

    def __init__(self, api_key, db_path="my_database.duckdb"):
        """
        Initialize with your Alpha Vantage API key and DuckDB path

        Args:
            api_key (str): Your Alpha Vantage API key
            db_path (str): Path to the DuckDB database file
        """
        self.api_key = api_key
        self.db_path = db_path
        self.base_url = "https://www.alphavantage.co/query"
        self.conn = duckdb.connect(db_path)
        self._initialize_database()

    def _initialize_database(self):
        """Set up the database tables if they don't exist"""

        # Create a table for option chains
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS option_chains (
                symbol VARCHAR,
                expiration_date DATE,
                last_updated TIMESTAMP,
                PRIMARY KEY (symbol, expiration_date)
            )
        """)

        # Create a table for option contracts
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS option_contracts (
                symbol VARCHAR,
                expiration_date DATE,
                contract_type VARCHAR,
                strike_price DECIMAL(20, 4),
                contract_name VARCHAR,
                last_trading_day DATE,
                last_price DECIMAL(20, 4),
                mark_price DECIMAL(20, 4),
                bid DECIMAL(20, 4),
                ask DECIMAL(20, 4),
                change DECIMAL(20, 4),
                percent_change DECIMAL(20, 4),
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility DECIMAL(20, 4),
                delta DECIMAL(20, 4),
                gamma DECIMAL(20, 4),
                theta DECIMAL(20, 4),
                vega DECIMAL(20, 4),
                rho DECIMAL(20, 4),
                inserted_at TIMESTAMP,
                PRIMARY KEY (contract_name, expiration_date),
                FOREIGN KEY (symbol, expiration_date) REFERENCES option_chains(symbol, expiration_date)
            )
        """)

    def _normalize_contract(self, contract_data):
        """
        Normalize the contract data to a consistent format

        Args:
            contract_data (dict): Raw contract data from API

        Returns:
            dict: Normalized contract data
        """
        # Map API fields to our expected field names
        normalized = {
            # Map common fields
            "contractName": contract_data.get("contractID", contract_data.get("symbol", "")),
            "strikePrice": contract_data.get("strike", 0),
            "lastPrice": contract_data.get("last", 0),
            "mark": contract_data.get("mark", 0),
            "bid": contract_data.get("bid", 0),
            "ask": contract_data.get("ask", 0),
            "volume": contract_data.get("volume", 0),
            "openInterest": contract_data.get("open_interest", 0),
            "impliedVolatility": contract_data.get("implied_volatility", 0),

            # Greeks
            "delta": contract_data.get("delta", 0),
            "gamma": contract_data.get("gamma", 0),
            "theta": contract_data.get("theta", 0),
            "vega": contract_data.get("vega", 0),
            "rho": contract_data.get("rho", 0)
        }

        # Use date as lastTradeDate if present
        if "date" in contract_data:
            normalized["lastTradeDate"] = contract_data["date"]

        return normalized

    def find_valid_trading_day(self, date_str=None, max_attempts=10):
        """
        Find a valid trading day on or before the given date.
        Markets are closed on weekends and holidays, so we need to check
        previous days until we find a valid trading day.

        Args:
            date_str (str, optional): Starting date in YYYY-MM-DD format.
                                    If None, uses yesterday's date.
            max_attempts (int): Maximum number of days to check backwards

        Returns:
            str: Valid trading date in YYYY-MM-DD format
        """
        if date_str is None:
            date = datetime.now() - timedelta(days=1)
        else:
            date = datetime.strptime(date_str, "%Y-%m-%d")

        attempts = 0

        while attempts < max_attempts:
            # Format the current date
            current_date_str = date.strftime("%Y-%m-%d")

            # Check if it's a weekend (Saturday=5, Sunday=6)
            if date.weekday() >= 5:
                print(f"{current_date_str} is a weekend, checking previous day")
                date -= timedelta(days=1)
                attempts += 1
                continue

            # If we reached here, it's a weekday
            print(f"Using trading date: {current_date_str}")
            return current_date_str

        # If we've made too many attempts, just return the last date we tried
        return date.strftime("%Y-%m-%d")

    def get_option_chain(self, symbol, date=None):
        """
        Get option chain data for a given symbol

        Args:
            symbol (str): Stock symbol to retrieve option data for
            date (str, optional): Date in YYYY-MM-DD format to retrieve historical data.
                                If None, finds the most recent valid trading day.

        Returns:
            dict: Raw API response
        """
        # Find a valid trading date
        trading_date = self.find_valid_trading_day(date)

        params = {
            "function": "HISTORICAL_OPTIONS",
            "symbol": symbol,
            "date": trading_date,
            "apikey": self.api_key
        }

        print(f"Requesting options data for {symbol} on date {trading_date}")
        response = requests.get(self.base_url, params=params)

        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

        data = response.json()

        # Debug the API response
        print(f"API response keys: {data.keys()}")

        # The Alpha Vantage API returns a list of option contracts in the 'data' field
        # We need to transform this into our expected format: organize by expiration date, then call/put
        if "data" in data and isinstance(data["data"], list):
            print(f"Found {len(data['data'])} option contracts in API response")

            # Organize options by expiration date and type
            organized_data = {}
            for contract in data["data"]:
                # Extract expiration date
                expiration = contract.get("expiration", "")
                if not expiration:
                    continue

                # Create entry for this expiration if it doesn't exist
                if expiration not in organized_data:
                    organized_data[expiration] = {"calls": [], "puts": []}

                # Determine if it's a call or put
                option_type = contract.get("type", "").upper()
                if option_type == "CALL" or option_type == "C":
                    organized_data[expiration]["calls"].append(self._normalize_contract(contract))
                elif option_type == "PUT" or option_type == "P":
                    organized_data[expiration]["puts"].append(self._normalize_contract(contract))

            # Return in our expected format
            result = {"options": organized_data}
            print(f"Organized data into {len(organized_data)} expiration dates")
            return result
        else:
            # Handle case where we don't have the expected data format
            print("No option data found in response")
            return {"error": "No option data found in response"}

    def save_option_data(self, symbol):
        """
        Save option data for a given symbol to the DuckDB database

        Args:
            symbol (str): Stock symbol to retrieve and save option data for
        """
        print(f"Downloading option data for {symbol}...")
        data = self.get_option_chain(symbol)

        # Check if we got valid data
        if "error" in data:
            print(f"Error retrieving data: {data['error']}")
            return

        # Check for empty data
        if "options" not in data or not data["options"]:
            print(
                f"No options data returned for {symbol}. Check if the date is valid (weekends/holidays may have no data).")
            return

        # Extract and save data
        timestamp = datetime.now()

        options_count = 0
        calls_count = 0
        puts_count = 0

        # Process each expiration date
        options_data = data.get("options", {})

        # Check if options_data is a dictionary as expected
        if isinstance(options_data, dict):
            # Process each expiration date
            for expiration_date, chain_data in options_data.items():
                # Convert expiration date string to a date object
                try:
                    exp_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
                except ValueError:
                    print(f"Invalid expiration date format: {expiration_date}. Skipping.")
                    continue

                print(f"Processing expiration date: {expiration_date}")

                # Insert into option_chains table
                self.conn.execute("""
                    INSERT OR REPLACE INTO option_chains (symbol, expiration_date, last_updated)
                    VALUES (?, ?, ?)
                """, (symbol, exp_date, timestamp))

                # Process call options
                if "calls" in chain_data:
                    print(f"Found {len(chain_data['calls'])} call options")
                    calls_count += len(chain_data["calls"])
                    for call in chain_data["calls"]:
                        self._save_option_contract(symbol, exp_date, "CALL", call, timestamp)

                # Process put options
                if "puts" in chain_data:
                    print(f"Found {len(chain_data['puts'])} put options")
                    puts_count += len(chain_data["puts"])
                    for put in chain_data["puts"]:
                        self._save_option_contract(symbol, exp_date, "PUT", put, timestamp)

                options_count += 1
        else:
            print(f"Unexpected options data format: {type(options_data)}")
            return

        if options_count > 0:
            print(
                f"Successfully saved option data for {symbol}: {options_count} expiration dates, {calls_count} calls, {puts_count} puts")
        else:
            print(f"No options data was saved for {symbol}")

    def _save_option_contract(self, symbol, expiration_date, contract_type, contract_data, timestamp):
        """
        Save a single option contract to the database

        Args:
            symbol (str): Stock symbol
            expiration_date (date): Option expiration date
            contract_type (str): "CALL" or "PUT"
            contract_data (dict): Contract data from the API
            timestamp (datetime): Timestamp for this data
        """
        # Helper function to get field with default value
        def get_field(data, field, default=0):
            val = data.get(field, default)
            if val is None:
                return default
            return val

        # Extract contract data with safe defaults
        contract_name = get_field(contract_data, "contractName", "")
        strike_price = float(get_field(contract_data, "strikePrice", 0))

        # Parse last trading day - try multiple formats
        last_trading_day_str = get_field(contract_data, "lastTradeDate", "1970-01-01")
        try:
            if isinstance(last_trading_day_str, str):
                last_trading_day = datetime.strptime(last_trading_day_str, "%Y-%m-%d").date()
            else:
                last_trading_day = datetime(1970, 1, 1).date()
        except ValueError:
            print(f"Warning: Could not parse date '{last_trading_day_str}', using epoch")
            last_trading_day = datetime(1970, 1, 1).date()

        # Extract other numeric fields
        last_price = float(get_field(contract_data, "lastPrice", 0))
        mark_price = float(get_field(contract_data, "mark", 0))
        bid = float(get_field(contract_data, "bid", 0))
        ask = float(get_field(contract_data, "ask", 0))
        change = float(get_field(contract_data, "change", 0))
        percent_change = float(get_field(contract_data, "percentChange", 0))
        volume = int(get_field(contract_data, "volume", 0))
        open_interest = int(get_field(contract_data, "openInterest", 0))
        implied_volatility = float(get_field(contract_data, "impliedVolatility", 0))

        # Greeks (these might not always be available)
        delta = float(get_field(contract_data, "delta", 0))
        gamma = float(get_field(contract_data, "gamma", 0))
        theta = float(get_field(contract_data, "theta", 0))
        vega = float(get_field(contract_data, "vega", 0))
        rho = float(get_field(contract_data, "rho", 0))

        # Generate a unique name if missing
        if not contract_name:
            contract_name = f"{symbol}_{expiration_date.strftime('%Y%m%d')}_{contract_type}_{strike_price}"

        # Insert into option_contracts table
        try:
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
        except Exception as e:
            print(f"Error saving contract {contract_name}: {e}")

    def get_stored_option_data(self, symbol=None, expiration_date=None, contract_type=None):
        """
        Query the database for stored option data

        Args:
            symbol (str, optional): Filter by stock symbol
            expiration_date (str, optional): Filter by expiration date (YYYY-MM-DD)
            contract_type (str, optional): Filter by contract type ("CALL" or "PUT")

        Returns:
            pandas.DataFrame: Query results
        """
        query = "SELECT * FROM option_contracts WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if expiration_date:
            query += " AND expiration_date = ?"
            if isinstance(expiration_date, str):
                expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
            params.append(expiration_date)

        if contract_type:
            query += " AND contract_type = ?"
            params.append(contract_type.upper())

        return self.conn.execute(query, params).fetchdf()

    def close(self):
        """Close the database connection"""
        self.conn.close()


if __name__ == "__main__":
    # Get API key from environment variable for security
    api_key = get_api_key()

    if not api_key:
        print("Please set the ALPHA_VANTAGE_API_KEY environment variable")
        exit(1)

    # Initialize loader
    loader = AlphaVantageOptionDataLoader(api_key)

    print("=" * 50)
    print("Alpha Vantage Options Data Downloader")
    print("=" * 50)

    # List of symbols to download data for - try multiple popular stocks
    symbols = ["AAPL", "MSFT", "NVDA", "SPY"]

    # Try to find yesterday's trading day
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    valid_day = loader.find_valid_trading_day(yesterday)
    print(f"Using trading day: {valid_day}")

    # Download data for each symbol
    for symbol in symbols:
        try:
            print(f"\n{'=' * 30}")
            print(f"Processing {symbol}")
            print(f"{'=' * 30}")
            loader.save_option_data(symbol)
            # Sleep to avoid hitting API rate limits
            print("Waiting to avoid API rate limits...")
            time.sleep(15)  # Alpha Vantage free tier allows 5 requests per minute
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Query examples for each symbol
    print("\n" + "=" * 50)
    print("Data Summary:")
    print("=" * 50)

    for symbol in symbols:
        calls = loader.get_stored_option_data(symbol=symbol, contract_type="CALL")
        puts = loader.get_stored_option_data(symbol=symbol, contract_type="PUT")

        print(f"\n{symbol} data summary:")
        print(f"- Found {len(calls)} call options")
        print(f"- Found {len(puts)} put options")

        if not calls.empty:
            print(f"\nSample {symbol} call options:")
            print(calls.head(3))

    # Close connection
    loader.close()
    print("\nDatabase connection closed.")
