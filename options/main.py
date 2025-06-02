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
                bid DECIMAL(20, 4),
                ask DECIMAL(20, 4),
                change DECIMAL(20, 4),
                percent_change DECIMAL(20, 4),
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility DECIMAL(20, 4),
                delta DECIMAL(20, 4),
                gamma DECIMAL(20, 4),
                theta DECIMAL(20, 4),                vega DECIMAL(20, 4),
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
            "lastPrice": contract_data.get("last", contract_data.get("mark", 0)),
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
        }

        # Use date as lastTradeDate if present
        if "date" in contract_data:
            normalized["lastTradeDate"] = contract_data["date"]

        return normalized

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
        elif isinstance(options_data, list):
            # Alternative format where options come as a list
            print(f"Received options data as a list with {len(options_data)} entries")

            # Group options by expiration date
            options_by_expdate = {}
            for option in options_data:
                exp_date_str = option.get("expirationDate", "")
                if not exp_date_str:
                    continue

                if exp_date_str not in options_by_expdate:
                    options_by_expdate[exp_date_str] = {"calls": [], "puts": []}

                option_type = option.get("optionType", "").upper()
                if option_type == "CALL":
                    options_by_expdate[exp_date_str]["calls"].append(option)
                elif option_type == "PUT":
                    options_by_expdate[exp_date_str]["puts"].append(option)

            # Process the grouped options
            for expiration_date, chain_data in options_by_expdate.items():
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

                # Process calls and puts
                if chain_data["calls"]:
                    print(f"Found {len(chain_data['calls'])} call options")
                    calls_count += len(chain_data["calls"])
                    for call in chain_data["calls"]:
                        self._save_option_contract(symbol, exp_date, "CALL", call, timestamp)

                if chain_data["puts"]:
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

    def _get_field(self, data, *field_names, default=None):
        """Get the first field that exists in the data"""
        for field in field_names:
            if field in data:
                val = data[field]
                if val is None:
                    return default
                return val
        return default

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
        # Extract contract data with safe defaults, handling potential missing fields
        try:
            # Get contract name
            contract_name = ""
            for field in ["contractName", "contractSymbol", "symbol"]:
                if field in contract_data and contract_data[field]:
                    contract_name = contract_data[field]
                    break

            # If contract name is still empty, generate a synthetic one
            if not contract_name:
                print(f"Warning: Missing contract name in data: {contract_data}")
                contract_name = f"{symbol}_{expiration_date.strftime('%Y%m%d')}_{contract_type}_{contract_data.get('strikePrice', 0)}"

            # Get strike price
            strike_price = 0
            for field in ["strikePrice", "strike", "strike_price"]:
                if field in contract_data and contract_data[field] is not None:
                    try:
                        strike_price = float(contract_data[field])
                        break
                    except (ValueError, TypeError):
                        pass

            # Get last trading day
            last_trading_day = datetime(1970, 1, 1).date()
            for field in ["lastTradeDate", "lastTradeDateTime", "lastTrade", "expirationDate", "expiration_date", "expDate"]:
                if field in contract_data and contract_data[field]:
                    try:
                        date_str = contract_data[field]
                        if isinstance(date_str, str):
                            if "T" in date_str:  # ISO format
                                last_trading_day = datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
                            else:
                                last_trading_day = datetime.strptime(date_str, "%Y-%m-%d").date()
                            break
                    except (ValueError, TypeError):
                        pass

            # Get numeric fields with safe defaults
            last_price = 0
            for field in ["lastPrice", "last", "last_price"]:
                if field in contract_data and contract_data[field] is not None:
                    try:
                        last_price = float(contract_data[field])
                        break
                    except (ValueError, TypeError):
                        pass

            bid = 0
            for field in ["bid", "bidPrice"]:
                if field in contract_data and contract_data[field] is not None:
                    try:
                        bid = float(contract_data[field])
                        break
                    except (ValueError, TypeError):
                        pass

            ask = 0
            for field in ["ask", "askPrice"]:
                if field in contract_data and contract_data[field] is not None:
                    try:
                        ask = float(contract_data[field])
                        break
                    except (ValueError, TypeError):
                        pass

            change = 0
            for field in ["change", "netChange"]:
                if field in contract_data and contract_data[field] is not None:
                    try:
                        change = float(contract_data[field])
                        break
                    except (ValueError, TypeError):
                        pass

            percent_change = 0
            for field in ["percentChange", "percentage_change", "changePct"]:
                if field in contract_data and contract_data[field] is not None:
                    try:
                        percent_change = float(contract_data[field])
                        break
                    except (ValueError, TypeError):
                        pass

            volume = 0
            if "volume" in contract_data and contract_data["volume"] is not None:
                try:
                    volume = int(contract_data["volume"])
                except (ValueError, TypeError):
                    pass

            open_interest = 0
            for field in ["openInterest", "open_interest"]:
                if field in contract_data and contract_data[field] is not None:
                    try:
                        open_interest = int(contract_data[field])
                        break
                    except (ValueError, TypeError):
                        pass

            implied_volatility = 0
            for field in ["impliedVolatility", "iv", "volatility"]:
                if field in contract_data and contract_data[field] is not None:
                    try:
                        implied_volatility = float(contract_data[field])
                        break
                    except (ValueError, TypeError):
                        pass

            # Greeks
            delta = 0
            if "delta" in contract_data and contract_data["delta"] is not None:
                try:
                    delta = float(contract_data["delta"])
                except (ValueError, TypeError):
                    pass

            gamma = 0
            if "gamma" in contract_data and contract_data["gamma"] is not None:
                try:
                    gamma = float(contract_data["gamma"])
                except (ValueError, TypeError):
                    pass

            theta = 0
            if "theta" in contract_data and contract_data["theta"] is not None:
                try:
                    theta = float(contract_data["theta"])
                except (ValueError, TypeError):
                    pass

            vega = 0
            if "vega" in contract_data and contract_data["vega"] is not None:
                try:
                    vega = float(contract_data["vega"])
                except (ValueError, TypeError):
                    pass

            # Insert into option_contracts table
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
            print(f"Error saving option contract: {e}")
            print(f"Contract data: {contract_data}")

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

            # For a more robust implementation, you could also check for holidays
            # by making a test API call or using a holidays package

        # If we've made too many attempts, just return the last date we tried
        return date.strftime("%Y-%m-%d")

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
    symbols = ["AAPL", "MSFT", "NVDA"]

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
            time.sleep(12)  # Alpha Vantage free tier allows 5 requests per minute
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
