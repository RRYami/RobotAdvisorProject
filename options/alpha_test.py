import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json

# Get API key


def get_api_key():
    with open(r"C:\Users\renar\OneDrive\Bureau\AlphaVantage.txt", 'r') as file:
        api_key = file.read().strip()
    return api_key


def find_valid_trading_day(date_str=None, max_attempts=10):
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


def test_historical_options():
    api_key = get_api_key()
    base_url = "https://www.alphavantage.co/query"

    # Use popular stocks with options
    symbols = ["AAPL"]  # Limiting to one symbol for focused testing

    # Get yesterday's date as a trading day
    trading_date = find_valid_trading_day()

    print(f"\n{'=' * 50}")
    print(f"Testing HISTORICAL_OPTIONS endpoint with date: {trading_date}")
    print(f"{'=' * 50}")

    for symbol in symbols:
        print(f"\nTesting options data for {symbol}...")

        params = {
            "function": "HISTORICAL_OPTIONS",
            "symbol": symbol,
            "date": trading_date,
            "apikey": api_key
        }

        try:
            print(f"Request URL: {base_url}")
            print(f"Params: {params}")
            response = requests.get(base_url, params=params)

            if response.status_code == 200:
                data = response.json()

                # Print the structure of the response
                print(f"\nResponse structure:")
                if "data" in data and isinstance(data["data"], list):
                    print(f"- 'data' field is a list with {len(data['data'])} items")

                    if data["data"]:
                        # Print keys from the first item to understand structure
                        first_item = data["data"][0]
                        print(f"- Sample data item keys: {list(first_item.keys())}")

                        # Print a full sample contract
                        print("\nSample contract:")
                        print(json.dumps(first_item, indent=2))

                        # Count calls vs puts
                        calls = [item for item in data["data"] if item.get("type", "").upper() == "CALL"]
                        puts = [item for item in data["data"] if item.get("type", "").upper() == "PUT"]
                        print(f"\nFound {len(calls)} CALL options and {len(puts)} PUT options")

                        # Group by expiration date
                        expirations = {}
                        for item in data["data"]:
                            exp = item.get("expiration", "unknown")
                            if exp not in expirations:
                                expirations[exp] = {"calls": 0, "puts": 0}

                            if item.get("type", "").upper() == "CALL":
                                expirations[exp]["calls"] += 1
                            elif item.get("type", "").upper() == "PUT":
                                expirations[exp]["puts"] += 1

                        print("\nExpiration dates summary:")
                        for exp, counts in expirations.items():
                            print(f"- {exp}: {counts['calls']} calls, {counts['puts']} puts")
                else:
                    print("- No 'data' field found or it's not a list")
                    print(f"- Response keys: {list(data.keys())}")

                # Save the full response to a file for reference
                filename = f"{symbol}_options_{trading_date}.json"
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"\nFull response saved to {filename}")
            else:
                print(f"Failed with status code {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Exception: {str(e)}")

        print("\nHistorical options testing complete!")


if __name__ == "__main__":
    print("Testing Alpha Vantage HISTORICAL_OPTIONS API endpoint...")
    test_historical_options()
