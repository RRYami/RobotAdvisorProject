import os
import glob
from questdb.ingress import Sender, TimestampNanos, Protocol
import pandas as pd
import time
import psycopg2
from psycopg2 import sql
import retrying

# DIR

path = "./data"
# Sample DataFrame


def combine_csv_files(directory):
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {directory}")

    # Read and combine all CSV files
    dfs = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure 'Date' column is in datetime format
    if 'Date' in combined_df.columns:
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%m/%d/%Y')

    # Convert numerical columns to float (assuming they are yields or rates)
    numerical_columns = [col for col in combined_df.columns if col not in ['Date']]
    combined_df[numerical_columns] = combined_df[numerical_columns].astype(float)

    # Re order columns
    order = ['Date', '1 Mo', '1.5 Month', '2 Mo', '3 Mo', '4 Mo', '6 Mo',
             '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
    combined_df = combined_df[order]
    # Rename columns for consistency
    combined_df.columns = combined_df.columns.str.replace('Month', 'Mo')
    combined_df.columns = combined_df.columns.str.replace('1.5 Mo', '1_5 Mo')
    combined_df.columns = combined_df.columns.str.replace(' ', '_')

    print(f"Combined {len(csv_files)} CSV files with {len(combined_df)} rows")
    return combined_df


# Get QuestDB connection details from environment variables
QUESTDB_HOST = os.getenv('QUESTDB_HOST', 'questdb')  # Default to 'questdb' service name
QUESTDB_HTTP_PORT = os.getenv('QUESTDB_HTTP_PORT', '9000')  # HTTP port for ILP
QUESTDB_USER = os.getenv('QUESTDB_USER', 'admin')
QUESTDB_PASSWORD = os.getenv('QUESTDB_PASSWORD', 'questdb')


# Function to insert data using QuestDB HTTP (ILP)
@retrying.retry(stop_max_attempt_number=5, wait_fixed=2000)  # Retry 5 times, wait 2s between attempts
def insert_data(data):
    with Sender(Protocol.Http, QUESTDB_HOST, int(QUESTDB_HTTP_PORT),
                username=QUESTDB_USER, password=QUESTDB_PASSWORD) as sender:
        sender.dataframe(data, table_name="US_yield_curve", at='Date')
    print("Data inserted successfully!")


# Function to query data using PostgreSQL interface
def query_data():
    conn = None
    try:
        # Connect to QuestDB via PostgreSQL interface
        conn = psycopg2.connect(
            host=QUESTDB_HOST,
            port=os.getenv('QUESTDB_PORT', '8812'),  # PostgreSQL port
            user=QUESTDB_USER,
            password=QUESTDB_PASSWORD,
            dbname="qdb"
        )
        query = "SELECT * FROM yield_curve LIMIT 10;"
        df = pd.read_sql(query, conn)
        print("Query results:")
        print(df)
    except Exception as e:
        print(f"Query failed: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    # Wait for QuestDB to be ready
    print("Waiting for QuestDB to start...")
    time.sleep(15)  # Increased wait time to ensure QuestDB is ready
    try:
        df = combine_csv_files(path)
        insert_data(df)
        query_data()
    except Exception as e:
        print(f"Error: {e}")
