import psycopg2
import pandas as pd
from io import StringIO
import os

# Database connection parameters from environment variables
db_params = {
    "dbname": os.getenv("POSTGRES_DB", "stock_db"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "password"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

# CSV file path
csv_file = "XNAS_FILE.csv"

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_params)  # type: ignore
    cursor = conn.cursor()

    # Create table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS stock_data (
        ts_event TIMESTAMP,
        rtype VARCHAR(50),
        publisher_id INTEGER,
        instrument_id INTEGER,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC,
        volume BIGINT,
        symbol VARCHAR(10)
    );
    """
    cursor.execute(create_table_query)
    conn.commit()
    print("Table created successfully.")

    # Read CSV file
    df = pd.read_csv(csv_file)

    # Function to copy DataFrame to PostgreSQL using COPY
    def copy_from_stringio(conn, df, table):
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)

        cursor = conn.cursor()
        try:
            cursor.copy_from(buffer, table, sep=',', null='')
            conn.commit()
            print("Data imported successfully.")
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error: {error}")
            conn.rollback()
        finally:
            cursor.close()

    # Import CSV data to table
    copy_from_stringio(conn, df, 'stock_data')

except (Exception, psycopg2.DatabaseError) as error:
    print(f"Error: {error}")
finally:
    if conn:
        cursor.close()
        conn.close()
        print("Database connection closed.")
