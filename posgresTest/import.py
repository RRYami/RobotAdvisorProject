import psycopg2
import pandas as pd
from io import StringIO
import os
from typing import Optional, Dict, Any, List
from contextlib import contextmanager


class StockDataImporter:
    """A class for importing and managing stock data in PostgreSQL."""

    def __init__(self, db_params: Optional[Dict[str, str]] = None):
        """
        Initialize the StockDataImporter.

        Args:
            db_params: Database connection parameters. If None, uses environment variables.
        """
        self.db_params = db_params or {
            "dbname": os.getenv("POSTGRES_DB", "stock_db"),
            "user": os.getenv("POSTGRES_USER", "admin"),
            "password": os.getenv("POSTGRES_PASSWORD", "password"),
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": os.getenv("POSTGRES_PORT", "5432")
        }
        self.connection: Optional[psycopg2.extensions.connection] = None

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_params)  # type: ignore
            yield conn
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Database connection error: {error}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def create_stock_table(self, table_name: str = "stock_data") -> bool:
        """
        Create the stock data table if it doesn't exist.

        Args:
            table_name: Name of the table to create

        Returns:
            bool: True if successful, False otherwise
        """
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            ts_event TIMESTAMP,
            rtype VARCHAR(50),
            publisher_id INTEGER,
            instrument_id INTEGER,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC,
            volume BIGINT,
            symbol VARCHAR(10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(symbol);
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts_event ON {table_name}(ts_event);
        CREATE INDEX IF NOT EXISTS idx_{table_name}_instrument_id ON {table_name}(instrument_id);
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_query)
                    conn.commit()
                    print(f"Table '{table_name}' created successfully with indexes.")
                    return True
        except Exception as error:
            print(f"Error creating table: {error}")
            return False

    def import_csv_data(self, csv_file: str, table_name: str = "stock_data") -> bool:
        """
        Import data from CSV file to PostgreSQL table using COPY.

        Args:
            csv_file: Path to the CSV file
            table_name: Name of the target table

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"CSV file '{csv_file}' not found.")
            return False

        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            print(f"Read {len(df)} rows from '{csv_file}'")

            # Import data
            return self._copy_dataframe_to_table(df, table_name)

        except Exception as error:
            print(f"Error importing CSV data: {error}")
            return False

    def _copy_dataframe_to_table(self, df: pd.DataFrame, table_name: str) -> bool:
        """
        Copy DataFrame to PostgreSQL table using COPY command.

        Args:
            df: DataFrame to import
            table_name: Target table name

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                buffer = StringIO()
                df.to_csv(buffer, index=False, header=False)
                buffer.seek(0)

                with conn.cursor() as cursor:
                    cursor.copy_from(buffer, table_name, sep=',', null='',
                                     columns=('ts_event', 'rtype', 'publisher_id', 'instrument_id',
                                              'open', 'high', 'low', 'close', 'volume', 'symbol'))
                    conn.commit()
                    print(f"Data imported successfully to '{table_name}'.")
                    return True

        except Exception as error:
            print(f"Error copying data to table: {error}")
            return False

    def get_table_stats(self, table_name: str = "stock_data") -> Dict[str, Any]:
        """
        Get statistics about the table.

        Args:
            table_name: Name of the table

        Returns:
            dict: Table statistics
        """
        stats = {}

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    stats['total_rows'] = cursor.fetchone()[0]

                    # Get unique symbols
                    cursor.execute(f"SELECT COUNT(DISTINCT symbol) FROM {table_name}")
                    stats['unique_symbols'] = cursor.fetchone()[0]

                    # Get date range
                    cursor.execute(f"SELECT MIN(ts_event), MAX(ts_event) FROM {table_name}")
                    min_date, max_date = cursor.fetchone()
                    stats['date_range'] = {'min': min_date, 'max': max_date}

                    # Get sample symbols
                    cursor.execute(f"SELECT DISTINCT symbol FROM {table_name} ORDER BY symbol LIMIT 10")
                    stats['sample_symbols'] = [row[0] for row in cursor.fetchall()]

        except Exception as error:
            print(f"Error getting table stats: {error}")
            stats['error'] = str(error)

        return stats

    def query_by_symbol(self, symbol: str, table_name: str = "stock_data",
                        limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Query data for a specific symbol.

        Args:
            symbol: Stock symbol to query
            table_name: Name of the table
            limit: Maximum number of rows to return

        Returns:
            DataFrame or None if error
        """
        try:
            with self.get_connection() as conn:
                query = f"""
                SELECT ts_event, open, high, low, close, volume, symbol
                FROM {table_name}
                WHERE symbol = %s
                ORDER BY ts_event
                """
                if limit:
                    query += f" LIMIT {limit}"

                df = pd.read_sql(query, conn, params=(symbol,))
                return df

        except Exception as error:
            print(f"Error querying symbol '{symbol}': {error}")
            return None

    def delete_table_data(self, table_name: str = "stock_data") -> bool:
        """
        Delete all data from the table.

        Args:
            table_name: Name of the table

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"DELETE FROM {table_name}")
                    conn.commit()
                    print(f"All data deleted from '{table_name}'.")
                    return True

        except Exception as error:
            print(f"Error deleting table data: {error}")
            return False


def main():
    """Main function to demonstrate the class usage."""
    # Create importer instance
    importer = StockDataImporter()

    # CSV file path
    csv_file = "xnas-itch-20180501-20250516.ohlcv-1d.csv"

    # Create table
    if importer.create_stock_table():
        # Import data
        if importer.import_csv_data(csv_file):
            # Get and display stats
            stats = importer.get_table_stats()
            print("\nTable Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("Failed to import data.")
    else:
        print("Failed to create table.")


if __name__ == "__main__":
    main()
