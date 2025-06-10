import os
from dotenv import load_dotenv
# from etl.extractors import AlphaVantageExtractor
from utils.logger import setup_logger
from database.connection import DatabaseConnection
from database.schema import Schema
# Set up logger
logger = setup_logger(__name__)

# Main script to run the ETL process


def main():
    """
    Main function to run the ETL process
    """
    try:
        # Load environment variables
        # load_dotenv()
        # api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        # if not api_key:
        #     logger.error("API key not found in environment variables")
        #     return
        print("Starting ETL process...")
        logger.info("Starting ETL process...")
        # Create the schema if it doesn't exist
        db_connection = DatabaseConnection('./config/database.yaml')
        db_connection.connect()
        schema = Schema(db_connection)
        schema.create_schema()
        logger.info("Database schema created successfully.")
        # Create the tables in the schema
        schema.create_tables()
        logger.info("Database schema and tables created successfully.")
        print("Database schema and tables created successfully.")
        # # Initialize the data extractor
        # extractor = AlphaVantageExtractor()

        # for symbol in ["KO", "PLTR", "OKLO"]:
        #     logger.info(f"Extracting earnings data for symbol: {symbol}")
        #     for function in ["EARNINGS", "CASH_FLOW", "BALANCE_SHEET", "INCOME_STATEMENT", "SPLITS", "DIVIDENDS"]:
        #         logger.info(f"Extracting {function} data for {symbol}")
        #         # Extract data
        #         df = extractor.extract(
        #             api_key=api_key,
        #             symbol=symbol,
        #             function=function,
        #         )

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
    logger.info("ETL process completed successfully.")
