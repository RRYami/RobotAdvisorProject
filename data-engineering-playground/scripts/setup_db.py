"""
Script to set up the PostgreSQL database with the required schemas and tables
"""
from src.utils.logger import setup_logger
from src.database.schema import Schema
from src.database.connection import DatabaseConnection
import os
import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))


logger = setup_logger("setup_db")


def setup_database():
    """
    Set up the database with required schemas and tables
    """
    try:
        logger.info("Starting database setup...")

        # Create database connection
        db_conn = DatabaseConnection()

        with db_conn:
            # Create schema manager and initialize schema
            schema_manager = Schema(db_conn)

            # Create schema
            logger.info("Creating database schema...")
            schema_manager.create_schema()

            # Create tables
            logger.info("Creating database tables...")
            schema_manager.create_tables()

        logger.info("Database setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


if __name__ == "__main__":
    result = setup_database()
    sys.exit(0 if result else 1)
