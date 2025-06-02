import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatabaseConnection:
    """
    Postgres database connection class
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize database connection with config

        Args:
            config_path (Optional[str], optional): Path to config file. 
                                                   Defaults to None.
        """
        self.conn = None
        self.cursor = None

        if config_path:
            self.config = self._load_config(config_path)
        else:
            # Default to environment variables
            self.config = {
                'host': os.environ.get('DB_HOST', 'localhost'),
                'port': int(os.environ.get('DB_PORT', 5432)),
                'database': os.environ.get('DB_NAME', 'postgres'),
                'user': os.environ.get('DB_USER', 'postgres'),
                'password': os.environ.get('DB_PASSWORD', 'postgres'),
            }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load database configuration from YAML file

        Args:
            config_path (str): Path to the config file

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config.get('database', {})
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def connect(self) -> None:
        """
        Establish connection to the database
        """
        try:
            self.conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error connecting to the database: {e}")
            raise

    def close(self) -> None:
        """
        Close database connection
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def execute_query(self, query: str, params: tuple | None) -> list:
        """
        Execute a query and return the results

        Args:
            query (str): SQL query to execute
            params (tuple | None): Parameters for the query.

        Returns:
            list: Query results
        """
        if not self.conn or not self.cursor:
            logger.error("Database connection is not established. Call connect() first.")
            raise Exception("Database connection is not established. Call connect() first.")
        try:
            self.cursor.execute(query, params)

            # If query starts with SELECT, fetch results
            if query.strip().upper().startswith("SELECT"):
                result = self.cursor.fetchall()
                return result
            else:
                self.conn.commit()
                return []
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error executing query: {e}")
            raise

    def __enter__(self):
        """
        Context manager entry
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit
        """
        self.close()
