import duckdb
from datetime import datetime


def migrate_database(db_path="my_database.duckdb"):
    """
    Update the database schema to include missing columns
    """
    print(f"Connecting to database: {db_path}")
    conn = duckdb.connect(db_path)

    # Get current schema
    columns = conn.execute("PRAGMA table_info(option_contracts)").fetchall()
    column_names = [col[1] for col in columns]

    print(f"Current columns: {column_names}")

    # Add missing columns if needed
    missing_columns = []

    # Check for mark_price column
    if "mark_price" not in column_names:
        missing_columns.append(("mark_price", "DECIMAL(20, 4)"))

    # Check for rho column (another Greek letter often included in options data)
    if "rho" not in column_names:
        missing_columns.append(("rho", "DECIMAL(20, 4)"))

    # Add any other columns that might be in the updated schema but not in the original

    # Add missing columns
    for col_name, col_type in missing_columns:
        print(f"Adding column: {col_name} ({col_type})")
        conn.execute(f"ALTER TABLE option_contracts ADD COLUMN {col_name} {col_type}")

    if missing_columns:
        print(f"Added {len(missing_columns)} missing columns to option_contracts table")
    else:
        print("No missing columns to add")

    # Verify the updated schema
    updated_columns = conn.execute("PRAGMA table_info(option_contracts)").fetchall()
    updated_column_names = [col[1] for col in updated_columns]
    print(f"Updated columns: {updated_column_names}")

    conn.close()
    return missing_columns


if __name__ == "__main__":
    print("=" * 50)
    print("Database Schema Migration Tool")
    print("=" * 50)

    print(f"Starting migration at {datetime.now()}")
    added_columns = migrate_database()
    print(f"Migration completed at {datetime.now()}")

    if added_columns:
        print(f"\nAdded {len(added_columns)} columns: {[col[0] for col in added_columns]}")
        print("\nYou can now use the updated schema with the main_updated.py script")
    else:
        print("\nNo schema changes were needed")
