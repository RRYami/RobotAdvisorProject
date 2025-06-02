#!/bin/bash

# Script to set up PostgreSQL container and import data
# Usage: ./setup.sh

# Check if XNAS_FILE.csv exists in the current directory
XNAS_FILE="XNAS_FILE.csv"
if [ ! -f "$XNAS_FILE" ]; then
    echo "Error: File $XNAS_FILE not found in the current directory"
    exit 1
fi

echo "Starting PostgreSQL and pgAdmin with Docker..."
docker-compose up -d

echo "Waiting for PostgreSQL to start..."
for i in {1..30}; do
    if docker exec postgres-database_test pg_isready -U ADMIN -d database_test; then
        echo "PostgreSQL is ready!"
        break
    fi
    echo "Waiting for PostgreSQL... (attempt $i/30)"
    sleep 2
done

if [ $i -eq 30 ]; then
    echo "Error: PostgreSQL did not start within 60 seconds"
    exit 1
fi

echo "Importing data with import.py..."
python import.py "$XNAS_FILE"

echo "Setup complete! Your data is now available in PostgreSQL."
echo ""
echo "PostgreSQL connection details:"
echo "  - Host: localhost"
echo "  - Port: 5432"
echo "  - Database: database_test"
echo "  - User: ADMIN"
echo "  - Password: ADMIN"
echo ""
echo "pgAdmin is available at: http://localhost:5050"
echo "  - Email: admin@example.com"
echo "  - Password: ADMIN"
echo ""
echo "To connect to your database in pgAdmin:"
echo "1. Log in to pgAdmin at http://localhost:5050"
echo "2. Right-click on 'Servers' and select 'Create' > 'Server...'"
echo "3. Give it a name like 'Database Test'"
echo "4. Go to the 'Connection' tab and enter:"
echo "   - Host: postgres"
echo "   - Port: 5432"
echo "   - Database: database_test"
echo "   - Username: ADMIN"
echo "   - Password: ADMIN"
echo "5. Click 'Save'"