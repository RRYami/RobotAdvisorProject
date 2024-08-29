# Check if requirements.txt exists
if (Test-Path -Path "requirements.txt") {
    Write-Output "requirements.txt found. Activating Virtualenv and updating Python dependencies..."
    
    # Activate the virtual environment
    .\.venv\Scripts\Activate.ps1
    
    # Install/update dependencies from requirements.txt
    pip install -r requirements.txt --upgrade
    
    # Update requirements.txt with the current state of installed packages
    pip freeze > requirements.txt
    
    Write-Output "Dependencies updated and requirements.txt file refreshed."
} else {
    Write-Output "requirements.txt not found. Please make sure the file exists in the current directory."
}