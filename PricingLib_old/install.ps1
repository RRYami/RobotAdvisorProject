py --version

Write-Output "Creating and activating Virtual Environment"
py -m venv .venv
.venv/Scripts/activate.ps1

Write-Output "Installing required Python Packages"
py -m pip install --upgrade pip
pip install -r requirements.txt

deactivate