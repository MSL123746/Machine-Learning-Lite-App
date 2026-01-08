# PowerShell launcher for Machine Learning Lite Streamlit app
# Usage: right-click -> Run with PowerShell, or run from PowerShell prompt

$python = "C:/Users/MLee/AppData/Local/Programs/Python/Python313/python.exe"
$script = "c:\Users\MLee\Dev\streamlit_ml_lite.py"

# Stop any existing Streamlit instances first (safeguard against duplicates)
if (Test-Path "c:\Users\MLee\Dev\stop_streamlit.ps1") {
	& "c:\Users\MLee\Dev\stop_streamlit.ps1"
}

# Start Streamlit in a new console window so logs are visible
Start-Process -FilePath $python -ArgumentList '-m','streamlit','run',$script -WindowStyle Normal
