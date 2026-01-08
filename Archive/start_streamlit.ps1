# Start Streamlit after stopping any existing instances, then open the browser automatically
$python = "C:/Users/MLee/AppData/Local/Programs/Python/Python313/python.exe"
$script = "c:\Users\MLee\Dev\streamlit_ml_lite.py"

# Stop existing instances
if (Test-Path "c:\Users\MLee\Dev\stop_streamlit.ps1") {
    & "c:\Users\MLee\Dev\stop_streamlit.ps1"
}

# Start Streamlit in a new window and capture the process
$proc = Start-Process -FilePath $python -ArgumentList '-m','streamlit','run',$script -WindowStyle Normal -PassThru
Start-Sleep -Seconds 2

# Open browser to localhost (Streamlit usually serves on 8501)
try {
    Start-Process 'http://localhost:8501'
} catch {
    Write-Warning 'Could not open browser automatically. Open http://localhost:8501 manually.'
}

Write-Host "Started Streamlit (PID $($proc.Id)). Give it a few seconds to initialize."
