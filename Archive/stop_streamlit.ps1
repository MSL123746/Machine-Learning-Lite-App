# Stop any running Streamlit processes (by matching 'streamlit' in process command line)
$ps = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match 'streamlit') }
if ($ps) {
    $ps | ForEach-Object {
        Write-Host ("Stopping PID {0}: {1}" -f $_.ProcessId, $_.CommandLine)
        try {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
        } catch {
            Write-Warning "Failed to stop PID $_.ProcessId: $_"
        }
    }
} else {
    Write-Host 'No streamlit processes found.'
}
