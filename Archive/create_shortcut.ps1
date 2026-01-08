# Create a Desktop shortcut that launches the Streamlit app via the PowerShell launcher
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop 'ML_Lite.lnk'
$targetPowershell = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
$launcher = 'C:\Users\MLee\Dev\run_streamlit.ps1'

$WshShell = New-Object -ComObject WScript.Shell
$shortcut = $WshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPowershell
$shortcut.Arguments = "-NoExit -Command \"& '$launcher'\""
$shortcut.WorkingDirectory = 'C:\Users\MLee\Dev'
$shortcut.IconLocation = "$targetPowershell,0"
$shortcut.Save()
Write-Host "Created shortcut: $shortcutPath"
