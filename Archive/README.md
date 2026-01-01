Machine Learning Lite

Run the Streamlit app:

PowerShell (recommended):

```powershell
C:/Users/MLee/AppData/Local/Programs/Python/Python313/python.exe -m streamlit run "c:\Users\MLee\Dev\streamlit_ml_lite.py"
```

Or, if `streamlit` is on your PATH:

```powershell
streamlit run "c:\Users\MLee\Dev\streamlit_ml_lite.py"
```

Notes:
- Sample CSVs (`iris.csv`, `housing_small.csv`) are provided in the same folder for quick testing.
- If you upload a CSV, choose the target and features in Step 1, then Step 2 -> Start Training.
- Predictions (single and batch) are aligned to the training features automatically where possible.
- Requirements: see `requirements.txt` (streamlit, pandas, scikit-learn, matplotlib, numpy).

Launchers and shortcuts
 - Quick run (explicit Python path):
```powershell
C:/Users/MLee/AppData/Local/Programs/Python/Python313/python.exe -m streamlit run "c:\Users\MLee\Dev\streamlit_ml_lite.py"
```

 - Use the provided launchers (no typing):
	 - `run_streamlit.bat` — double-click to start the server and keep the terminal visible.
	 - `run_streamlit.ps1` — PowerShell launcher that starts the server in a new window.

Make `streamlit` available globally (optional)
- If you'd prefer to run `streamlit run ...` without the full python path, install it with `pipx` (recommended) or add the Python Scripts folder to your PATH.

Install `pipx` (if not installed) and then install Streamlit:
```powershell
C:/Users/MLee/AppData/Local/Programs/Python/Python313/python.exe -m pip install --user pipx
C:/Users/MLee/AppData/Local/Programs/Python/Python313/python.exe -m pipx ensurepath
pipx install streamlit
```

Creating a Desktop shortcut (optional)
- A helper script `create_shortcut.ps1` is included; run it to create a Desktop shortcut that launches the app (PowerShell will create `ML_Lite.lnk` on your Desktop):
```powershell
& 'c:\Users\MLee\Dev\create_shortcut.ps1'
```

Smoke test & troubleshooting
- After starting the server, open `http://localhost:8501` in your browser.
- If you see a warning about "missing ScriptRunContext" when importing Streamlit, it is benign — the app should still run normally when launched with `streamlit run`.
- If the app does not appear, check the server terminal for ERROR/TRACEBACK lines and paste them here so I can help troubleshoot.
