@echo off
cd /d %~dp0
call .\ml311_env\Scripts\activate.bat
python streamlit_ml_lite.py
pause