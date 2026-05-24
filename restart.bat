@echo off
echo Starting API server...
start cmd /k "cd /d %~dp0 && python api.py"

timeout /t 3 /nobreak > nul

echo Starting Streamlit interface...
start cmd /k "cd /d %~dp0 && streamlit run streamlit_app.py --server.port 8501"

echo Done! 
echo API: http://localhost:8000/docs
echo Streamlit: http://localhost:8501