@echo off
:: ╔══════════════════════════════════════════════════════════════════════╗
:: ║  Semantic Drift Analysis — Quick Start                              ║
:: ║  B.Tech NLP Major Project                                           ║
:: ╚══════════════════════════════════════════════════════════════════════╝

echo.
echo  ⚗  Semantic Drift Analysis — B.Tech NLP Major Project
echo  ══════════════════════════════════════════════════════
echo.

:: Check venv
if not exist ".venv\Scripts\python.exe" (
  echo  [ERROR] Virtual environment not found.
  echo  Run:   python -m venv .venv
  echo  Then:  .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)

:: Check data
if not exist "sgns\" (
  echo  [WARN] sgns/ folder not found.
  echo  Run:  .venv\Scripts\python main.py download
  echo  Or manually download from:
  echo    http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip
  echo  and unzip here.
  echo.
)

echo  Starting API server...
echo  Dashboard will open automatically.
echo  If not, open:  frontend\index.html  in your browser.
echo.

.venv\Scripts\python main.py serve

pause
