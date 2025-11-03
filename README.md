Quality-cycles — Quick start

1) Create venv

# Windows PowerShell
python -m venv .venv; .\.venv\Scripts\Activate.ps1

2) Install deps

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

3) Run Streamlit (from project root)

# recommended (ensures `src` package is importable)
python -m streamlit run app.py

Notes:
- `app.py` already prepends the project root to sys.path so running `python -m streamlit run app.py` from the project root should allow `from src...` imports to work.
- If your editor's linter (pylance) still flags unresolved imports, make sure the workspace folder is set to the project root. In VS Code: "File -> Open Folder" and choose the `quality-cycles` folder, or set PYTHONPATH in workspace settings.
