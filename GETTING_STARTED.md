# Getting Started — E-Commerce Sales Analysis & Prediction System

This guide gets the project running on a fresh machine, from zero to a
working test suite and real data pulled in, in order, with exact commands.

Everything below assumes:
- Windows with git-bash (MSYS) shell — same commands work on macOS/Linux bash too.
- `uv` is installed (https://docs.astral.sh/uv/). Check with `uv --version`.

---

## Step 1 — Get the code

```bash
git clone https://github.com/op4704/ecommerce-sales-analysis.git
cd ecommerce-sales-analysis
```

If you already have the folder locally, just:
```bash
cd "O:\Development\Data Analysis & Prediction System"
```

---

## Step 2 — Install dependencies

This creates a `.venv` and installs everything from `pyproject.toml` /
`uv.lock` (pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels,
jupyter, openpyxl, pytest).

```bash
uv sync
```

First run downloads ~130 packages and can take several minutes on a slow
connection (jupyterlab, statsmodels, and pandas are the largest). If it
times out, retry with a longer timeout:

```bash
UV_HTTP_TIMEOUT=600 uv sync
```

---

## Step 3 — Verify the install with the test suite

Run this before touching any data — it proves the environment and the
core pipeline logic (cleaning rules, data loader caching) both work,
using small synthetic data (no download needed):

```bash
uv run pytest -v
```

Expected: `11 passed`.

---

## Step 4 — Pull the real dataset

The dataset is **not committed to git** (it's ~135MB combined) — you pull
it fresh via the loader module. This one command downloads the "Online
Retail II" dataset directly from the UCI archive, caches it locally, and
prints a summary:

```bash
uv run python -m ecommerce_sales_analysis.data_loader
```

Expected output:
```
Loaded 1,067,371 rows, 8 columns
```

This creates `data/raw/online_retail_II.xlsx` (original) and
`data/raw/online_retail_ii.csv` (cached copy notebooks/scripts read from —
subsequent runs use this cache instead of re-downloading).

---

## Step 5 — Run the cleaning pipeline

Sanity-check that raw data cleans correctly end-to-end:

```bash
uv run python -c "
from ecommerce_sales_analysis.data_loader import load_raw_data
from ecommerce_sales_analysis.cleaning import clean_online_retail, cleaning_summary

raw = load_raw_data()
cleaned = clean_online_retail(raw)
print(cleaning_summary(raw, cleaned))
"
```

Expected output:
```
{'raw_rows': 1067371, 'cleaned_rows': 779425, 'rows_removed': 287946, 'pct_removed': 26.98}
```

---

## Step 6 — Launch Jupyter and work through the notebooks

```bash
uv run jupyter lab
```

This opens Jupyter Lab in your browser. Work through the notebooks **in
numeric order** — each one builds on the last and explains *why* each step
matters, not just the code:

1. `notebooks/01_data_loading.ipynb` — inspect the raw data
2. `notebooks/02_data_cleaning.ipynb` — clean it, save to `data/processed/`
3. *(03 onward: EDA, feature engineering, modeling — not built yet, see Roadmap below)*

To run a single notebook headlessly (e.g. in CI) instead of opening Jupyter Lab:
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_data_loading.ipynb
```

---

## Step 7 — Open the project in VS Code (optional)

```bash
code "O:\Development\Data Analysis & Prediction System"
```

Select the `.venv` interpreter when prompted (or via
`Ctrl+Shift+P` → "Python: Select Interpreter" → `.venv/Scripts/python.exe`)
so VS Code's Jupyter extension and linting use the right environment.

---

## Common commands cheat sheet

| Task | Command |
|---|---|
| Install/update all deps | `uv sync` |
| Add a new dependency | `uv add <package>` |
| Add a dev-only dependency | `uv add --dev <package>` |
| Run tests | `uv run pytest -v` |
| Re-download raw data (ignore cache) | `uv run python -c "from ecommerce_sales_analysis.data_loader import load_raw_data; load_raw_data(force_download=True)"` |
| Launch Jupyter Lab | `uv run jupyter lab` |
| Run a script inside the venv | `uv run python <script>.py` |
| Run the module directly | `uv run python -m ecommerce_sales_analysis.data_loader` |

---

## Roadmap (what's built vs. what's next)

- [x] Project scaffold (uv, folder structure, git, GitHub repo)
- [x] Data loading (`data_loader.py`, notebook 01)
- [x] Data cleaning (`cleaning.py`, notebook 02)
- [x] Test suite (`tests/`, 11 tests)
- [x] Notebook 03 — EDA & visualization
- [ ] Notebook 04 — feature engineering
- [ ] Notebook 05 — model training
- [ ] Notebook 06 — model evaluation
- [ ] Notebook 07 — prediction & final report
