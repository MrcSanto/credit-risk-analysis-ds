PYTHON     = python
DATA_RAW   = data/raw/credit_risk_dataset.csv
DATA_CLEAN = data/processed/credit_clean.csv

.PHONY: all install data notebooks dashboard report clean help

help:
	@echo ""
	@echo "  make install    Instala as dependencias via pip"
	@echo "  make data       Gera data/processed/credit_clean.csv"
	@echo "  make notebooks  Executa os 3 notebooks em ordem"
	@echo "  make dashboard  Inicia o dashboard (localhost:8501)"
	@echo "  make report      Exporta o relatorio para PDF (requer pandoc)"
	@echo "  make report-html Exporta o relatorio para HTML (sem pandoc)"
	@echo "  make all        install + data + notebooks"
	@echo ""

install:
	pip install -r requirements.txt
	$(PYTHON) -m playwright install chromium

$(DATA_CLEAN): $(DATA_RAW)
	$(PYTHON) -c "import sys; sys.path.insert(0, '.'); \
from pathlib import Path; \
from src.data import load_raw_data, clean_data, save_processed_data; \
df = load_raw_data('$(DATA_RAW)'); \
df = clean_data(df); \
Path('data/processed').mkdir(parents=True, exist_ok=True); \
save_processed_data(df, '$(DATA_CLEAN)'); \
print('Gerado: $(DATA_CLEAN)')"

data: $(DATA_CLEAN)

notebooks: data
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling.ipynb

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

# Exporta o relatório para PDF (via Playwright/Chromium — sem pandoc)
report:
	$(PYTHON) -m jupyter nbconvert --to html --no-input reports/relatorio.ipynb --output relatorio --output-dir reports/
	$(PYTHON) -c "from playwright.sync_api import sync_playwright; from pathlib import Path; html=Path('reports/relatorio.html').resolve(); pdf=Path('reports/relatorio.pdf').resolve(); p=sync_playwright().__enter__(); b=p.chromium.launch(); pg=b.new_page(); pg.goto(html.as_uri(),wait_until='networkidle'); pg.pdf(path=str(pdf),format='A4',margin={'top':'15mm','bottom':'15mm','left':'15mm','right':'15mm'},print_background=True); b.close(); print(f'PDF: {pdf}')"

# Exporta para HTML sem código
report-html:
	$(PYTHON) -m jupyter nbconvert --to html --no-input reports/relatorio.ipynb --output relatorio --output-dir reports/

all: install data notebooks

clean:
	rm -f $(DATA_CLEAN)
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true
