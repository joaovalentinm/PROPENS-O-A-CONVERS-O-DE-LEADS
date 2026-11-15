# Modelo de Propensão à Conversão de Leads
# Uso: `make all` recria o projeto inteiro do zero.

PYTHON ?= python
NBRUN = $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace

.PHONY: all install data train notebooks test clean

all: install data train notebooks test ## Executa tudo do zero

install: ## Instala as dependências
	$(PYTHON) -m pip install -r requirements.txt

data: ## Gera os dados simulados em data/raw/leads.csv
	$(PYTHON) -m src.generate_data

train: ## Treina os modelos e salva em models/
	$(PYTHON) -m src.train

notebooks: ## Executa os notebooks e atualiza as figuras
	$(NBRUN) notebooks/01_eda.ipynb
	$(NBRUN) notebooks/02_modelagem.ipynb
	$(NBRUN) notebooks/03_impacto_negocio.ipynb

test: ## Roda os testes
	$(PYTHON) -m pytest -q

clean: ## Remove artefatos gerados
	rm -f data/raw/*.csv models/*.joblib
	find data/processed reports/figures -type f ! -name .gitkeep -delete
