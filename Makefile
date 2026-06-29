# Makefile for the Stage_2A project
# Quick usage:
#   make            -> create the venv + install dependencies (the "install" target)
#   make run        -> run Python/Exemples.py
#   make clean      -> remove the venv and Python caches
#   make help       -> show help

# --- Configuration ---------------------------------------------------------
VENV        := venv
PYTHON      := python3
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP    := $(VENV)/bin/pip
REQS        := requirements.txt
# Sentinel file: marks that dependencies have been installed
STAMP       := $(VENV)/.installed

.DEFAULT_GOAL := install

# --- Virtualenv creation ---------------------------------------------------
$(VENV_PYTHON):
	@echo ">> Creating the virtual environment in ./$(VENV)"
	$(PYTHON) -m venv $(VENV)
	@$(VENV_PIP) install --upgrade pip

# --- Dependency installation ----------------------------------------------
# Re-runs automatically if requirements.txt changes.
$(STAMP): $(VENV_PYTHON) $(REQS)
	@echo ">> Installing dependencies from $(REQS)"
	$(VENV_PIP) install -r $(REQS)
	@touch $(STAMP)

.PHONY: install
install: $(STAMP)   ## Create the venv and install all dependencies
	@echo ""
	@echo "============================================================"
	@echo " Environment ready!"
	@echo "------------------------------------------------------------"
	@echo " Next steps:"
	@echo "   1) Activate the venv:  source $(VENV)/bin/activate"
	@echo "   2) Run a script:       python Python/Exemples.py"
	@echo "      (or directly:       make run)"
	@echo "============================================================"

# --- Run -------------------------------------------------------------------
.PHONY: run
run: $(STAMP)       ## Run Python/Exemples.py
	@echo ">> Running Python/Exemples.py"
	cd Python && ../$(VENV_PYTHON) Exemples.py

# --- Cleanup ---------------------------------------------------------------
.PHONY: clean
clean:              ## Remove the venv and Python caches
	@echo ">> Removing the venv and caches"
	rm -rf $(VENV)
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

# --- Help ------------------------------------------------------------------
.PHONY: help
help:               ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
