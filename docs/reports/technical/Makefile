SHELL := /bin/bash
LATEXMK ?= latexmk
PYTHON ?= python3
BUILD_DIR := build
MAIN := main.tex
PDF := $(BUILD_DIR)/main.pdf

.PHONY: all pdf check clean distclean

all: check

pdf:
	mkdir -p $(BUILD_DIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
		-outdir=$(BUILD_DIR) $(MAIN)

check: pdf
	$(PYTHON) tools/check_report.py --root . --aux $(BUILD_DIR)/main.aux \
		--log $(BUILD_DIR)/main.log --pdf $(PDF)

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR) $(MAIN)
	rm -f $(BUILD_DIR)/main.bbl $(BUILD_DIR)/main.blg

distclean: clean
	rm -rf $(BUILD_DIR)
