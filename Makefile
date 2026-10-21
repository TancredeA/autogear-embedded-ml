# Full pipeline. `make all` reproduces every number and figure of the report (about 35 min on 2 cores).
PY := python3

.PHONY: all install data dataset validate event main timing cross analyse firmware figures report test clean
all: dataset validate event main timing cross analyse firmware figures

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

data:            ; $(PY) scripts/01_download_sources.py
dataset: data    ; $(PY) scripts/02_build_dataset.py
validate:        ; $(PY) scripts/03_validate_gear_reconstruction.py
event:           ; $(PY) scripts/04_event_classifier_baseline.py
main:            ; $(PY) scripts/05_target_models_vs_tables.py
timing:          ; $(PY) scripts/06_timing_sweep.py
cross:           ; $(PY) scripts/07_cross_rider.py
analyse:         ; $(PY) scripts/08_analyse_model.py
firmware: analyse
	$(PY) scripts/09_export_firmware_model.py
	$(MAKE) -C firmware test
figures:         ; $(PY) scripts/10_make_figures.py
report:          ; cd report && latexmk -pdf -interaction=nonstopmode main.tex && cp main.pdf autogear_extension_report.pdf
test:            ; $(PY) -m pytest -q
clean:           ; rm -rf data/processed firmware/build
