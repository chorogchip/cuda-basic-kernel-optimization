MAKEFLAGS += --no-print-directory

KERNELS := $(notdir $(wildcard kernels/*))
BASELINE_KERNELS := convolution_2d fused_ops gather_scatter histogram layernorm prefix_scan radix_sort reduction softmax spmv stream_compaction top_k transpose vector_add
RUN_TARGETS := $(addprefix run-,$(KERNELS))
RUN_BASELINE_TARGETS := $(addprefix run-baseline-,$(BASELINE_KERNELS))
CLEAN_RUN_TARGETS := $(addprefix clean-run-,$(KERNELS))

.PHONY: all clean run-all run-baseline-all summarize-all plot-all pipeline-all pipeline-baseline-all clean-run-all $(KERNELS) FORCE

all: $(KERNELS)

run-all: $(RUN_TARGETS)

run-baseline-all: $(RUN_BASELINE_TARGETS)

summarize-all:
	@./scripts/summarize_maxperf.sh

plot-all:
	@./scripts/plot_all_results.sh

pipeline-all: all run-all run-baseline-all summarize-all plot-all

pipeline-baseline-all: all run-baseline-all summarize-all plot-all

clean-run-all: $(CLEAN_RUN_TARGETS)

clean:
	@for kernel in $(KERNELS); do $(MAKE) -C kernels/$$kernel clean; done

$(KERNELS):
	@if [ ! -d kernels/$@ ]; then \
		echo "Unknown kernel: $@"; \
		exit 1; \
	fi
	@$(MAKE) -C kernels/$@

run-%: FORCE
	@if [ ! -d kernels/$* ]; then \
		echo "Unknown kernel: $*"; \
		exit 1; \
	fi
	@$(MAKE) -C kernels/$* run

run-baseline-%: FORCE
	@if [ ! -d kernels/$* ]; then \
		echo "Unknown kernel: $*"; \
		exit 1; \
	fi
	@$(MAKE) -C kernels/$* run-baseline

clean-run-%: FORCE
	@if [ ! -d kernels/$* ]; then \
		echo "Unknown kernel: $*"; \
		exit 1; \
	fi
	@$(MAKE) -C kernels/$* clean-run

pipeline-%: FORCE
	@if [ ! -d kernels/$* ]; then \
		echo "Unknown kernel: $*"; \
		exit 1; \
	fi
	@$(MAKE) -C kernels/$*
	@$(MAKE) -C kernels/$* run
	@if printf '%s\n' $(BASELINE_KERNELS) | grep -qx "$*"; then \
		$(MAKE) -C kernels/$* run-baseline; \
	fi
	@./scripts/summarize_maxperf.sh kernels/$*/results/$*_baseline_run.txt 2>/dev/null || true
	@for file in kernels/$*/results/*_run.txt; do \
		[ -f "$$file" ] && ./scripts/summarize_maxperf.sh "$$file"; \
	done
	@./scripts/plot_elem_per_sec.py kernels/$*/results/*_run.txt
	@maxperf_files=$$(find kernels/$*/results -maxdepth 1 -name '*_run_maxperf.txt' -size +0c | sort); \
	if [ -n "$$maxperf_files" ]; then \
		./scripts/plot_elem_per_sec.py $$maxperf_files; \
	fi

FORCE:
