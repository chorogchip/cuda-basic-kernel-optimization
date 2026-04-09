MAKEFLAGS += --no-print-directory

KERNELS := $(notdir $(wildcard kernels/*))
RUN_TARGETS := $(addprefix run-,$(KERNELS))
CLEAN_RUN_TARGETS := $(addprefix clean-run-,$(KERNELS))

.PHONY: all clean run-all clean-run-all $(KERNELS) FORCE

all: $(KERNELS)

run-all: $(RUN_TARGETS)

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

clean-run-%: FORCE
	@if [ ! -d kernels/$* ]; then \
		echo "Unknown kernel: $*"; \
		exit 1; \
	fi
	@$(MAKE) -C kernels/$* clean-run

FORCE:
