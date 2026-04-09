ROOT_DIR := $(abspath $(CURDIR)/../..)
KERNEL_DIR := $(CURDIR)
BIN_DIR := $(KERNEL_DIR)/bin
RESULTS_DIR := $(KERNEL_DIR)/results
# Each kernel family writes one aggregated sweep log under its own results dir.
RUN_OUTPUT_FILE := $(RESULTS_DIR)/$(KERNEL_NAME)_run.txt

NVCC ?= nvcc
NVCCFLAGS ?= -O3

MAIN_SRC := $(ROOT_DIR)/common/cumain.cu
LIB_SRC := $(ROOT_DIR)/common/culib.cu
# Ignore blank lines and comments so sizes.txt stays easy to annotate.
SIZES := $(shell sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$$/d' $(CONFIG_DIR)/sizes.txt)
TARGETS := $(addprefix $(BIN_DIR)/,$(TARGET_NAMES))

.PHONY: all run clean-run clean

all: $(TARGETS)

run:
	@$(MAKE) --no-print-directory -s $(RUN_OUTPUT_FILE)

$(RUN_OUTPUT_FILE): $(TARGETS)
	@mkdir -p $(dir $@)
	@: > $@
	@# Run every built target across every configured size and append to one file.
	@for target in $(TARGET_NAMES); do \
		for n in $(SIZES); do \
			printf 'running: %s %s\n' "$$target" "$$n"; \
			./bin/$$target $$n >> $@; \
		done; \
	done

$(BIN_DIR):
	@mkdir -p $@

# TARGET_FLAGS_<name> comes from the kernel-local targets.mk include.
$(BIN_DIR)/%: $(SRC) $(MAIN_SRC) $(LIB_SRC) $(TARGETS_FILE) | $(BIN_DIR)
	@printf 'building: %s\n' "$(notdir $@)"
	@$(NVCC) $(SRC) $(MAIN_SRC) $(LIB_SRC) $(NVCCFLAGS) $(TARGET_FLAGS_$*) \
		-DMY_CUDA_SOURCE_NAME="\"$(KERNEL_NAME)\"" \
		-DMY_CUDA_BUILD_CONFIG="\"config=$(notdir $(CONFIG_DIR))/$(notdir $(TARGETS_FILE)),target=$*\"" \
		-o $@

clean-run:
	@rm -f $(RUN_OUTPUT_FILE)

clean:
	@rm -f $(TARGETS)
