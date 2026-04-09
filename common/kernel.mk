ROOT_DIR := $(abspath $(CURDIR)/../..)
KERNEL_DIR := $(CURDIR)
BIN_DIR := $(KERNEL_DIR)/bin
RESULTS_DIR := $(KERNEL_DIR)/results
# By default, each kernel family writes one aggregated sweep log under its own
# results dir. Versioned kernels can opt into one log per version.
RUN_OUTPUT_FILE := $(RESULTS_DIR)/$(KERNEL_NAME)_run.txt
DEFAULT_ROW := __default__
comma := ,
empty :=
space := $(empty) $(empty)

NVCC ?= nvcc
NVCCFLAGS ?= -O3

MAIN_SRC := $(ROOT_DIR)/common/cumain.cu
LIB_SRC := $(ROOT_DIR)/common/culib.cu
# Ignore blank lines and comments so sizes.txt stays easy to annotate.
SIZES := $(shell sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$$/d' $(CONFIG_DIR)/sizes.txt)
TARGET_ROWS_RAW := $(shell sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$$/d' $(TARGETS_FILE))
TARGET_ROWS := $(if $(strip $(TARGET_ROWS_RAW)),$(TARGET_ROWS_RAW),$(DEFAULT_ROW))
VERSION_VALUES := $(strip $(VERSIONS))
BUILD_ROWS := $(if $(VERSION_VALUES),$(VERSION_VALUES),$(DEFAULT_ROW))

define row_to_values
$(if $(filter $(DEFAULT_ROW),$(1)),,$(subst $(comma),$(space),$(strip $(1))))
endef

define values_to_suffix
$(if $(strip $(1)),_$(subst $(space),_,$(strip $(1))))
endef

define build_to_name
$(KERNEL_NAME)$(call values_to_suffix,$(call row_to_values,$(1)))$(call values_to_suffix,$(call row_to_values,$(2)))
endef

define build_to_flags
$(strip \
	$(if $(VERSION_DEFINE),-D$(VERSION_DEFINE)=$(1)) \
	$(if $(strip $(PARAM_DEFINES)),$(call values_to_flags,$(PARAM_DEFINES),$(call row_to_values,$(2)))) \
)
endef

define values_to_flags
$(strip \
	$(if $(strip $(1)),-D$(firstword $(1))=$(firstword $(2))) \
	$(if $(word 2,$(1)),$(call values_to_flags,$(wordlist 2,$(words $(1)),$(1)),$(wordlist 2,$(words $(2)),$(2)))) \
)
endef

$(foreach build_row,$(BUILD_ROWS), \
	$(foreach param_row,$(TARGET_ROWS), \
		$(eval target_name := $(call build_to_name,$(build_row),$(param_row))) \
		$(eval TARGET_NAMES += $(target_name)) \
		$(eval TARGET_FLAGS_$(target_name) := $(call build_to_flags,$(build_row),$(param_row))) \
		$(if $(filter-out $(DEFAULT_ROW),$(build_row)),$(eval TARGET_NAMES_VERSION_$(build_row) += $(target_name))) \
	) \
)

TARGETS := $(addprefix $(BIN_DIR)/,$(TARGET_NAMES))
RUN_OUTPUT_FILES := $(if $(VERSION_VALUES),$(foreach version,$(VERSION_VALUES),$(RESULTS_DIR)/$(KERNEL_NAME)_$(version)_run.txt),$(RUN_OUTPUT_FILE))

.PHONY: all run clean-run clean

all: $(TARGETS)

run:
	@$(MAKE) --no-print-directory -s $(RUN_OUTPUT_FILES)

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

define make_version_run_rule
$(RESULTS_DIR)/$(KERNEL_NAME)_$(1)_run.txt: $(addprefix $(BIN_DIR)/,$(TARGET_NAMES_VERSION_$(1)))
	@mkdir -p $$(dir $$@)
	@: > $$@
	@for target in $(TARGET_NAMES_VERSION_$(1)); do \
		for n in $(SIZES); do \
			printf 'running: %s %s\n' "$$$$target" "$$$$n"; \
			./bin/$$$$target $$$$n >> $$@; \
		done; \
	done
endef

$(foreach version,$(VERSION_VALUES),$(eval $(call make_version_run_rule,$(version))))

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
	@rm -f $(RUN_OUTPUT_FILE) $(RUN_OUTPUT_FILES)

clean:
	@rm -f $(TARGETS)
