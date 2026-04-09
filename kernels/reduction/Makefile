KERNEL_NAME := reduction
SRC := $(CURDIR)/src/reduction.cu
CONFIG_DIR := $(CURDIR)/configs
TARGETS_FILE := $(CONFIG_DIR)/targets.mk
TARGETS_GEN := $(CONFIG_DIR)/generate_targets.sh
.DEFAULT_GOAL := all

$(TARGETS_FILE): $(TARGETS_GEN)
	@$(TARGETS_GEN)

-include $(TARGETS_FILE)

include ../../common/kernel.mk
