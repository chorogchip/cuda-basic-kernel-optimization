KERNEL_NAME := reduction
SRC := $(CURDIR)/src/reduction.cu
CONFIG_DIR := $(CURDIR)/configs
TARGETS_FILE := $(CONFIG_DIR)/target_params.txt
TARGETS_GEN := $(CONFIG_DIR)/generate_targets.sh
PARAM_DEFINES := MY_BLOCKDIM MY_READPERTHREAD
VERSION_DEFINE := REDUCTION_VERSION
VERSIONS := 1 2
.DEFAULT_GOAL := all

$(TARGETS_FILE): $(TARGETS_GEN)
	@$(TARGETS_GEN)

include ../../common/kernel.mk
