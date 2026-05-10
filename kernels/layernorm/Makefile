KERNEL_NAME := layernorm
SRC := $(CURDIR)/src/layernorm.cu
CONFIG_DIR := $(CURDIR)/configs
TARGETS_FILE := $(CONFIG_DIR)/target_params.txt
PARAM_DEFINES := MY_BLOCKDIM
VERSION_DEFINE := LAYERNORM_VARIANT
VERSIONS := 1

include ../../common/kernel.mk
