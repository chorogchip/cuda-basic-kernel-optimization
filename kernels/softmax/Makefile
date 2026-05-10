KERNEL_NAME := softmax
SRC := $(CURDIR)/src/softmax.cu
CONFIG_DIR := $(CURDIR)/configs
TARGETS_FILE := $(CONFIG_DIR)/target_params.txt
PARAM_DEFINES := MY_BLOCKDIM
VERSION_DEFINE := SOFTMAX_VARIANT
VERSIONS := 1

include ../../common/kernel.mk
