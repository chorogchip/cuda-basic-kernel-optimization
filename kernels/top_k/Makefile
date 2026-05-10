KERNEL_NAME := top_k
SRC := $(CURDIR)/src/top_k.cu
CONFIG_DIR := $(CURDIR)/configs
TARGETS_FILE := $(CONFIG_DIR)/target_params.txt
PARAM_DEFINES := MY_BLOCKDIM MY_TOPK
VERSION_DEFINE := TOP_K_VARIANT
VERSIONS := 1

include ../../common/kernel.mk
