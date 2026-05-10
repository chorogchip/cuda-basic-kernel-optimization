KERNEL_NAME := spmv
SRC := $(CURDIR)/src/spmv.cu
CONFIG_DIR := $(CURDIR)/configs
TARGETS_FILE := $(CONFIG_DIR)/target_params.txt
PARAM_DEFINES := MY_BLOCKDIM MY_ROW_NNZ
VERSION_DEFINE := SPMV_VARIANT
VERSIONS := 1
LDLIBS := -lcusparse

include ../../common/kernel.mk
