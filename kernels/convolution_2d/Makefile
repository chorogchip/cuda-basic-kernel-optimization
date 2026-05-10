KERNEL_NAME := convolution_2d
SRC := $(CURDIR)/src/convolution_2d.cu
CONFIG_DIR := $(CURDIR)/configs
TARGETS_FILE := $(CONFIG_DIR)/target_params.txt
PARAM_DEFINES := MY_TILE_DIM MY_FILTER_RADIUS
VERSION_DEFINE := CONVOLUTION_2D_VARIANT
VERSIONS := 1

include ../../common/kernel.mk
