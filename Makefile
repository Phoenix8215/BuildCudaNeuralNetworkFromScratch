CONFIG_LOCAL := config/Makefile.config

include $(CONFIG_LOCAL)

BUILD_PATH    := build
SRC_PATH      := src
INC_PATH      := include
CUDA_DIR      := /usr/local/cuda-$(CUDA_VER)



KERNELS_SRC   := $(wildcard $(SRC_PATH)/*.cu) \
                 $(wildcard $(SRC_PATH)/nn_utils/*.cu) \
                 $(wildcard $(SRC_PATH)/layers/*.cu) \
                 $(wildcard $(SRC_PATH)/cost_functions/*.cu) \
                 $(wildcard $(SRC_PATH)/datasets/*.cu)

APP_OBJS      := $(patsubst $(SRC_PATH)/%, $(BUILD_PATH)/%, $(KERNELS_SRC:.cu=.cu.o))

APP_MKS       := $(APP_OBJS:.o=.mk)

APP_DEPS      := $(KERNELS_SRC)
APP_DEPS      += $(wildcard $(SRC_PATH)/*.h)
APP_DEPS      += $(wildcard $(SRC_PATH)/nn_utils/*.h)
APP_DEPS      += $(wildcard $(SRC_PATH)/layers/*.h)
APP_DEPS      += $(wildcard $(SRC_PATH)/cost_functions/*.h)
APP_DEPS      += $(wildcard $(SRC_PATH)/datasets/*.h)

# -----------------------------------------------------

CUCC          := $(CUDA_DIR)/bin/nvcc
CXXFLAGS      := -std=c++17 -pthread -fPIC
CUDAFLAGS     := --shared -Xcompiler -fPIC

INCS          := -I $(CUDA_DIR)/include \
                 -I $(SRC_PATH) \
                 -I $(SRC_PATH)/layers \
                 -I $(SRC_PATH)/nn_utils \
                 -I $(SRC_PATH)/cost_functions \
                 -I $(SRC_PATH)/datasets \
                 -I $(INC_PATH) \

LIBS          := -L "$(CUDA_DIR)/lib64" \
                 -lcudart -lcublas -lcudnn \
				 -lcurand

ifeq ($(DEBUG),1)
CUDAFLAGS     += -g -O0 -G
CXXFLAGS      += -g -O0
else
CUDAFLAGS     += -O3
CXXFLAGS      += -O3
endif

ifeq ($(SHOW_WARNING),1)
CUDAFLAGS     += -Wall -Wunused-function -Wunused-variable -Wfatal-errors
CXXFLAGS      += -Wall -Wunused-function -Wunused-variable -Wfatal-errors
else
CUDAFLAGS     += -w
CXXFLAGS      += -w
endif

.PHONY: all update show clean $(APP)

all: 
	$(MAKE) $(APP)

update: $(APP)
	@echo finished updating 😎😎😎$<

$(APP): $(APP_DEPS) $(APP_OBJS)
	@$(CXX) $(APP_OBJS) -o $@ $(LIBS) $(INCS)
	@echo finished building $@. Have fun!!🥰🥰🥰

show: 
	@echo $(BUILD_PATH)
	@echo $(APP_DEPS)
	@echo $(INCS)
	@echo $(APP_OBJS)
	@echo $(APP_MKS)

clean:
	-rm -rf $(APP) 😭
	-rm -rf $(BUILD_PATH) 😭

ifneq ($(MAKECMDGOALS), clean)
-include $(APP_MKS)
endif

# Create necessary directories
$(BUILD_PATH)/%/:
	@mkdir -p $@

# Compile CXX
$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/%.cpp | $(BUILD_PATH)/%/
	@echo Compile CXX $@
	@$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCS)

$(BUILD_PATH)/%.cpp.mk: $(SRC_PATH)/%.cpp | $(BUILD_PATH)/%/
	@echo Compile Dependence CXX $@
	@$(CXX) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(CXXFLAGS) $(INCS) 

# Compile CUDA
$(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.cu | $(BUILD_PATH)/%/
	@echo Compile CUDA $@
	@$(CUCC) -o $@ -c $< $(CUDAFLAGS) $(INCS)

$(BUILD_PATH)/%.cu.mk: $(SRC_PATH)/%.cu | $(BUILD_PATH)/%/
	@echo Compile Dependence CUDA $@
	@$(CUCC) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(CUDAFLAGS)
