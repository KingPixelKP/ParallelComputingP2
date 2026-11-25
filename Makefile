CUDA_CHECK ?= OFF

ifneq ($(CUDA_CHECK), OFF)
    $(info Building with CUDA checks enabled.)
    CUDA_FLAGS := -DCHECK
else
    $(info Building without CUDA checks.)
    CUDA_FLAGS :=
endif

NUM_CORES := $(shell nproc)
CMAKE_CONFIG_FLAGS := -DCMAKE_BUILD_TYPE=Release \
					   -G "Unix Makefiles" \
					   -DCMAKE_CUDA_FLAGS="$(CUDA_FLAGS)" \
					   -DCMAKE_CXX_FLAGS="$(CUDA_FLAGS)"
CMAKE_BUILD_FLAGS := -j$(NUM_CORES)

.PHONY: all clean clean_logs clean_build configure build build_pipeline help
.DEFAULT_GOAL := help

help: # Thanks to https://stackoverflow.com/users/4317383/rose in https://stackoverflow.com/a/64996042
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

all: build ## Build the project

clean_logs: ## Clean log files
	@echo "Cleaning log files..."
	rm -f logs/*.log || true
	rm -f logs/*.txt || true
	@echo "Log files cleaned."

clean_build: ## Clean build directory
	@echo "Cleaning build directory..."
	rm -rf build bin || true
	@echo "Build directory cleaned."

clean: clean_logs clean_build ## Clean both log files and build directory
	@echo "Clean completed."

configure: ## Configure the project using CMake
	@echo "Configuring project..."
	cmake -S . -B build $(CMAKE_CONFIG_FLAGS) | tee logs/configure.log
	@echo "Configuration completed."

build: configure ## Build the project
	@echo "Building project using $(NUM_CORES) cores..."
	cmake --build build $(CMAKE_BUILD_FLAGS) | tee logs/build.log
	@echo "Build completed."

build_pipeline: configure ## Build the block_forward target for the pipeline
	@echo "Building target block_forward using $(NUM_CORES) cores..."
	cmake --build build $(CMAKE_BUILD_FLAGS) --target block_forward | tee logs/build_block_forward.log
	@echo "Target block_forward build completed."

test: build ## Run tests
	@echo "Running tests..."
# Find all _test in bin
	@find bin -type f -name '*_test' -exec {} \; | tee logs/test.log
	@echo "Tests completed."