# Host test (gcc) and Cortex-M4F size report (arm-none-eabi-gcc).
#   make test        build and run the host test against vectors.csv (scripts/09 first)
#   make size        compile the library for the nRF52840 (Cortex-M4F) and print its footprint
LIB    := autogear_ml/src
SRCS   := $(LIB)/ag_features.c $(LIB)/ag_gbdt.c $(LIB)/ag_controller.c $(LIB)/ag_model_data.c
CFLAGS := -std=c99 -O2 -Wall -Wextra -Werror -I$(LIB)

ARM_CC   ?= arm-none-eabi-gcc
ARM_SIZE ?= arm-none-eabi-size
ARM_FLAGS := -std=c99 -Os -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 \
             -ffunction-sections -fdata-sections -Wall -Wextra -Werror -I$(LIB)

.PHONY: test size clean
test: host_test/host_test
	cd host_test && ./host_test vectors.csv

host_test/host_test: host_test/host_test.c $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

size:
	@mkdir -p build/arm
	@for f in $(SRCS); do $(ARM_CC) $(ARM_FLAGS) -c $$f -o build/arm/$$(basename $$f .c).o || exit 1; done
	$(ARM_SIZE) -t build/arm/*.o

clean:
	rm -rf build host_test/host_test
