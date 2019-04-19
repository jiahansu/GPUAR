BUILD_DIR = ./
CPP_FLAG =  --std=c++11 -O3
#-g -D_DEBUG
LIBS = 
SRC = src/
INCLUDE = -Ilib
CPP = nvcc
CPP_FLAG += 
INCLUDE += -I"common"
LIBS = -lcuda -lcudart
EXECUTION=gpuar
OBJECTS = progress_monitor.o compressor.o cpu_compressor.o gpu_compressor.o gpuar_kernel.o main.o

#OBJECTS = cube.o
LD_FLAGS = -Wno-deprecated-gpu-targets
TARGET = $(BUILD_DIR)/$(EXECUTION)

debug_build:$(TARGET) done
clean: 
	rm -rf gpuar vc140.pdb *.o shader *.o.tmp test.dSYM
done:

$(TARGET): $(OBJECTS)
	$(CPP) -o $@ $(LD_FLAGS) $(OBJECTS) $(LIBS)

compressor.o:src/compressor.cpp
	$(CPP) -c -o $@ $< $(CPP_FLAG) $(INCLUDE)

cpu_compressor.o:src/cpu_compressor.cpp
	$(CPP) -c -o $@ $< $(CPP_FLAG) $(INCLUDE)

gpu_compressor.o:src/gpu_compressor.cpp
	$(CPP) -c -o $@ $< $(CPP_FLAG) $(INCLUDE)

progress_monitor.o:src/progress_monitor.cpp
	$(CPP) -c -o $@ $< $(CPP_FLAG) $(INCLUDE)

gpuar_kernel.o:src/gpuar_kernel.cu
	$(CPP) -c -o $@ $< $(CPP_FLAG) $(INCLUDE)

main.o:src/main.cpp
	$(CPP) -c -o $@ $< $(CPP_FLAG) $(INCLUDE)