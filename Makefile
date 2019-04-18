BUILD_DIR = ./
CPP_FLAG =  --std=c++11 -g -D_DEBUG
LIBS = 
SRC = src/
INCLUDE = -Ilib
CPP = nvcc
CPP_FLAG += 
INCLUDE += -I"common"
LIBS = -lcuda -lcudart
EXECUTION=gpuar
OBJECTS = container.o gpuar_kernel.o main.o

#OBJECTS = cube.o
LD_FLAGS = -Wno-deprecated-gpu-targets
TARGET = $(BUILD_DIR)/$(EXECUTION)

debug_build:$(TARGET) done
clean: 
	rm -rf gpuar vc140.pdb *.o shader *.o.tmp test.dSYM
done:

$(TARGET): $(OBJECTS)
	$(CPP) -o $@ $(LD_FLAGS) $(OBJECTS) $(LIBS)


container.o:src/container.cpp
	$(CPP) -c -o $@ $< $(CPP_FLAG) $(INCLUDE)

gpuar_kernel.o:src/gpuar_kernel.cu
	$(CPP) -c -o $@ $< $(CPP_FLAG) $(INCLUDE)

main.o:src/main.cpp
	$(CPP) -c -o $@ $< $(CPP_FLAG) $(INCLUDE)