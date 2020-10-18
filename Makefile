CC=icpc #clang
CFLAGS=-std=c++11 -g `llvm-config --cflags`
LD=icpc#clang++
LDFLAGS=`llvm-config --cxxflags --ldflags --libs core executionengine mcjit interpreter analysis native bitwriter --system-libs`

all: int_jit_llvm

int_jit_llvm.o: int_jit_llvm.c
	    $(CC) $(CFLAGS) -c $<

int_jit_llvm: int_jit_llvm.o
	    $(LD) $< $(LDFLAGS) -o $@

int_jit_llvm.bc: int_jit_llvm
	    ./int_jit_llvm 0 0

int_jit_llvm.ll: int_jit_llvm.bc
	    llvm-dis $<

clean:
	    -rm -f int_jit_llvm.o int_jit_llvm int_jit_llvm.bc int_jit_llvm.ll
