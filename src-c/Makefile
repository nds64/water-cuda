PLATFORM=gcc
include ../config/Makefile.in.$(PLATFORM)
CC=$(NVCC)
CFLAGS=$(NVCCFLAGS)
# ===
# Main driver and sample run

cpu: ldriver.o shallow2d-cpu.o stepper.o
	$(CC) $(CFLAGS) $(LUA_CFLAGS) -o $@ $^ $(LUA_LIBS) $(LIBS) -pg

gpu: ldriver.o shallow2d-gpu.o stepper.o
	$(NVCC) $(NVCCFLAGS) $(LUA_CFLAGS) -o $@ $^ $(LUA_LIBS) $(LIBS) -pg

lshallow: ldriver.o shallow2d.o stepper.o
	$(CC) $(CFLAGS) $(LUA_CFLAGS) -o $@ $^ $(LUA_LIBS) $(LIBS) -pg

ldriver.o: ldriver.c shallow2d.h
	$(CC) $(CFLAGS) $(LUA_CFLAGS) -c $< -pg

shallow2d-cpu.o: shallow2d.c
	$(CC) $(CFLAGS) -c $< -o $@ -pg

shallow2d-gpu.o: shallow2d.cu
	$(NVCC) $(NVVCFLAGS) -c $< -o $@ -pg

stepper.o: stepper.c
	$(CC) $(CFLAGS) -c $< -pg

# ===
# Documentation

shallow.md: shallow2d.h shallow2d.c stepper.h stepper.c ldriver.c
	ldoc $^ -o $@

# ===
# Clean up

.PHONY: clean
clean:
	rm -f lshallow *.o
	rm -f shallow.md
