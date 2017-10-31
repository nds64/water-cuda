PLATFORM=gcc
include ../config/Makefile.in.$(PLATFORM)
CC=$(NVCC)
CFLAGS=$(NVCCFLAGS)

# ===
# Main driver and sample run

lshallow: ldriver.o shallow2d.o stepper.o
	$(CC) $(CFLAGS) $(LUA_CFLAGS) -o $@ $^ $(LUA_LIBS) $(LIBS) -pg

ldriver.o: ldriver.c shallow2d.h
	$(CC) $(CFLAGS) $(LUA_CFLAGS) -c $< -pg

shallow2d.o: shallow2d.cu
	$(CC) $(CFLAGS) -c $< -pg

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
