# Common arguments called with all compilation statements
COMMON_ARGUMENTS          := -Wall -Werror -g #-pg
OPTIMIZATION_LEVEL        := -O3
PART1_SERIAL_FILE         := stewart_nate_lab4p1_serial.c
PART1_SERIAL_OUTPUT       := lab4p1_serial
PART1_PARALLEL_FILE       := stewart_nate_lab4p1_parallel.cu
PART1_PARALLEL_OUTPUT     := lab4p1_parallel
PART2_SERIAL_FILE         := stewart_nate_lab4p2_serial.c
PART2_SERIAL_OUTPUT       := lab4p2_serial
PART2_PARALLEL_FILE       := stewart_nate_lab4p2_parallel.cu
PART2_PARALLEL_OUTPUT     := lab4p2_parallel
PART3_PARALLEL_FILE       := stewart_nate_lab4p3.cu
PART3_PARALLEL_OUTPUT     := lab4p3_parallel
BINARIES := $(PART1_SERIAL_OUTPUT) $(PART1_PARALLEL_OUTPUT) $(PART2_SERIAL_OUTPUT) $(PART2_PARALLEL_OUTPUT) $(PART3_PARALLEL_OUTPUT)

all: $(BINARIES) 
# Serial execution. Part 1
lab4p1_serial: $(PART1_SERIAL_FILE)
	gcc $(OPTIMIZATION_LEVEL) $(PART1_SERIAL_FILE) -o $(PART1_SERIAL_OUTPUT)

# Parallel execution. Part 1
lab4p1_parallel: $(PART1_PARALLEL_FILE)
	nvcc -O -o $(PART1_PARALLEL_OUTPUT) $(PART1_PARALLEL_FILE)

# Serial execution. Part 2 
lab4p2_serial: $(PART2_SERIAL_FILE)
	gcc $(OPTIMIZATION_LEVEL) $(PART2_SERIAL_FILE) -o $(PART2_SERIAL_OUTPUT)

# Parallel execution. Part 2
lab4p2_parallel: $(PART2_PARALLEL_FILE)
	nvcc -O -o $(PART2_PARALLEL_OUTPUT) $(PART2_PARALLEL_FILE)

# Parallel execution. Part 3
lab4p3_parallel: $(PART3_PARALLEL_FILE)
	nvcc -O -o $(PART3_PARALLEL_OUTPUT) $(PART3_PARALLEL_FILE)

clean:
	rm -f $(BINARIES) *.out *.o  
