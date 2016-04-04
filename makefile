# Common arguments called with all compilation statements
COMMON_ARGUMENTS          := -Wall -Werror -g #-pg
OPTIMIZATION_LEVEL        := -O0
PART1_SERIAL_FILE         := stewart_nate_lab4p1.c
PART1_SERIAL_OUTPUT       := lab4p1_serial
PART2_SERIAL_FILE         := stewart_nate_lab4p2.c
PART2_SERIAL_OUTPUT       := lab4p2_serial
BINARIES := $(PART1_SERIAL_OUTPUT) $(PART2_SERIAL_OUTPUT)

all: lab4p1_serial lab4p2_serial
# Serial execution. Part 1
lab4p1_serial: $(PART1_SERIAL_FILE)
	gcc $(OPTIMIZATION_LEVEL) $(PART1_SERIAL_FILE) -o $(PART1_SERIAL_OUTPUT)

# Serial execution. Part 2 
lab4p2_serial: $(PART2_SERIAL_FILE)
	gcc $(OPTIMIZATION_LEVEL) $(PART2_SERIAL_FILE) -o $(PART2_SERIAL_OUTPUT)

clean:
	rm -f $(BINARIES) *.out *.o  
