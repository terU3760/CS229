CC=gcc
CFLAGS=-Wall -O2 -c
LDFLAGS=-lgsl -lgslcblas
SRC=gsl_test.c 
OBJ=$(SRC:.c=.o)
EXECUTABLE=gsl_test
.PHONY: clean

all: $(SRC) $(EXECUTABLE)

$(EXECUTABLE): $(OBJ) 
	$(CC) $(LDFLAGS) $(OBJ) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o $(EXECUTABLE)
