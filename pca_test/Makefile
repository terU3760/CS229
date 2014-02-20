CC=gcc
CFLAGS=-Wall -O2 -c
LDFLAGS=-lm -lgsl -lgslcblas
SRC=pca_test.c 
OBJ=$(SRC:.c=.o)
EXECUTABLE=pca_test
.PHONY: clean

all: $(SRC) $(EXECUTABLE)

$(EXECUTABLE): $(OBJ) 
	$(CC) $(LDFLAGS) $(OBJ) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o $(EXECUTABLE)
