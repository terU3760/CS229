CC=gcc
CFLAGS=-Wall -O2 -c #-g -c
LDFLAGS=-lm
SRC=nn.c 
OBJ=$(SRC:.c=.o)
EXECUTABLE=nn
.PHONY: clean

all: $(SRC) $(EXECUTABLE)

$(EXECUTABLE): $(OBJ) 
	$(CC) $(LDFLAGS) $(OBJ) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o $(EXECUTABLE)
