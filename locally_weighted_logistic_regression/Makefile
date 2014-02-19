CC=gcc
CFLAGS=-Wall -O2 -c #-g -c
LDFLAGS=-lm -ljpeg
SRC=lwlr.c 
OBJ=$(SRC:.c=.o)
EXECUTABLE=lwlr
.PHONY: clean

all: $(SRC) $(EXECUTABLE)

$(EXECUTABLE): $(OBJ) 
	$(CC) $(LDFLAGS) $(OBJ) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o $(EXECUTABLE)
