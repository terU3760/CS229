NEUROPH_PATH=./neuroph-core-2.7.jar
JC=javac
JFLAGS=-cp $(NEUROPH_PATH)
.SUFFIXES: .java .class
MNIST_NN=mnist_nn.class
OBJECTS=$(SOURCES:.java=.class)
EXECUTABLES=mnist_nn.class

.PHONY: clean

.java.class:
	$(JC) $(JFLAGS) $<

all: $(EXECUTABLES)

mnist_nn: $(MNIST_NN)

clean:
	rm -rf *.png $(EXECUTABLES)
