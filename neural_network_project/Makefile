CC = gcc
CFLAGS = -Wall -Wextra -std=c99

OBJ = main.o neural_network.o utils.o

all: neural_network

neural_network: $(OBJ)
	$(CC) $(CFLAGS) -o neural_network $(OBJ)

main.o: main.c neural_network.h
	$(CC) $(CFLAGS) -c main.c

neural_network.o: neural_network.c neural_network.h utils.h
	$(CC) $(CFLAGS) -c neural_network.c

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c

clean:
	rm -f *.o neural_network
