NAME = ppsearch
OBJS = ppsearch.o main.o # cpudetect.o

CFLAGS = -O3 -march=native
LDFLAGS = $(CFLAGS) -lm

CC = gcc

all: $(NAME)

$(NAME): $(OBJS)
	${CC} $(LDFLAGS) -o $@ $(OBJS)

clean:
	rm $(NAME) $(OBJS)

main.o: src/main.c Makefile # cpudetect.h
	$(CC) $(CFLAGS) -c $<

ppsearch.o: src/ppsearch.c Makefile
	$(CC) $(CFLAGS) -c $<
	
