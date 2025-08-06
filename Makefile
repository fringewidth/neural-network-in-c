CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -I.

SRC = main.c memman.c mtrx.c nn.c
OBJ = $(SRC:.c=.o)

all: run

run: $(OBJ)
	@$(CC) $(OBJ) -o temp_exec
	@./temp_exec
	@rm temp_exec

%.o: %.c
	@$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) temp_exec
