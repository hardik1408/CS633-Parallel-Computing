CC = mpicc
CFLAGS = -O3 -lm
TARGET = src

all: $(TARGET)

$(TARGET): src.c
	$(CC) $(CFLAGS) -o $@ $<

clean:	
	rm -f $(TARGET)
