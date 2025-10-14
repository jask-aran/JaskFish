CC ?= gcc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic -pthread
LDFLAGS ?= -pthread

TARGET := cpvsengine
SOURCES := cpvsengine.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
