import sys
import io
import chess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

while True:
    line = sys.stdin.readline().strip()
    if line == 'quit':
        break
    else:
        print(line)