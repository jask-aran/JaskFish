import sys
import io
import chess
import threading
import queue

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

print('test')