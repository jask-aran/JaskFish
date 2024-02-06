import subprocess
import queue

from utils import color_text

def engine_output_processor(output_queue, proc):
    while True:
        output = proc.stdout.readline()
        if output == '' and proc.poll() is not None:
            break
        if output:
            output_queue.put(output.strip())
    
def send_command(proc, command):
    print(color_text('Sending   ', '32') + command)
    proc.stdin.write(command + "\n")
    proc.stdin.flush()
    


