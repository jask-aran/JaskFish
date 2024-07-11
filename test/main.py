import sys
import subprocess
import queue
import threading
import time

from utils import color_text, debug_text

def engine_output_processor(proc):
    while True:
        output = proc.stdout.readline().strip()
        if output == '' and proc.poll() is not None:
            break
        elif output:
            print(color_text('RECIEVED ', '34') + output)

def send_command(proc, command):
    print(color_text('SENDING  ', '32') + command)
    proc.stdin.write(command + "\n")
    proc.stdin.flush()
    
def handle_command_go(proc, fen_string):
    send_command(proc, f"position fen {fen_string}")
    send_command(proc, "go")

def main():
    engine_process = subprocess.Popen(
        ["python3", "engine.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, # Redirect stderr to stdout
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    engine_thread = threading.Thread(target=engine_output_processor, args=(engine_process,), daemon=True)
    engine_thread.start()
    
    time.sleep(0.5)
    while True:
        command = input("Enter command: ")
        if command.strip().lower() == 'quit':
            send_command(engine_process, 'quit')
            print(debug_text("Quitting..."))
            # Wait for the engine process to finish. If it doesn't terminate within 5 seconds, the function will raise a TimeoutExpired exception.
            engine_process.wait(timeout=5)
            break
        elif command:
            send_command(engine_process, command)
            time.sleep(0.01)
            
            


if __name__ == "__main__":
    main()