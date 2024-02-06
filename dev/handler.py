import subprocess 
import time
import sys
import threading

def send_command(proc, command):
    print(f"Sending command: {command}")
    proc.stdin.write(command + "\n")
    proc.stdin.flush()

def read_response(proc):
    while True:
        output = proc.stdout.readline()
        if output == '' and proc.poll() is not None:
            break
        if output:
            print(f"Received: {output.strip()}")
            
def spam(proc):
    while True:
        send_command(proc, 'isready')
        time.sleep(0.5)  # Wait for 1 second before sending the next command


# Start engine.py as a subprocess
process = subprocess.Popen(["python3", "engine.py"], 
                           stdin=subprocess.PIPE, 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, 
                           text=True, 
                           bufsize=1,  # Line buffered
                           universal_newlines=True)

# Use a separate thread to read responses
threading.Thread(target=read_response, args=(process,), daemon=True).start()
# threading.Thread(target=spam, args=(process,), daemon=True).start()


print("Enter chess engine commands (type 'exit' to quit):")

# Main loop to read commands from handler script's stdin
while True:
    command = input()
    if command == "quit":
        send_command(process, command)  # Ensure engine.py exits cleanly
        time.sleep(0.5)
        break
    else:
        send_command(process, command)

# Terminate the subprocess
process.terminate()