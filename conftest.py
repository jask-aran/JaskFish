import sys
import os

# Get the absolute path of the 'src' directory
src_dir = os.path.abspath(os.path.dirname(__file__))

# Add the src directory to sys.path if it's not already there
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)