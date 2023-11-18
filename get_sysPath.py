import os
import sys

# Get the path to the current script
script_path = os.path.dirname(os.path.realpath(__file__))
Add all subfolders to sys.path
for root, dirs, files in os.walk(script_path):
    for directory in dirs:
        subfolder_path = os.path.abspath(os.path.join(root, directory))
        sys.path.append(subfolder_path)
print(sys.path)


