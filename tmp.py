import os
current_dir = os.path.dirname(__file__)
print(current_dir)
print(os.path.abspath(os.path.join(current_dir, '..')))