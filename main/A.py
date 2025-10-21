import os

folder_path = "/home/lwc/data/CID"

files = os.listdir(folder_path)
files.sort()

for i, file in enumerate(files):
    _, ext = os.path.splitext(file)
    new_name = "CID_" + str(i+1) + ext
    os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
    print(os.path.join(folder_path, file), os.path.join(folder_path, new_name))