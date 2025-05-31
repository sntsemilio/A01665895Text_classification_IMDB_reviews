import os

def list_all_files(directory='.'):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

if __name__ == "__main__":
    files = list_all_files()
    for file in files:
        print(file)