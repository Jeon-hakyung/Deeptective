import os

def generate_list_file(root_dir, list_file, folder_labels):
    with open(list_file, 'w') as f:
        for folder, label in folder_labels.items():
            folder_path = os.path.join(root_dir, folder)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(root, file)
                        file_path = file_path.replace('\\', '/')
                        f.write(f"{file_path} {label}\n")

train_root = "C:/Users/DS/PycharmProjects/pythonProject/hanium/train"
val_root = "C:/Users/DS/PycharmProjects/pythonProject/hanium/val"

train_list_file = "C:/Users/DS/PycharmProjects/pythonProject/hanium/train_list.txt"
val_list_file = "C:/Users/DS/PycharmProjects/pythonProject/hanium/valid_list.txt"

# 각 폴더에 대해 레이블을 수동으로 설정합니다.
train_folder_labels = {
    'train1': 0,
    'train2': 1,
    'train3': 2,
    'train4': 3,
    'train5': 4,
    'train6': 5
}

val_folder_labels = {
    'test1': 0,
    'test2': 1,
    'test3': 2,
    'test4': 3,
    'test5': 4,
    'test6': 5
}

generate_list_file(train_root, train_list_file, train_folder_labels)
generate_list_file(val_root, val_list_file, val_folder_labels)
