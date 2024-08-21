import os
from natsort import natsorted

# 경로 설정
root_dir = r'C:\Users\DS\Desktop\deeptective\deeptective\audio_driven2'
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')

# 리스트 파일 경로 설정
train_list_path = os.path.join(root_dir, 'train_list.txt')
valid_list_path = os.path.join(root_dir, 'valid_list.txt')

# 라벨 지정 함수
def get_label(file_name):
    if 'train' in file_name.lower():
        return 0
    elif 'test' in file_name.lower():
        return 1
    return -1  # 예상치 못한 경우, 기본 라벨을 설정 (이 경우를 잡기 위해 -1을 반환)

# 파일 리스트 작성 함수
def create_list_file(data_dir, list_file_path):
    with open(list_file_path, 'w') as list_file:
        for root, _, files in natsorted(os.walk(data_dir), key=lambda x: x[0]):
            for file_name in natsorted(files):
                if file_name.endswith(('.jpg', '.jpeg', '.png')):  # 이미지 파일만 포함
                    file_path = os.path.relpath(os.path.join(root, file_name), root_dir)
                    label = get_label(file_name)  # 라벨 지정
                    if label != -1:  # 유효한 라벨만 기록
                        list_file.write(f"{file_path.replace('\\', '/')} {label}\n")

# train_list.txt 생성
create_list_file(train_dir, train_list_path)

# valid_list.txt 생성
create_list_file(val_dir, valid_list_path)

print(f"'{train_list_path}'와 '{valid_list_path}' 파일이 생성되었습니다.")
