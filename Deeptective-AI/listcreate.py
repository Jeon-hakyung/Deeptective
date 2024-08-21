import os

import transforms

# 절대 경로를 사용하여 루트 디렉토리 설정
root_dir = "C:/Users/DS/PycharmProjects/pythonProject/training_data"
train_list_path = os.path.join(root_dir, "train_list_1st.txt")
valid_list_path = os.path.join(root_dir, "test_list_1st.txt")

# 디렉토리 존재 여부 확인 및 생성
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# 훈련 데이터 리스트 생성
train_list = []
for folder_name in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 확장자에 맞게 수정
                file_path = os.path.join(folder_path, file_name)
                train_list.append(file_path)

# 검증 데이터 리스트 생성 (여기서는 임의로 20%를 검증 데이터로 사용)
split_idx = int(len(train_list) * 0.8)
valid_list = train_list[split_idx:]
train_list = train_list[:split_idx]

# 리스트 파일로 저장
with open(train_list_path, 'w') as f:
    for item in train_list:
        f.write("%s\n" % item)

with open(valid_list_path, 'w') as f:
    for item in valid_list:
        f.write("%s\n" % item)

