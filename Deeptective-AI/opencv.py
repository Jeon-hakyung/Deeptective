
import cv2
import os

# 테스트 데이터 디렉토리 생성
test_dir = "D:/labeling/test1/"
os.makedirs(test_dir, exist_ok=True)

# 훈련 데이터 디렉토리 생성
train_dir = "D:/labeling/train1/"
os.makedirs(train_dir, exist_ok=True)

count = 1
test = 1
train = 1

for i in range(1,4):
    vidcap = cv2.VideoCapture("C:\\Users\\DS\\Downloads\\1.mp4")

    print("%d번째 영상 시작" %i)

    while(vidcap.isOpened()):
        ret, image = vidcap.read()

        if(ret==False):
            print("%d번째 영상 끝" %i)
            print("train=%d" %train)
            print("test=%d" %test)
            break

        if(int(vidcap.get(1)) % 5 == 0):

            num=count % 10

            if num==0 or num==5: # 20%는 test data, 80%는 train data
                cv2.imwrite("D:/labeling/test1/test%s.jpg" %str(test).zfill(5), image)
                test +=1
            else:
                cv2.imwrite("D:/labeling/train1/train%s.jpg" %str(train).zfill(5), image)
                train += 1

            count += 1

    vidcap.release()

print("devide end")