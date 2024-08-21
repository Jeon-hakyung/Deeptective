import easydict
import os
import sys
from PIL import Image
import tqdm
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

# cuDNN 벤치마크 모드 설정 (필요 없지만 남겨둠)
cudnn.benchmark = True

args = easydict.EasyDict({
    "gpu": 0,  # GPU 설정, 기본적으로 첫 번째 GPU 사용
    "num_workers": 4,  # 일반적으로 GPU에서는 더 높은 값이 적절 (CPU 환경과 다르게 설정)
    "root": "C:/Users/DS/Desktop/deeptective/deeptective/audio_driven2",
    "train_list": "C:/Users/DS/Desktop/deeptective/deeptective/audio_driven2/train_list.txt",
    "valid_list": "C:/Users/DS/Desktop/deeptective/deeptective/audio_driven2/valid_list.txt",
    "learning_rate": 0.001,
    "num_epochs": 4,
    "batch_size": 32,
    "save_fn": "deepfake_c0_xception_tuned.pth.tar",
})

assert os.path.isfile(args.train_list), 'wrong path'
assert os.path.isfile(args.valid_list), 'wrong path'

# GPU가 있으면 GPU로 설정, 없으면 CPU로 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
Author: Andreas Rössler,
Implemented in https://github.com/ondyari/FaceForensics under MIT license
"""


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


## 기존 Xception에 Dropout만 추가
class xception(nn.Module):
    def __init__(self, num_out_classes=2, dropout=0.5):
        super(xception, self).__init__()

        self.model = Xception(num_classes=num_out_classes)
        self.model.last_linear = self.model.fc
        del self.model.fc

        num_ftrs = self.model.last_linear.in_features
        if not dropout:
            self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
        else:
            self.model.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_ftrs, num_out_classes)
            )

    def forward(self, x):
        x = self.model(x)
        return x




### 데이터셋 전처리 밑 로드

xception_default = {
    'train': transforms.Compose([transforms.CenterCrop((299, 299)),
                                 transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.Normalize([0.5]*3, [0.5]*3),
                                 ]),
    'valid': transforms.Compose([transforms.CenterCrop((299, 299)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5]*3, [0.5]*3),
                                 ]),
    'test': transforms.Compose([transforms.CenterCrop((299, 299)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5] * 3, [0.5] * 3),
                                ]),
}

## train, Validate, Dataset 함수

# util

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


## 모델 체크포인트를 저장하고 학습률을 조정하는 유틸리티 함수 정의

# custom dataset
# 이미지 경로와 레이블을 저장하는 'ImageRecord' 클래스를 정의하고 데이터셋을 로드하고 변환하는 'DFDCDataset' 클래스 정의

class ImageRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class DFDCDatatset(data.Dataset):
    def __init__(self, root_path, list_file, transform=None):
        self.root_path = root_path
        self.list_file = list_file
        self.transform = transform

        self._parse_list()

    def _load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def _parse_list(self):
        self.image_list = [ImageRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.image_list[index]
        image_name = os.path.join(self.root_path, record.path)
        image = self._load_image(image_name)

        if self.transform is not None:
            image = self.transform(image)

        return image, record.label

    def __len__(self):
        return len(self.image_list)


##
# train / validate
## train 함수는 모델을 학습시키고, validate 함수는 모델을 검증한다.

def train(train_loader, model, criterion, optimizer, epoch):
    n = 0
    running_loss = 0.0
    running_corrects = 0

    model.train()

    with tqdm.tqdm(train_loader, total=len(train_loader), desc="Train", file=sys.stdout) as iterator:
        for images, target in iterator:
            images, target = images.to(device), target.to(device)

            # Outputs and target shapes and types check
            outputs = model(images)
            print(f"Outputs shape: {outputs.shape}")
            print(f"Target shape: {target.shape}")
            print(f"Target dtype: {target.dtype}")

            if target.dtype != torch.long:
                target = target.long()

            _, pred = torch.max(outputs.data, 1)

            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n += images.size(0)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(pred == target.data)

            epoch_loss = running_loss / float(n)
            epoch_acc = running_corrects / float(n)

            log = 'loss - {:.4f}, acc - {:.3f}'.format(epoch_loss, epoch_acc)
            iterator.set_postfix_str(log)
    scheduler.step()


def validate(test_loader, model, criterion):
    n = 0
    running_loss = 0.0
    running_corrects = 0

    model.eval()

    with tqdm.tqdm(valid_loader, total=len(valid_loader), desc="Valid", file=sys.stdout) as iterator:
        for images, target in iterator:
            images, target = images.to(device), target.to(device)

            with torch.no_grad():
                output = model(images)

            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)

            n += images.size(0)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(pred == target.data)

            epoch_loss = running_loss / float(n)
            epoch_acc = running_corrects / float(n)

            log = 'loss - {:.4f}, acc - {:.3f}'.format(epoch_loss, epoch_acc)
            iterator.set_postfix_str(log)

    return epoch_acc

if __name__ == '__main__':
    # 데이터셋 로드 및 데이터 로더 설정
    train_dataset = DFDCDatatset(root_path=args.root, list_file=args.train_list, transform=xception_default['train'])
    val_dataset = DFDCDatatset(root_path=args.root, list_file=args.valid_list, transform=xception_default['valid'])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 모델 초기화 및 설정
    model = xception(num_out_classes=6, dropout=0.5)
    print("=> creating model '{}'".format('xception'))

    # 모델을 CPU 또는 GPU로 이동
    model = model.to(device)

    # 모델 가중치 파일이 존재하면 로드
    fn = args.save_fn
    
    if os.path.isfile(fn):
        checkpoint = torch.load(fn, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> model weight '{}' is loaded".format(fn))
    else:
        print("=> model weight file not found, training from scratch")

    # 손실 함수와 옵티마이저 및 스케줄러 설정
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 학습 및 평가
    best_acc = 0.0
    for epoch in range(args.num_epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        acc = validate(valid_loader, model, criterion)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=args.save_fn)

        scheduler.step()

    print('Best validation accuracy: {:.2f}%'.format(best_acc))


