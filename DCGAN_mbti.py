from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import pickle
import pandas
import matplotlib.pyplot as plt

des_dir = "./processing_data/" # 저장할 폴더 위치

imageSize = 64    
batchSize = 64 

dataset = dset.ImageFolder(root=des_dir,
                           transform=transforms.Compose([ # 전처리 작업
                               transforms.Scale(imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])) 
                           # 이미지의 경우 픽셀 값 하나는 0~255의 값
                           # ToTensor()로 타입 변경시 0~1 사이의 갑스올 바뀜
                           # Normalize : -1 ~ 1 값으로 정규화 시킴 

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size= batchSize,
                                         shuffle=True)

# DCGAN01
nz     = 100      # 잠재 벡터의 길이(크기)
nc     = 3        # number of channel - RGB
ngf    = 64       # Generator를 거치는 피쳐 맵의 크기
ndf    = 64       #  Discriminator를 거치는 피쳐 맵의 크기
niter  = 1000      # total number of epoch
lr     = 0.0001   # learning rate
beta1  = 0.5      # hyper parameter of Adam optimizer
ngpu   = 1        # 가능한 gpu의 수. 이 값이 0이라면 CPU 모드에서 작동할 것이다. 0보다 더 큰 수일 경우, 숫자만큼의 GPU에서 작동할 것이다.







imageSize = 32    
batchSize = 32    

outf = "./output_DCGAN03/"

def weights_init(m): # 모든 모델의 weight는 랜덤하게 초기화 되어야 한다.
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)    # 평균 0, 표준편차 0.02
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)    # 평균 1.0, 표준편차 0.02
        m.bias.data.fill_(0)


class _netG(nn.Module): # Genneratior
    # 클래스 형태의 모델은 항상 nn.Module을 상속받아야 한다.
    # discriminator, D는 이미지를 인풋으로, 인풋 이미지가 진짜일 확률을 아웃풋으로 하는 이진 분류(binary classification) 네트워크이다.
    def __init__(self, ngpu):
        super(_netG, self).__init__() # nn.Module.__init__() 을 실행
        self.ngpu = ngpu
        self.main = nn.Sequential(

            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), 
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

            #anh함수는 아래의 그림과 같이 입력값의 총합을 -1에서 1사이의 값으로 변환해 주며, 원점 중심(zero-centered)이기 때문에, 시그모이드와 달리 편향 이동이 일어나지 않는다. 하지만, tanh함수 또한 입력의 절대값이 클 경우 -1이나 1로 수렴하게 되므로 그래디언트를 소멸시켜 버리는 문제가 있다.



            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        # 모델이 학습데이터를 입력받아서 순전파를 진행시키는 함수
        # 함수 이름이 반드시 forward여야 한다.
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module): # Discriminator
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()

            # state size. 1
        )

    def forward(self, input): # 왜 한번 더 들어갔는지? 
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1) # view : reshape, squeeze : 자동으로 원하는 차원이나 전체에서 하나만 남아있는 경우 없애주는 역할


netG = _netG(ngpu)
netG.apply(weights_init) # 생성자 가중치 적용
print(netG)

netD = _netD(ngpu)
netD.apply(weights_init) # 판별자 가중치 적용 
print(netD)


criterion = nn.BCELoss() 
# class가 2개인 binary case인 경우 사용 
# GAN의 판별자 D는 real or fake를 판단하기 때문에. real일 때 y = 1, fake일 때 y = 0
input = torch.FloatTensor(batchSize, 3, imageSize,imageSize)
noise = torch.FloatTensor(batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)

label = torch.FloatTensor(batchSize)
real_label = 1
fake_label = 0

netD.cuda()
netG.cuda()
criterion.cuda()
input, label = input.cuda(), label.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()


fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# betas는 기울기와 그 제곱의 실행 평균을 계산하는 데 사용되는 계수
result_dict = {}
loss_D,loss_G,score_D,score_G1,score_G2 = [],[],[],[],[]


for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        # train with real
        netD.zero_grad()
        # 모든 매개변수의 gradient 값을 초기화 시킨다. 
        # Gradient Vanishing 문제때문 -> 깊이가 깊은 심층신경망에서는 역전파 알고리즘이 입력층으로 전달됨에 따라 그래디언트가 점점 작아져 결국 가중치 매개변수가 업데이트 되지 않는 경우가 발생
        # 시그모이드 함수 문제점1 : 입력의 절대값이 크게 되면 0이나 1로 수렴하게 되는데, 이러한 뉴런들은 그래디언트를 소멸(kill) 시켜 버린다
        # 시그모이드 함수 문제점2 : 원점 중심이 아니다(Not zero-centered), 따라서, 평균이 ​이 아니라 ​이며, 시그모이드 함수는 항상 양수를 출력하기 때문에 출력의 가중치 합이 입력의 가중치 합보다 커질 가능성이 높다. 이것을 편향 이동(bias shift)이라 하며, 이러한 이유로 각 레이어를 지날 때마다 분산이 계속 커져 가장 높은 레이어에서는 활성화 함수의 출력이 0이나 1로 수렴하게 되어 그래디언트 소실 문제가 일어나게 된다.




        real_cpu, _ = data
        batch_size = real_cpu.size(0)

        real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)

        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)

        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()



    vutils.save_image(real_cpu,
            '%s/real_samples.png' % outf,
            normalize=True)
    fake = netG(fixed_noise)
    vutils.save_image(fake.data,
            '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
            normalize=True)
    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
      % (epoch, niter, i, len(dataloader),
         errD.data, errG.data, D_x, D_G_z1, D_G_z2))
    loss_D.append(errD.data)
    loss_G.append(errG.data)
    score_D.append(D_x)
    score_G1.append(D_G_z1)
    score_G2.append(D_G_z2)
    result_dict = {"loss_D":loss_D,"loss_G":loss_G,"score_D":score_D,"score_G1":score_G1,"score_G2":score_G2}
    pickle.dump(result_dict,open("./{}/result_dict.p".format(outf),"wb"))

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG.pth' % (outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (outf))
