import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import time as time


# conv
in_Chanel = 1
out_Chanel = 16
kernel_Size = 3
kernel_Size1 = 1
# out_Conv
out_conv_in = 2
out_conv_out = 1
# extender
Chanel_mul = 4
# upSample
# up_Sample = 2
# up_Sample_Mode = "bilinear"
# max pool
pool_Kernel = 2
pool_Stride = 2
pool_Kernel_4 = 4
pool_Stride_4 = 4
# upSamp = nn.Upsample(scale_factor=up_Sample, mode=up_Sample_Mode, align_corners=True)
pad_1 = nn.ConstantPad2d(1, 1)
pad_2 = nn.ConstantPad2d(2, 1)
pool = nn.MaxPool2d(pool_Kernel, pool_Stride, return_indices=True)
unPool = nn.MaxUnpool2d(pool_Kernel, pool_Stride)
unPool_2 = nn.MaxUnpool2d(pool_Kernel_4, pool_Stride_4)
pool_2 = nn.MaxPool2d(pool_Kernel_4, pool_Stride_4, return_indices=True)


def correct_pixel_2(res, labels):
    return (res.eq(labels).sum().item() * 100) / (1024 * 1024 * 8)


def correct_pixel(res, labels):
    return (res.eq(labels).sum().item() * 100) / (1024 * 1024 * 4)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.enConv1 = nn.Conv2d(in_Chanel, out_Chanel, kernel_Size)

        self.enConv2 = nn.Conv2d(in_Chanel * out_Chanel, out_Chanel * Chanel_mul, kernel_Size)

        self.enConv3 = nn.Conv2d(in_Chanel * out_Chanel * Chanel_mul, out_Chanel * Chanel_mul * Chanel_mul, kernel_Size)

        self.enConv4 = nn.Conv2d(in_Chanel * out_Chanel * Chanel_mul * Chanel_mul,
                                 out_Chanel * Chanel_mul * Chanel_mul * Chanel_mul, kernel_Size)

        self.midConv1 = nn.Conv2d(in_Chanel * out_Chanel * Chanel_mul * Chanel_mul * Chanel_mul,
                                  out_Chanel * Chanel_mul * Chanel_mul * Chanel_mul, kernel_Size)

        self.BnNorm = nn.BatchNorm2d(1024)

        self.midConv2 = nn.Conv2d(in_Chanel * out_Chanel * Chanel_mul * Chanel_mul * Chanel_mul,
                                  out_Chanel * Chanel_mul * Chanel_mul * Chanel_mul, kernel_Size)

        self.decConv2 = nn.Conv2d(out_Chanel * Chanel_mul * Chanel_mul * Chanel_mul,
                                  in_Chanel * out_Chanel * Chanel_mul * Chanel_mul, kernel_Size)

        self.decConv3 = nn.Conv2d(out_Chanel * Chanel_mul * Chanel_mul, in_Chanel * out_Chanel * Chanel_mul,
                                  kernel_Size)

        self.decConv4 = nn.Conv2d(out_Chanel * Chanel_mul, in_Chanel * out_Chanel, kernel_Size)

        self.decConv5 = nn.Conv2d(out_Chanel, in_Chanel * 2, kernel_Size)

        self.out = nn.Conv2d(out_conv_in, out_conv_out, kernel_Size1)

    def forward(self, tensor):
        tensor = self.enConv1(tensor)
        tensor = pad_1(tensor)
        tensor = func.relu(tensor)
        tensor, first_max_pool = pool(tensor)

        tensor = self.enConv2(tensor)
        tensor = pad_1(tensor)
        tensor = func.relu(tensor)
        tensor, second_max_pool = pool(tensor)

        tensor = self.enConv3(tensor)
        tensor = pad_1(tensor)
        tensor = func.relu(tensor)
        tensor, third_max_pool = pool(tensor)

        tensor = self.enConv4(tensor)
        tensor = pad_1(tensor)
        tensor = func.relu(tensor)
        tensor, forth_max_pool = pool(tensor)

        tensor = self.midConv1(tensor)
        tensor = pad_1(tensor)
        tensor = func.relu(tensor)

        tensor = self.BnNorm(tensor)

        tensor = self.midConv2(tensor)
        tensor = pad_1(tensor)
        tensor = func.relu(tensor)

        tensor = unPool(tensor, forth_max_pool)
        tensor = self.decConv2(tensor)
        tensor = pad_1(tensor)
        tensor = func.relu(tensor)

        tensor = unPool(tensor, third_max_pool)
        tensor = self.decConv3(tensor)
        tensor = pad_1(tensor)
        tensor = func.relu(tensor)

        tensor = unPool(tensor, second_max_pool)
        tensor = self.decConv4(tensor)
        tensor = pad_1(tensor)
        tensor = func.relu(tensor)

        tensor = unPool(tensor, first_max_pool)
        tensor = self.decConv5(tensor)
        tensor = pad_1(tensor)
        tensor = func.relu(tensor)

        tensor = self.out(tensor)
        return tensor


device = torch.device('cuda')
images = torch.load("D:/tensorFile/imagesTrain1.pt")
labels = torch.load("D:/tensorFile/labelsTrain1.pt")
model_Load = torch.load("C:/Users/Muco/Desktop/ders/401/model/model_weiS.pth")
optimizer_Load = torch.load("C:/Users/Muco/Desktop/ders/401/model/model_optiS.pth")
train_Loader = torch.utils.data.DataLoader(dataset=images, batch_size=4, shuffle=False)

model = NeuralNetwork()
model.load_state_dict(model_Load)
model.train()
# model.eval()
model = model.to(device)
print(model)
l_R = 3e-3
MaxRate = 0
start = time.time()
temp = 0
mod = 9

optimizer = optim.Adam(model.parameters(), lr=l_R, weight_decay=1e-3)
optimizer.load_state_dict(optimizer_Load)
criterion = nn.BCEWithLogitsLoss()
for epoch in range(100):
    curr = 0
    for images in train_Loader:
        trainL = []
        for sizeL in range(4):
            trainL.append(labels[(sizeL + curr)])
            trainLT = torch.stack(trainL)
        curr += 4
        images = images.to(device)
        trainLT = trainLT.to(device)
        res = model(images)
        loss = criterion(res, trainLT)
        loss.backward()
        optimizer.step()
        resR = res.round()
        TempRateR = correct_pixel(resR, trainLT)
        TempRate = correct_pixel(res, trainLT)
        # if TempRate > MaxRate:
        #     MaxRate = TempRate
        #     bestM = model.state_dict()
        #     bestRes = res
        #     bestM_OP = optimizer.state_dict()
        print("=================", "Epoch =", epoch, "------------", "Current =", curr / 4, "=================")
        print("loss", loss.item())
        print("Learning_Rate = ", l_R)
        print("Current Correct Rate = %", TempRate, " Current round Correct Rate = %", TempRateR)
    if epoch % mod == 0:
        from PIL import Image
        import torchvision.transforms as transforms

        test = images[0][0].cpu()
        test2 = res[0][0].cpu().round()
        test3 = res[0][0].cpu()
        test4 = trainLT[0][0].cpu()
        trans = transforms.ToPILImage()
        im4 = trans(test4)
        im3 = trans(test3)
        im2 = trans(test2)
        im = trans(test)
        Image._show(im2)
        Image._show(im3)
        Image._show(im4)
        Image._show(im)
    epoch += 1
torch.save(model.state_dict(), "C:/Users/Muco/Desktop/ders/401/model_weiS.pth")
torch.save(optimizer.state_dict(), "C:/Users/Muco/Desktop/ders/401/model_optiS.pth")
# torch.save(bestM, "C:/Users/Muco/Desktop/ders/401/model_weiS_Best.pth")
# torch.save(bestM_OP, "C:/Users/Muco/Desktop/ders/401/model_optiS_best.pth")
print("Time Spend =", time.time() - start)
