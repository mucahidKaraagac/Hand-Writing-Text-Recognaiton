import torch
import torch.nn as nn
import torch.nn.functional as func
import util
BATCH_SIZE = 256


class LSTMNetwork(nn.Module):
    def __init__(self):
        super(LSTMNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=(2, 1))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=(2, 1))
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=(2, 1))
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2)

        # self.pool_2x1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        # self.pool_4x2 = nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2))
        # self.pool_4x1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.rnn = nn.LSTM(input_size=512, hidden_size=128, batch_first=True, bidirectional=True)
        # self.rnn2 = nn.LSTM(input_size=64*2, hidden_size=32, batch_first=True, bidirectional=True)
        self.output = nn.Linear(in_features=128*2, out_features=80)

    def forward(self, tensor):
        tensor = self.conv1(tensor)
        tensor = func.relu(tensor)
#         tensor = self.pool_2x1(tensor)
#         # print(tensor.shape)
        tensor = self.conv2(tensor)
        tensor = func.relu(tensor)
#         tensor = self.pool_2x1(tensor)
#         # print(tensor.shape)
        tensor = self.conv3(tensor)
        tensor = func.relu(tensor)
#         tensor = self.pool_4x2(tensor)
#         # print(tensor.shape)
        tensor = self.conv4(tensor)
        tensor = func.relu(tensor)
#         tensor = self.pool_4x2(tensor)
#         # print(tensor.shape)
        tensor = self.conv5(tensor)
        tensor = func.relu(tensor)
#         tensor = self.pool_4x2(tensor)
#         # print(tensor.shape)
        tensor = self.conv6(tensor)
        tensor = func.relu(tensor)

        tensor = self.conv7(tensor)
        tensor = func.relu(tensor)

        # tensor = self.conv8(tensor)
        # tensor = func.relu(tensor)

        tensor = torch.squeeze(tensor)
        tensor = tensor.permute(0, 2, 1)
        tensor, _ = self.rnn(tensor)
        # tensor, _ = self.rnn2(tensor)
        tensor = self.output(tensor)
        tensor = func.log_softmax(tensor, dim=2)
        return tensor
#
# model = LSTMNetwork()
# t = torch.zeros((256, 1, 64, 256))
# out = model(t)
# print(out.shape)
#
device = torch.device('cuda')
batch_images = torch.load("C:/Users/Muco/Desktop/ders/401/tensor_file/Test_images_2.pt")#20480 Train #3072 Test #test2 1024
texts = torch.load("C:/Users/Muco/Desktop/ders/401/tensor_file/Test_labels_2.pt")
texts_lengths = torch.load("C:/Users/Muco/Desktop/ders/401/tensor_file/Test_lengths_2.pt")
train_Loader = torch.utils.data.DataLoader(dataset=batch_images, batch_size=BATCH_SIZE, shuffle=False)
model_Load = torch.load("C:/Users/Muco/Desktop/ders/401/model_LSTM_weiS_1_2.pth")
optimizer_Load = torch.load("C:/Users/Muco/Desktop/ders/401/model_LSTM_optiS_1_2.pth")

# print(texts[1])
# print(texts_lengths)
model = LSTMNetwork()
model.load_state_dict(model_Load)
model.train()
# output = model(tensor)
# print(batch_images[0].shape)
model = model.to(device)
ctc_loss = torch.nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 80], 10)
optimizer.load_state_dict(optimizer_Load)
#lr=0.00003
for epoch in range(100):
    curr = 0

    for batch_images in train_Loader:
        label_length = []
        label = []
        x = []
        for sizeL in range(BATCH_SIZE):
            x.append(texts[sizeL + curr])
            for inner in range(len(texts[sizeL + curr])):
                label.append(texts[(sizeL + curr)][inner])
            label_length.append(texts_lengths[(sizeL + curr)])
        curr += BATCH_SIZE
        train_label = torch.tensor(label)
        train_length = torch.IntTensor(label_length)
        train_length = train_length.squeeze().to(device)
        output_length = torch.zeros((BATCH_SIZE,)).fill_(125).type(torch.IntTensor).to(device)
        batch_images = batch_images.to(device)
        train_label = train_label.to(device)
        output = model(batch_images)
        ctc_in = output.permute(1, 0, 2)
        loss = ctc_loss(ctc_in, train_label, output_length.cpu(), train_length.cpu())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()
        util.show_result(output, x)
        print("\n")
        print("Batch Index: " + str(curr / BATCH_SIZE) + " epoch : " + str(epoch))
        print("Batch Loss: ", loss)

torch.save(model.state_dict(), "C:/Users/Muco/Desktop/ders/401/model_LSTM_weiS_1_2.pth")
torch.save(optimizer.state_dict(), "C:/Users/Muco/Desktop/ders/401/model_LSTM_optiS_1_2.pth")
