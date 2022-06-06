import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import shutil
import cv2


start = time.time()
trans = transforms.ToTensor()
transGray = transforms.Grayscale()
images = []
labels = []
exs = 1536
for num in range(512):
    num += 1


    #shutil.copy("D:HTR(D)/512_tensor_ready_img/1/base" + str(num+exs) + ".png",  "D:HTR(D)/512_tensor_ready_version_2/1/base" + str(num+exs) + ".png")
    #image_cv = cv2.imread("D:HTR(D)/512_tensor_ready_img/1/base" + str(num+exs) + ".png")

    image = Image.open("C:/HTR(D)/512_tensor_ready_img/4/base" + str(num+exs) + ".png")

    #cv2.imshow("im", image_cv)
    #k = cv2.waitKey()

    #image = cv2.resize(image, (1025, 256))
    #cv2.imwrite("D:HTR(D)/512_tensor_ready_version_2/1.1/base" + str(num+exs) + ".png", image)

    #image = Image.open("D:HTR(D)/512_tensor_ready_img/1/base" + str(num+exs) + ".png")

    image = transGray(image)
    image = trans(image)

    #image.show()

    #print("base" + str(num+exs))
    images.append(image)

torch.save(images, "C:/HTR(D)/tensors/images_4.pt")
print(time.time() - start)