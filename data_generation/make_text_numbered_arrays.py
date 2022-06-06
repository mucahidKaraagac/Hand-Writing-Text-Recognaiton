import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import shutil
import cv2


def get_length_array(batch):
    len_array = []
    for list in batch:
        base_length = []
        base_length.append(len(list))
        len_array.append(base_length)

    return len_array


def compare_max_len(current_text_file, current_max):

    if(current_max > 33):
        print("*")

    max_len = current_max

    current_len = 0

    for word in current_text_file:

        for letter in word:

            if letter != "\n" and letter != "-" and letter != "#":

                current_len += 1


    if(current_max < current_len):
        max_len = current_len

    return max_len


def do_padding(batch, batch_max):
    count = 0
    for list in batch:
        #print(len(list))

        while (len(list) < batch_max):
            list.append(74)
        count += 1
    return batch

# 74th element is the blank key
dict = {"":0, "a":1,"b":2, "c":3, "d":4, "e":5, "f":6, "g":7, "h":8, "i":9, "j":10, "k":11, "l":12, "m":13, "n":14, "o":15,
        "p":16, "q":17, "r":18, "s":19, "t":20, "u":21, "v":22,"w":23, "x":24, "y":25, "z":26 ,
        "A":27, "B":28, "C":29, "D":30, "E":31, "F":32, "G":33, "H":34, "I":35, "J":36, "K":37, "L":38, "M":39, "N":40,
        "O":41, "P":42, "Q":43, "R":44, "S":45, "T":46, "U":47, "V":48, "W":49, "X":50, "Y":51, "Z":52,
        ".":53, ",":54, "?":55, "!":56, "'":57, "\"":58, ";":59, ":":60, "(":61, ")":62, "&":63,
        "0":64, "1":65, "2":66, "3":67, "4":68, "5":69, "6":70, "7":71, "8":72, "9":73, " ":74}

text_tensor = []
target_len_tensor = []

start = time.time()
exs = 1536
batch_count = 0
batch_max_len = 0
current_batch = []

all_tensors = []
current_max = 0


for num in range(512):
    num += 1

    current_text_file = open("C:/HTR(D)/512_tensor_ready_txt/4/base" + str(num+exs) + ".txt")

    current_text_file_2 = open("C:/HTR(D)/512_tensor_ready_txt/4/base" + str(num+exs) + ".txt")

    print(current_text_file)

    base_tensor = []

    current_max = compare_max_len(current_text_file, current_max)

    for line in current_text_file_2:
        print(line)
        #initialize a new tensor for the new word on the image
        line1 = line
        word_tensor = []
        for word in line.split(" "):

            for letter in word:

                if letter != "\n" and letter != "-" and letter != "#" and letter != "*":
                    print(dict[letter])
                    base_tensor.append(dict[letter])

            #base_tensor.append(74)

    all_tensors.append(base_tensor)


# lengths

len_arr = get_length_array(all_tensors)

for index in range(512):

    print(str(index)+ "      LEN ALL_TENSORS: " +  str(len(all_tensors[index])) + "  LEN_ARR: " + str(len_arr[index][0]))

    if(len(all_tensors[index]) != len_arr[index][0]):

        print("index is : ", index)

all_tensors = do_padding(all_tensors, current_max)




torch.save(all_tensors, "C:/HTR(D)/tensors/text_4_new.pt")
torch.save(len_arr, "C:/HTR(D)/tensors/text_4_target_original_length_new.pt")

print(time.time() - start)
