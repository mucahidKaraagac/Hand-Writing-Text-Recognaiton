import string

import cv2

import os

from PIL import Image, ImageEnhance


import os


def get_words(): #this method reads and returns the words from a text file
    names_file = open("D://HTR(D)//words.txt", "r")

    new_text_file = open("D://HTR(D)//new_words.txt", "w+")

    for entry in names_file:
        #print(entry.split())

        entry_elems = entry.split()

        #print(entry_elems[-1])

        new_text_file.write(entry_elems[0]+" "+entry_elems[-1]+"\n")

    new_text_file.close()




path = "D:\\"

# Check current working directory.
retval = os.getcwd()

#print("Current working directory %s" % retval)

# Now change the directory
os.chdir( path )

# Check current working directory.
retval = os.getcwd()

#print("Directory changed successfully %s" % retval)

import numpy as np


original1024path = "C://Users//yusuf//Desktop//HTR(C)//Data_Related//square_1024_template.png"
original1024_image = Image.open(original1024path)

DISTANCE = 50 #There exist 20 pixels empty space between words.
def getData():
    basepath = 'D:/HTR(D)/data/fki_computer_vision/words_fki/words'
    count = 0
    entries = np.zeros()
    for entry in os.listdir(basepath):
        print(entry)
        segments = []
        for segment in os.listdir(basepath + "/" + entry):
            images = []
            for image in os.listdir(basepath + "/" + entry + "/" + segment):
                if count > 100 and count < 102:
                    current_img = Image.open(basepath + "/" + entry + "/" + segment + "/" +image)
                    current_img.show()
                    print(current_img.size) #gets the shape of the image
                    count = count + 1
                images.append(image)
            segments.append(images)
            count = count + 1
        entries.append(segments)
        print()

DISTANCE = 50
class Base1024:
    image = None
    path = ""
    rowHeights = []
    rowCount = 1
    lastX = 0
    lastY = DISTANCE
    name = ""
    isFilled = False
    y_val = 0;

    def __init__(self, name, image, path):
        self.name = name
        self.image = image
        self.path = path

    def calculateCoordinates(self, imageWidth, imageHeight):
        X = 0; Y = 0
        if (self.lastX + imageWidth < 1024 - DISTANCE*2): #same row coordinates, x changes, y stays the same, row has one new element
            #print("same row", end="    ")
            X = self.lastX + DISTANCE
            self.lastX = X + imageWidth


            Y = self.lastY
            #lastY stays the same
            self.rowHeights.append(imageHeight)
            self.isFilled = False

        elif (self.lastY + imageHeight*4 + 206 < 256): # add a row, x is resetted, y changes, row is emptied
            #print("lastY :" , self.lastY   ,  "           im height : ", imageHeight, "       distance : ", DISTANCE)
            #print("calc : ", self.lastY + imageHeight + DISTANCE)
            self.rowCount = self.rowCount + 1
            #print( self.rowHeights)
            #print("new row", end = "    ")

            X = DISTANCE #reset the X value
            self.lastX = X + imageWidth

            self.emptyRowHeights() #new row added old row value deleted
            self.rowHeights.append(imageHeight)
            self.isFilled = False

            Y = self.lastY + (150) + max(self.rowHeights)
            self.lastY = Y

            #self.y_val = max(self.rowHeights) + self.y_val
            #Y = Y + (DISTANCE) + self.y_val
            #self.lastY = Y + imageHeight

        else: #this base1024 is full
            self.isFilled = True

        return (self.isFilled, X, Y)

    def pasteImage(self, newImage):
        #newImage.show()
        (isFilled, X, Y) = self.calculateCoordinates(newImage.size[0], newImage.size[1])
        if(not isFilled):
            #print("x: ", X, "                       Y: ", Y)
            self.image.paste(newImage, (X,Y))
            self.image.save(self.path);
        #else do nothing this base is over.

        #return false if the the current base is filled, return true if not
        return (not isFilled, X, Y)

    def pasteWordBox(self, X, Y, imageWidth, imageHeight):
        import cv2
        self.image = cv2.rectangle(self.image, (X, Y), ((X + imageWidth), (Y + imageHeight)), (0, 0, 0), -1)
        cv2.imwrite(self.path, self.image)


        return self.image #is it necessary?

    def emptyRowHeights(self):
        self.rowHeights.clear()


def retrieve_line(search_name):
    word_names = open("D://HTR(D)//new_words.txt", "r")
    result = -1
    # this method find the first occurence of given string and returns the line it is in
    for line in word_names:
        #print("search_name is ", search_name)
        #print("line is ", line)

        if search_name in line:
            result = line.split()[-1]
            break
    return result


def generateData():


    # open the image labels text file


    wordPath = 'D:/HTR(D)/data/fki_computer_vision/words_fki/words' #it is used to get the words one by one
    count = 0

    generatedBaseImagePath = "D://HTR(D)//PreprocessData//base//";
    generatedBaseMapImagePath = "D://HTR(D)//PreprocessData//baseMap//";

    basePath = "base0"

    entries = []
    for entry in os.listdir(wordPath):
        #print(entry)
        segments = []

        generatedBaseImagePathVar = generatedBaseImagePath + entry
        generatedBaseMapImagePathVar = generatedBaseMapImagePath + entry

        os.mkdir(generatedBaseImagePathVar, 0o755)
        os.mkdir(generatedBaseMapImagePathVar, 0o755)

        prev_segment_name = "-1"
        for segment in os.listdir(wordPath + "/" + entry):
            images = []

            #define the specific paths of images according to segment name


            #os.mkdir(generatedBaseImagePathVar+"//"+segment, 0o755)
            #os.mkdir(generatedBaseMapImagePathVar+"//"+segment, 0o755)

            #generatedBaseImage_segmented_PathVar = generatedBaseImagePath + entry + "//" + segment
            #generatedBaseMapImage_segmented_PathVar = generatedBaseMapImagePath + entry + "//" + segment

            (basePath, base1024Image, base1024MapImage) = createBlank1024(basePath, generatedBaseImagePathVar, generatedBaseMapImagePathVar)
            current_base_text = open("D://HTR(D)//PreprocessData//base//" + entry + "//" + basePath + ".txt", "w+")

            for image in os.listdir(wordPath + "/" + entry + "/" + segment):
                try:
                    #print("wordPath : ", wordPath + "/" + entry + "/" + segment + "/" + image)
                    newImage = Image.open(wordPath + "/" + entry + "/" + segment + "/" + image)

                    #print("image filename is: ", image[:-4])
                    #print(newImage.filename)

                    enhancer = ImageEnhance.Contrast(newImage)
                    newImage = enhancer.enhance(10)

                    (isPasted, pasteX, pasteY) = base1024Image.pasteImage(newImage)
                    #print("\n", base1024Image.name, "\n")
                    if(not isPasted):
                        #print("\n\n*********************************************************NEW**************************************************\n")
                        #add the current one to 1024s list
                        (basePath, base1024Image, base1024MapImage) = createBlank1024(basePath, generatedBaseImagePathVar, generatedBaseMapImagePathVar)
                        current_base_text.close()
                        current_base_text = open("D://HTR(D)//PreprocessData//base//" + entry + "//" + basePath + ".txt", "w+")
                        #print("\n", base1024Image.name, "\n")
                        (isPasted, pasteX, pasteY) = base1024Image.pasteImage(newImage)

                        if(retrieve_line(image[:-4]) == -1):
                            print("-1")
                        current_base_text.write(str(retrieve_line( image[:-4])) + "\n")
                        base1024MapImage.pasteWordBox(pasteX, pasteY, newImage.size[0], newImage.size[1])
                    else:
                        #print("retrieve: ", image[:-4])

                        if(retrieve_line(image[:-4]) == -1):
                            print("-1")
                        current_base_text.write(str(retrieve_line(image[:-4])) + "\n")
                        #print("write is done")
                        base1024MapImage.pasteWordBox(pasteX, pasteY, newImage.size[0], newImage.size[1])
                except:
                    print("An error occured.")
            current_base_text.close()

            #images.append(image)
            #image for loops

        #segment for loop
        isCurrentBlankFilled = True
        segments.append(images)
        count = count + 1
        #entry for loop
        entries.append(segments)




def createBlank1024(lastBlank1024Path, pathBase, pathMap):
    #returns the path of the newly created blank 1024 png file and the base image itself
    import shutil;

    baseNumber = int(lastBlank1024Path[4:]) #get the baseNumber of the previous base1024 image

    variable_path = "base" + str(baseNumber+1) #increment the number by one and set the path of the new blank1024 image

    source_path = "C://Users//yusuf//Desktop//HTR(C)//Data_Related//square_1024_template - Copy.PNG"

    shutil.copyfile(source_path, pathBase + "//" + variable_path+".png")
    shutil.copyfile(source_path, pathMap + "//" + variable_path + "Map" + ".png")
    base1024ImagePNG = Image.open(pathBase + "//" + variable_path+".png")
    base1024MapImagePNG = cv2.imread(pathMap + "//" + variable_path + "Map" + ".png",0) #gets the image using cv2
    base1024Image = Base1024(variable_path, base1024ImagePNG, pathBase + "//" + variable_path+".png")
    base1024MapImage = Base1024(variable_path + "Map", base1024MapImagePNG, pathMap + "//" + variable_path + "Map" + ".png")


    return (variable_path, base1024Image, base1024MapImage)

#prev_var_path = createBlank1024("base0")
#for x in range(0,10):
    #prev_var_path = createBlank1024(prev_var_path)

generateData()
get_words()
