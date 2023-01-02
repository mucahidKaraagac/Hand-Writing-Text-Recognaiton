# HANDWRITEND TEXT READER 
## _From The Scratch CUSTOM DEEPLEARNING BASE OCR_

Handwritten Text Reader is a solution to convert the non-digital from the handwritten words to digital format. Conversation process has 2 phases. These phases are segmantion of the text area and recognaiton of the what it is.

## Features

- Precise segmation with high input resolution  
- Custom Neural Network design for the split the task   
- Custom data-generator

## Tech

This network uses some librarys and framework:

- [PyTorch] - An open source machine learning framework that accelerates the path from research prototyping to production deployment.
- [PIL] - provides a real-time optimized Computer Vision library
- [OpenCV] - provides a real-time optimized Computer Vision library
- [Numpy] - Offers comprehensive mathematical functions

## Working Principles

 - Handwritten Text Recognizer (HTR) has 2 differnt type of deeplearning network model. Those are :
    - Convoluotion Neural Network (CNN)  
    - Convoluotion Neural Network with Long Term Short Memory (CNN+LSTM)
 -  First custom cnn is based on autoencoder architecture for the segmation of the text area of the images.
 -  Other custom cnn+lstm model is a hybrid model. 
    -  Cnn part for the exract feauture of the text areas. 
    -  Lstm part of it make meaning full text from the feature vectors which are crated form part of the network
 - Custom data generation scripts help for the ease of use and also future proof for adding new languages.
 


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Numpy]: <https://numpy.org>
   [PyTorch]: <https://pytorch.org>
   [PIL]: <https://python-pillow.org>
   [OpenCV]: <https://opencv.org>
