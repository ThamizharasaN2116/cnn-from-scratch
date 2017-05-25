# cnn-from-scratch

CNN from scratch

## Make sure the following directory structure is in place

```
project
│   README.md
│   *.ipynb   
│
└───images
│   │   folders for specific category
│   │   eg. apple
│   │   eg. banana
|
└───converted (Keep an empty folder)
```

The PreprocessData.ipynb will preprocess the images from from images driectory (such as resizing, flipping) and place them in converted folder. We will be converting an image to 4 times of the same image. If the total images in images directory is 5K the resulting image will be 20K with the help of data augmentation technique

## PreprocessingData.ipynb

This notebook will resize images to 69x69x3, perform data augmentation, convert the images into numpy array, perform shuffling and write into data.pkl file with help of pickle

