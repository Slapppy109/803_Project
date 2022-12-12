# import required libraries
import os
import cv2
import numpy as np
import mediapipe as mp
from os import listdir
import random
import glob
import stat

# initialize mediapipe 
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

image_path = 'images'
images = os.listdir(image_path)

# image_index= 0
# bg_image = cv2.imread(image_path+'/'+images[image_index])

bg_image = np.zeros([100,100,3],dtype=np.uint8)
bg_image.fill(255) # or img[:] = 255

# good filepath './asl_alphabet_train/A/A1.jpg'

#todo add parameter
def removeBackground(filepath,):
    # read input image
    img = cv2.imread(filepath)
    print('filepath',filepath)


    # flip the frame to horizontal direction
    # frame = cv2.flip(img, 1)
    # frame = img

    height , width, channel = img.shape
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = selfie_segmentation.process(RGB)
    mask = results.segmentation_mask
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # (thresh, binRed) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)

    condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > 0.1


    bg_image = np.zeros([100,100,3],dtype=np.uint8)
    bg_image.fill(255)
    # resize the background image to the same size of the original frame
    bg_image = cv2.resize(bg_image, (width, height))

    # combine frame and background image using the condition
    output_image = np.where(condition, img, bg_image)

    return output_image, img


#loop through all images
start_path = './asl_alphabet_train_augmented'
# if os.path.isdir(start_path):
#     os.chmod(start_path , stat.S_IWRITE)
#     os.remove(start_path)
if not os.path.isdir(start_path):
    os.makedirs(start_path)

list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T',
'U','V','W','X','Y','Z','nothing','space','del']

for items in list:
    path = os.path.join(start_path, items)
    if not os.path.isdir(path):
        os.mkdir(path)

# get the path/directory
# folder_dir = ["./asl_alphabet_train/A/*.jpg"]
# images = glob.glob(random.choice(folder_dir))
# random_image = random.choice(images)
# print(random_image)

train_dir = './asl_alphabet_train/'
train_folders = os.listdir(train_dir)
for folder in train_folders:
    files = os.listdir(train_dir + folder)
    print('Reading images from ' + train_dir + folder + '/ ...')
    # for file in files[:1000]:
    filepath = [train_dir + folder + '/*.jpg']
    print('first filepath', filepath)
    y = 0 
    while y < 450:#tbd 500
        images = glob.glob(random.choice(filepath))
        # print("images", images)
        random_image = random.choice(images)
        # print('second filepath',random_image)
        # opened_rand = cv2.imread(random_image)
        # cv2.imshow(random_image)

        output_image, img = removeBackground(random_image)
        file_end = folder + str(y) + '.jpg'
        print("file end", file_end)
        new_path = os.path.join(start_path, folder,file_end)
        print('final filepath', new_path)
        #write output image 
        cv2.imwrite(new_path, output_image)
        #also include input image to match
        orig_file_end = folder + str(y) + '_orig.jpg'
        orig_file_path = os.path.join(start_path, folder,orig_file_end)
        print("original filpath", orig_file_path)
        cv2.imwrite(orig_file_path, img)

        # cv2.imshow('random num', output_image)
        # cv2.waitKey(0)
        y = y+1

    #function to process image
    


    #save off image



# # define range of blue color in HSV
# lower_yellow = np.array([15,50,180])
# upper_yellow = np.array([40,255,255])

# # Create a mask. Threshold the HSV image to get only yellow colors
# mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# # Bitwise-AND mask and original image
# result = cv2.bitwise_and(img,img, mask= mask)

# display the mask and masked image
# cv2.imshow('Mask',mask)
# cv2.waitKey(0)
# cv2.imshow('Masked Image',output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()