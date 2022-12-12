import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import mediapipe as mp
import time
# # #-------------------------NORMAL
CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def prepare(filepath):
    # IMG_SIZE = 48
    IMG_SIZE = 200
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) 

# model = tf.keras.models.load_model("./asl_trained_model")
model = tf.keras.models.load_model("./asl_trained_model_augmented")
# model = tf.keras.models.load_model("./Res_Net_asl_trained_model")

cap = cv2.VideoCapture(0)#use 0 if using inbuilt webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    prediction = model.predict([prepare(frame)])
    final = (CATEGORIES[int(np.argmax(prediction[0]))])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,final,(200,100), font, 1, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow('Input', frame)
    

    c = cv2.waitKey(1)
    if c == 27: # hit esc key to stop
        break

cap.release()
cv2.destroyAllWindows()


#----------------------------------------BACKGROUND REMOVAL
# # initialize mediapipe 
# mp_selfie_segmentation = mp.solutions.selfie_segmentation
# selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# bg_image = np.zeros([100,100,3],dtype=np.uint8)
# bg_image.fill(255) # or img[:] = 255

# CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

# def prepare(filepath):
#     # IMG_SIZE = 48
#     IMG_SIZE = 200
#     new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
#     return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) 

# # model = tf.keras.models.load_model("./asl_trained_model")
# model = tf.keras.models.load_model("./asl_trained_model_augmented")
# # model = tf.keras.models.load_model("./Res_Net_asl_trained_model")

# cap = cv2.VideoCapture(0)#use 0 if using inbuilt webcam

# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")


# IMG_SIZE = 200

# while True:
#     key = cv2.waitKey(1)
#     ret, frame = cap.read()
#     height , width, channel = frame.shape
#     RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # get the result 
#     results = selfie_segmentation.process(RGB)

#     # extract segmented mask
#     mask = results.segmentation_mask

#     # it returns true or false where the condition applies in the mask
#     condition = np.stack(
#       (results.segmentation_mask,) * 3, axis=-1) > 0.6

#     # resize the background image to the same size of the original frame
#     bg_image = cv2.resize(bg_image, (width, height))

#     # combine frame and background image using the condition
#     output_image = np.where(condition, frame, bg_image)




#     # prepared = prepare(output_image)
#     prediction = model.predict([prepare(output_image)])

#     pic_filepath = './takenPic'
#     if not os.path.isdir(pic_filepath):
#         os.makedirs(pic_filepath)


#     individual_filepath = './takenPic/Z_01.jpg'
#     if key == ord('m'):
#         new_image_sized = cv2.resize(output_image, (IMG_SIZE, IMG_SIZE))
#         cv2.imwrite(individual_filepath, new_image_sized)
#         print("This is filepath", individual_filepath)
#         time.sleep(5.5)
#         break
        

#     final = (CATEGORIES[int(np.argmax(prediction[0]))])
    
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(output_image,final,(200,100), font, 1, (0,0,0), 2, cv2.LINE_AA)

#     cv2.imshow('Input', output_image)
    

#     # c = cv2.waitKey(1)
#     # if c == 27: # hit esc key to stop
#     #     break

# cap.release()
# cv2.destroyAllWindows()

