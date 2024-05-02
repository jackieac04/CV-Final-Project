import os
from PIL import ImageGrab
import cv2
import numpy as np
import keyboard
import time

all_frames = np.array([])

left_count = 1279
none_count = 931
right_count = 842


last_key_time = time.time()  # Initialize the last key press time

while(True):
    #getting frame
    img = ImageGrab.grab() #x, y, width, height
    img_np = np.array(img) #convert image to numpy array
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) #convert to grayscale

    # directory = r"C:/Users/annawang/Desktop/CS1430_Projects/CV-Final-Project/code/data/train"
    # directory = r"C:/Users/prana/OneDrive/Documents/GitHub/CV-Final-Project/code/data/train"
    directory = r"C:/Users\smerc/OneDrive/Desktop/Sophomore Year/CS1430_Projects/CV-Final-Project/code\data/train"
    os.chdir(directory)

    #recording keyboard info
    

    # Check if 'left', 'right', or 'esc' keys are pressed
    if keyboard.is_pressed('left') and time.time() - last_key_time > 1:
        last_key_time = time.time()  # Update the last key press time
        os.chdir(directory + "/left")
        cv2.imwrite("left" + str(left_count) + ".png", frame)
        left_count += 1
    elif keyboard.is_pressed('right') and time.time() - last_key_time > 1:
        last_key_time = time.time()  # Update the last key press time
        os.chdir(directory + "/right")
        cv2.imwrite("right" + str(right_count) + ".png", frame)
        right_count += 1
    elif keyboard.is_pressed('esc'):
        break
            
    # Check if it's been half second since the last key press for none
    if time.time() - last_key_time > 2:
        last_key_time = time.time()
        os.chdir(directory + "/none")
        cv2.imwrite("none" + str(none_count) + ".png", frame)
        none_count += 1
        

    #break loop on escape
    key = cv2.waitKey(1) 
    if key == 27: #if escape key is pressed
        break

cv2.destroyAllWindows()
print(all_frames)
print(all_frames.shape)
# print(keyboard_values.shape)
