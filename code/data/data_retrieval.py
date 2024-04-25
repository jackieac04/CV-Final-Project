import os
from PIL import ImageGrab
import cv2
import numpy as np
import keyboard

all_frames = np.array([])

left_count = 1
right_count = 1
none_count = 1

while(True):
    #getting frame
    img = ImageGrab.grab(bbox=(100, 10, 600, 300)) #x, y, width, height
    img_np = np.array(img) #convert image to numpy array
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) #convert to grayscale

    directory = r"C:/Users/smerc/OneDrive/Desktop/Sophomore Year/CS1430_Projects/CV-Final-Project/code/data/train"
    os.chdir(directory)

    #recording keyboard info
   
    event = keyboard.read_event() 
    if event.event_type == "down":
        if event.name == "left":
            os.chdir(directory+"/left")
            cv2.imwrite("left" + str(left_count) + ".png", frame)
            left_count += 1
        elif event.name == "right":
            os.chdir(directory+"/right")
            cv2.imwrite("right" + str(right_count) + ".png", frame)
            right_count += 1
        elif event.name == "esc":
            break
    else:
        os.chdir(directory+"/none")
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
