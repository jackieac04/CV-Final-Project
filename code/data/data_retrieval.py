from PIL import ImageGrab
import cv2
import numpy as np
import keyboard

all_frames = np.array([])

while(True):
    #getting frame
    img = ImageGrab.grab(bbox=(100, 10, 600, 300)) #x, y, width, height
    img_np = np.array(img) #convert image to numpy array
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) #convert to grayscale
    all_frames = np.append(all_frames, frame) #append to all_frames
    
    cv2.imshow("frame", frame) #display frame

    keyboard_values = np.array([])

    #TODO: ensure there's a keyboard value for each frame
    #recording keyboard info
    event = keyboard.read_event()
    if event.event_type == "down":
        if event.name == "left":
            keyboard_values = np.append(keyboard_values, "left")
            print(keyboard_values)
        elif event.name == "right":
            keyboard_values = np.append(keyboard_values, "right")
            print(keyboard_values)
        elif event.name == "esc":
            break
        else:
            keyboard_values = np.append(keyboard_values, "none")

    #break loop on escape
    key = cv2.waitKey(1) 
    if key == 27: #if escape key is pressed
        break

cv2.destroyAllWindows()
print(all_frames)
print(all_frames.shape)
print(keyboard_values.shape)
