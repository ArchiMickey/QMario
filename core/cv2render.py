from os import stat
import cv2
from config.mario import MarioConfig

def cv2render(state): 
    # a function to print the frames inside the frame stack
    for i in range(4):
        state_img = cv2.resize(state[i], (1000, 1000))
        state_img = cv2.cvtColor(state_img, cv2.COLOR_BGR2RGB)
        cv2.imshow("state", state_img)
        cv2.waitKey(1)
    return

