import argparse
import numpy as np
import cv2
from os import path
 

class MaskPainter():
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image_copy = self.image.copy()

        self.mask = np.zeros(self.image.shape)
        self.mask_copy = self.mask.copy()
        self.size = 4
        self.to_draw = False

        self.window_name = "Draw mask. s:save; r:reset; q:quit"


    def _paint_mask_handler(self, event, x, y, flags, param):    
        if event == cv2.EVENT_LBUTTONDOWN:  #indicates that the left mouse button is pressed / start painting
            self.to_draw = True        

        elif event == cv2.EVENT_MOUSEMOVE: #indicates that the mouse pointer has moved over the window / where to paint
            if self.to_draw:
                cv2.rectangle(self.image, (x-self.size, y-self.size), 
                              (x+self.size, y+self.size),
                              (255, 0, 0), -1)
                cv2.rectangle(self.mask, (x-self.size, y-self.size), 
                              (x+self.size, y+self.size),
                              (255, 255, 255), -1)
                cv2.imshow(self.window_name, self.image)

        elif event == cv2.EVENT_LBUTTONUP:#indicates that left mouse button is released / stop painting
            self.to_draw = False        


    def paint_mask(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, 
                             self._paint_mask_handler)
 
        while True:        
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF #get the key
 
            if key == ord("r"): #reset all
                self.image = self.image_copy.copy()
                self.mask = self.mask_copy.copy()
     
            elif key == ord("s"): #save the mask
                break

            elif key == ord("q"): #quit
                cv2.destroyAllWindows()
                exit()

        temp_mask = self.mask
        cv2.imshow("Press any key to save the mask", temp_mask)
        cv2.waitKey(0)
        maskPath = path.join(path.dirname(self.image_path), 
                              'mask.png')
        cv2.imwrite(maskPath, self.mask)
 
        # close all open windows
        cv2.destroyAllWindows()
        return maskPath

