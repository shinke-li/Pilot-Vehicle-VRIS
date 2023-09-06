import cv2
import numpy as np
import os.path as osp

TRACK_WIDTH = 2 #2 meters


def random_center(x0, y0, x1, y1, x2, max_x=1920, low_ratio=0.3, high_ratio=0.8):
    # Calculate the equation of the lines connecting the top point (x0, y0)
    # with the bottom two points (x1, y1) and (x2, y1).
    r = np.random.uniform(low=low_ratio, high=high_ratio)
    y2 = r*( y0-y1)+y1
    
    m1 = (y0 - y1) / (x0 - x1)
    b1 = y1 - m1 * x1
    m2 = (y0 - y1) / (x0 - x2)
    b2 = y1 - m2 * x2
    
    x_intersect1 = (y2 - b1) / m1
    x_intersect2 = (y2 - b2) / m2

    if 0 <= x_intersect1 <= max_x and 0 <= x_intersect2 <= max_x:
        low_x = min(x_intersect1, x_intersect2)
        high_x =  max(x_intersect1, x_intersect2)
        x_new = np.random.uniform(low=low_x, high=high_x)
        return np.array([x_new, y2]), high_x - low_x
    else:
        return None, None


def paste_object_image(background_img, object_img, x, y, mode='fusion', alpha = 0.5, **kwargs):
    #assert mode in ['']
    # Compute the size of the object image after scaling
    object_size = (object_img.shape[1], object_img.shape[0])
    
    # Compute the coordinates of the object image in the background image
    x1 = x - object_size[0]//2
    x2 = x1 + object_size[0]
    y1 = y - object_size[1]//2
    y2 = y1 + object_size[1]
    
    # Crop the object image if it goes outside the boundaries of the background image
    if x1 < 0:
        object_img = object_img[:, -x1:]
        x1 = 0

    if x2 > background_img.shape[1]:
        object_img = object_img[:, :-(x2-background_img.shape[1])]
        x2 = background_img.shape[1]

    if y1 < 0:
        object_img = object_img[-y1:, :]
        y1 = 0

    if y2 > background_img.shape[0]:
        object_img = object_img[:-(y2-background_img.shape[0]), :]
        y2 = background_img.shape[0]
    
    # Create a mask for the object image and its inverse
    _, mask = cv2.threshold(object_img[:,:,3], 1, 255, cv2.THRESH_BINARY)
    
    # Copy the object image to the background image using seamlessClone
    if mode == 'fusion':
        center = ((x1+x2)//2, (y1+y2)//2)
        output = cv2.seamlessClone(object_img[:,:,:3], background_img, mask, center, cv2.NORMAL_CLONE)
    elif mode == 'add':
        
        output = background_img.copy()
        mask = object_img[:, :, -1] / 255.0
        mask = cv2.merge([mask, mask, mask])
        h, w = object_img.shape[:2]
        roi = output[y1:y2, x1:x2]
        result = (object_img[:, :, :3] * alpha * mask + roi  * (1 - mask ) + roi  * mask * (1-alpha)).astype(np.uint8)
        output[y1:y2, x1:x2] = result

    elif mode == 'fusion_add':
        center = ((x1+x2)//2, (y1+y2)//2)
        output = cv2.seamlessClone(object_img[:,:,:3], background_img, mask, center, cv2.NORMAL_CLONE)
        mask = object_img[:, :, -1] / 255.0
        mask = cv2.merge([mask, mask, mask])
        h, w = object_img.shape[:2]
        roi = output[y1:y2, x1:x2]
        result = (object_img[:, :, :3] * alpha * mask + roi  * (1 - mask ) + roi  * mask * (1-alpha)).astype(np.uint8)
        output[y1:y2, x1:x2] = result    

    return output

class Barrier(object):
    def __init__(self, img_path) -> None:
        self.img, self.low_length, self.high_length = self.read(img_path)  
    
    def read(self,img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        print(img.shape)
        name = osp.splitext(osp.basename(img_path))[0]
        low_length, high_length = float(name.split('_')[1]),float(name.split('_')[2])
        return img, low_length, high_length 
    
    def random_scale_factor(self,  pixel_width_of_track=10):
        rnd_width = np.random.uniform(self.low_length, self.high_length)
        img_width = self.img.shape[1]
        pixel_per_width = pixel_width_of_track / float(TRACK_WIDTH)
        return float(pixel_per_width * rnd_width) / img_width

        # Define a function to scale the object image and compute the new center
    def process_object_image(self,  scale_factor):
        # Scale the object image
        img2 = self.img
        # Generate a mask for the object image by thresholding the alpha channel
        b, g, r, alpha = cv2.split(img2)
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        img2 = img2[y:y+h, x:x+w, ...]
        scaled_img2 = cv2.resize(img2, None, fx=scale_factor, fy=scale_factor)
        return scaled_img2
    
    def get_random_barrier(self, pixel_width_of_track=10):
        scale_factor = self.random_scale_factor(pixel_width_of_track)
        return self.process_object_image(scale_factor) 

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obj_img = './distortions/person_1.8_2.3.png'
    obj = Barrier(obj_img)
    plt.ims