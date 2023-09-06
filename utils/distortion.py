import cv2
import numpy as np
def distortion_func(x, y, amp=10, freq=2, **kwargs):
    dx = amp  * np.sin(2 * np.pi * y * freq / 180)
    dy = amp * np.sin(2 * np.pi * x * freq / 180)

    return int(x + dx), int(y + dy)

def local_distortion(img, x, y, width, height, distortion_func=distortion_func, **func_args):
    h, w = img.shape[:2]
    output_img = img.copy()
    #output_img[y:y + height, x:x + width]=255
    for i in range(y, y + height):
        for j in range(x, x + width):
            src_x, src_y = distortion_func(j, i, amp=func_args['amp']*(1-(i-y)/height), freq=func_args['freq'])

            if 0 <= src_x < w and 0 <= src_y < h:
                output_img[i, j, ...] = img[int(src_y), int(src_x), ...]

    #output_img[y:y + height, x:x + width] = np.where(output_img[y:y + height, x:x + width] != 0, output_img[y:y + height, x:x + width], img[y:y + height, x:x + width])
    return output_img


def smooth_distortion_func(x, y, amp=10, freq=2, **kwargs):
    dx = amp  * np.sin(2 * np.pi * y * freq / 180)
    dy = amp * np.sin(2 * np.pi * x * freq / 180) * 0

    return x + dx, y + dy

def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())
    
def pixel_distortion(img, pixel_x, pixel_y, distortion_func=smooth_distortion_func,**func_args):
    h, w = img.shape[:2]
    output_img = img.copy()
    norm_pixel_y =  min_max_norm(pixel_y)
    x, y = distortion_func(pixel_x, pixel_y,  
                           amp=func_args['amp']*(1 - norm_pixel_y), 
                           freq=func_args['freq'])
    mask =  np.logical_and(np.logical_and(x>=0, x<w),
                            np.logical_and(0<=y, y<h))
    output_img[pixel_y[mask].astype(int), pixel_x[mask].astype(int), ...] = output_img[y[mask].astype(int), x[mask].astype(int), ...]
    return output_img

def random_distortion_region(x0, y0, x1, y1, x2, 
                  max_x=1920, 
                  low_ratio=0.2,
                  high_ratio=0.5, 
                  offset=20):
    # Calculate the equation of the lines connecting the top point (x0, y0)
    # with the bottom two points (x1, y1) and (x2, y1).
    r = np.random.uniform(low=low_ratio, high=high_ratio)
    m1 = (y0 - y1) / (x0 - x1)
    b1 = y1 - m1 * x1
    m2 = (y0 - y1) / (x0 - x2)
    b2 = y1 - m2 * x2
    y2 = int(r*( y0-y1)+y1)
    all_x = []
    all_y = []
    for y in np.arange(int(y0), int(y2)):
        x_intersect1 = int((y - b1) / m1)
        x_intersect2 = int((y - b2) / m2)
        max_xx = np.max([x_intersect1 + offset, x_intersect2 + offset]).astype(int)
        if max_xx > max_x: max_xx = max_x
        min_x = np.min([x_intersect1 - offset, x_intersect2- offset]).astype(int)
        if min_x<0: min_x = 0
        all_x.append(np.arange(min_x, max_xx))
        all_y.append(np.full((len(all_x[-1]), ), fill_value=y))
        
    return np.concatenate(all_x), np.concatenate(all_y)


def random_amp_freq(amp_range=[10, 30], freq_range=[1, 3]):
    return np.random.uniform(amp_range[0], amp_range[1]), np.random.uniform(freq_range[0], freq_range[1])