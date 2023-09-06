import numpy as np
import json 
import cv2 
from utils.find_hough import find_line_through_point, intersection_point
from utils.barrier_noise import random_center,paste_object_image, Barrier
from utils.distortion import random_distortion_region, pixel_distortion, random_amp_freq

RAIL_AREA_YSIZE = 300
RAIL_BOTTOM_AREA_YSIZE = 20
LABEL_FILE = 'label.json'
RAIL_TRACK_LABEL = ['rail-raised', 'rail-embedded']
RAIL_WAY_LABEL = 'rail-embedded'
MAX_LINE_GAP = 10

def get_label_list(f):
    with open(f, 'r') as jf:
        label_dict = json.load(jf)
        #print(label_dict.keys())
    return  list(label_dict.keys())
    
LABEL_LIST = get_label_list(LABEL_FILE)

def bool_to_mask(bool_mask):
    mymask = bool_mask.astype(np.uint8)
    mymask[mymask==0] = 0
    mymask[mymask==1] = 255
    return mymask

def get_mask(uint8_mask, label, return_bool=False):
    if isinstance(label, int):
        mask = uint8_mask[:, :, 0] == label 
        return mask if return_bool else bool_to_mask(mask)
    elif isinstance(label, str):
        label = LABEL_LIST.index(label)
        mask = uint8_mask[:, :, 0] == label 
        return mask if return_bool else bool_to_mask(mask)
    elif isinstance(label, list) or isinstance(label, tuple) :
        mask = np.zeros_like(uint8_mask)[:,:,0].astype(bool)
        for l in label:
            l = l if isinstance(l, int) else  LABEL_LIST.index(l)
            mask = np.logical_or(uint8_mask[:, :, 0] == l, mask)
        return mask if return_bool else bool_to_mask(mask)
    raise AssertionError('Label format wrong.') 

def load_img(img_path, ):
    im = cv2.imread(img_path)
    return im

def load_mask(mask_path, ):
    im = cv2.imread(mask_path)
    return im

class RailExtractor(object):
    def __init__(self, img_path, mask_path) -> None:
        self.img = load_img(img_path) 
        self.mask = load_mask(mask_path)
        self.rail_mask= self.get_rail_mask()
        self.roi = self.get_ROI(self.rail_mask)
        
    def get_rail_mask(self):
        return get_mask(self.mask, RAIL_TRACK_LABEL)
    
    def find_rail_inits_by_middle(self, mask, 
                          bottom1=RAIL_AREA_YSIZE, 
                          bottom2=RAIL_BOTTOM_AREA_YSIZE,):
        middle = mask.shape[1] / 2 
        mymask = mask[-bottom1:, :, ...]
        contours, _ = cv2.findContours(mymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mids = []
        for ct in contours:
            x, y, w, h = cv2.boundingRect(ct)
            ct_mask = mymask[-bottom2:, x:x+w]
            rect_contours, _ = cv2.findContours(ct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edges = []
            for rct in rect_contours:
                x_, y_, w_, h_ = cv2.boundingRect(rct)
                edge_dist = min(np.abs(x_+ w_/2.), np.abs(x_+ w_/2.-w))
                edges.append([edge_dist, x_+ w_/2.])
            if len(edges) >0:
                edges = np.array(edges)
                relative_x_loc = edges[np.argmin(edges[:, 0]), 1]
                x_loc = x + relative_x_loc
                mids.append([np.abs(x_loc - middle), x_loc, x, y, w, h])
        mids = np.array(mids)
        ind = np.argsort(mids[:,0])
        return mids[ind[:2], 1:]
    
    def find_rail_inits_by_line_length(self, mask,
                            bottom1=RAIL_AREA_YSIZE, 
                          bottom2=RAIL_BOTTOM_AREA_YSIZE,):
        mymask = mask[-bottom2:, :, ...]
        wholemask = mask[-bottom1:, :, ...]
        mymask = self.delation_process(mymask)
        contours, _ = cv2.findContours(mymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        ct_list = []
        for ct in contours:
            x, y, w, h = cv2.boundingRect(ct)
            l, area_x = find_line_through_point(
                int(x+w/2.), 
                wholemask.shape[0] - 1, 
                binary_mask=wholemask, 
                y_loc=0,
                min_line_length = bottom1 / 2., 
                max_line_gap = MAX_LINE_GAP
                )
            line_length = (x+w/2. - area_x[0])**2
            ct_list.append([line_length,
                            x+w/2,
                            l[0],
                            l[1],
                            l[2] ,
                            l[3]
                            ])
        ct_list = np.array(ct_list)
        sorted_inds = np.argsort(ct_list[:, 0])
        assert len(sorted_inds) >= 2, 'not enough lines.'
        return ct_list[sorted_inds[:2], 1:]
    
    def get_ROI(self, mask):
        y = mask.shape[0] - 1
        lines = self.find_rail_inits_by_line_length(mask)
        #print(lines)
        vp_x, vp_y = intersection_point(lines[0, 1:], lines[1, 1:])
        return [[lines[0,0], y], 
                [lines[1,0], y], 
                [vp_x, vp_y + y - RAIL_AREA_YSIZE ]] 
        #point1, point2, vanishing point
    
    def delation_process(self, mask):
        return mask 
    

def random_loc(roi, img, size=None, roi_h=0.8, roi_l=0.4):
    p1, p2 ,vp = roi[0], roi[1], roi[2]
    height = p1[1] - vp[1]
    height_h = p1[1] - roi_h * height
    height_l = p1[1] - roi_l * height
    r = np.random.rand()
    r_height = r * (height_h - height_l)  + height_l
    
    width = np.abs(p1[0] -  p2[0]) * r
    width_ratio = 1./ np.abs(p1[0] -  p2[0]) 
    



def add_obsatcle(img_ori, mask_ori, img_obs,  **kwargs):
    re = RailExtractor(img_path=img_ori, 
                        mask_path=mask_ori)
    p1, p2, vp = re.roi[0], re.roi[1], re.roi[-1]
    assert  np.abs(p1[0] - p2[0]) >= 200, 'Two tracks are not correctly extracted.' 
    p3, width=random_center(vp[0], vp[1], p1[0], p1[1], p2[0])
    obj = Barrier(img_obs)
    obj_im = obj.get_random_barrier(width)
    new_im = paste_object_image(re.img, obj_im,
                                int(p3[0]), int(p3[1]),
                                 **kwargs)
    return new_im 

def add_distortion(img_ori, mask_ori,  **kwargs):
    re = RailExtractor(img_path=img_ori, 
                        mask_path=mask_ori)
    p1, p2, vp = re.roi[0], re.roi[1], re.roi[-1]
    assert  np.abs(p1[0] - p2[0]) >= 200, 'Two tracks are not correctly extracted.' 
    x, y = random_distortion_region(vp[0], vp[1], p1[0], p1[1], p2[0],)
    amp, freq = random_amp_freq()
    #print(amp, freq)
    distorted_im = pixel_distortion(re.img, x, y, amp=amp, freq=freq, **kwargs)
    return distorted_im 
    
    