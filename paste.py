from cmath import sqrt
import cv2
import numpy as np
import os, glob
import json
import math
def paste(base_image, pedestrian, x, y, head_x, head_y, position_cache, count, scale_rate):
    p_height = int(pedestrian.shape[0]*scale_rate)
    p_width = int(pedestrian.shape[1]*scale_rate)
    pedestrian =  cv2.resize(pedestrian,(p_width,p_height))
    head_x = int(head_x*scale_rate)
    head_y = int(head_y*scale_rate)
    x_left = x - head_x
    y_top = y - head_y
    
    mask = pedestrian.copy()
    mask = mask.astype(bool)
    figure_mask = mask.astype(np.int64)[:,:,0]
    mask = np.invert(mask)
    bk_mask = mask.astype(np.int64)[:,:,0]
    position_mask = np.ones_like(mask[:,:,0])*y
    local_position_cache = position_cache[y_top:y_top+p_height,x_left:x_left+p_width]
    indicate_mask = position_mask - local_position_cache
    indicate_mask[indicate_mask >= 0] = 0
    indicate_mask[indicate_mask < 0] = 1
    indicate_mask = indicate_mask*figure_mask
    indicate_mask = indicate_mask + bk_mask
    indicate_mask = indicate_mask.astype(np.int64)
    
    position_cache[y_top:y_top+p_height,x_left:x_left+p_width] = position_cache[y_top:y_top+p_height,x_left:x_left+p_width]*indicate_mask
    invert_indicate_mask = np.invert(indicate_mask.astype(bool))
    invert_indicate_mask = invert_indicate_mask.astype(np.int64)
    position_cache[y_top:y_top+p_height,x_left:x_left+p_width] = position_cache[y_top:y_top+p_height,x_left:x_left+p_width] + invert_indicate_mask*y
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # cv2.circle(pedestrian, (head_x,head_y),2,[0,0,255],-1)
    indicate_mask = indicate_mask.reshape(indicate_mask.shape[0], indicate_mask.shape[1],1)
    three_channel_indicate_mask = np.concatenate((indicate_mask, indicate_mask, indicate_mask), axis=2)
    invert_indicate_mask = invert_indicate_mask.reshape(invert_indicate_mask.shape[0], invert_indicate_mask.shape[1], 1)
    three_channel_invert_indicate_mask = np.concatenate((invert_indicate_mask, invert_indicate_mask, invert_indicate_mask), axis=2)
    base_image[y_top:y_top+p_height,x_left:x_left+p_width,:] = base_image[y_top:y_top+p_height,x_left:x_left+p_width,:]*three_channel_indicate_mask
    base_image[y_top:y_top+p_height,x_left:x_left+p_width,:] = base_image[y_top:y_top+p_height,x_left:x_left+p_width,:] + pedestrian*three_channel_invert_indicate_mask
    # base_image = cv2.putText(base_image, str(count)+"("+str(x)+","+str(y)+")", (int(x),int(y)), font, 0.25, (0, 0, 255), 1)
    return base_image, position_cache
     

def resizePedestrian(pedestrian,current_area):
    p_height = pedestrian.shape[0]
    p_width = pedestrian.shape[1]
    rate = math.sqrt(current_area/(p_height*p_width))
    new_height = int(p_height*rate)
    new_width = int(p_width*rate)
    pedestrian = cv2.resize(pedestrian,(new_width,new_height))
    return pedestrian


def createRandomPoint(roi_map, rois):
    x1,y1,x2,y2 = min(rois[:,0]), min(rois[:,1]), max(rois[:,0]), max(rois[:,1])
    while 1 :
        x = np.random.randint(0,x2+1-x1)
        y = np.random.randint(0,y2+1-y1)
        if roi_map[y,x] == 1:
            break
    return x1+x, y1+y

def getPedestrianIndex(y,rois,total_pedestrians,total_pooling_rate):
    x1,y1,x2,y2 = min(rois[:,0]), min(rois[:,1]), max(rois[:,0]), max(rois[:,1])
    y = y-y1
    y2 = y2 - y1
    index = int((y/y2)*total_pedestrians)
    real_index = np.random.randint(index*total_pooling_rate,(index+1)*total_pooling_rate)
    return real_index      

def getXRange(rois):
    rois[-1,1] = rois[0,1]
    range_x1 = rois[0,0]
    range_x2 = rois[-1,0]
    return range_x1, range_x2


def randomPlacePedestrian(args, pedestrians_pooling, base_image, probability_map):
    base_height = base_image.shape[0]
    base_width = base_image.shape[1]
    probability_array = probability_map.reshape(probability_map.shape[0]*probability_map.shape[1])
    position_array = np.arange(probability_map.shape[0]*probability_map.shape[1])
    while 1 :
        if args.real_distribution:
            if args.reward_distribution:
                p = [0.5,0.5]
            else:
                p = [1,0]
            c = [1,2]
            o = np.random.choice(c,p=p)
            if o == 1:
                pos_index = np.random.choice(position_array, p=probability_array)
                y, x = np.unravel_index(pos_index, probability_map.shape)
            elif o == 2:
                x = np.random.randint(1, base_width)
                y = np.random.randint(1, base_height)
            
        elif args.pos_pattern == 'uniform':
            x = np.random.randint(1, base_width)
            y = np.random.randint(1, base_height)

        index = np.random.randint(0,len(pedestrians_pooling))
        pedestrain_info = pedestrians_pooling[index]
        pedestrian_height = pedestrain_info[1]["height"]
        pedestrian_width = pedestrain_info[1]["width"]
        head_x = pedestrain_info[1]["ann"]["x"]
        head_y = pedestrain_info[1]["ann"]["y"]
    
        max_pedestrian_area = args.max_pedestrian_area
        current_area = (y/base_height)*max_pedestrian_area*(2/3)
        rate = math.sqrt(current_area/(pedestrian_height*pedestrian_width))
        
        pedestrian_height = int(pedestrian_height*rate)
        pedestrian_width = int(pedestrian_width*rate)
        head_x = int(head_x*rate)
        head_y = int(head_y*rate)

        x1  = x - head_x
        y1 = y - head_y
        
        x2 = x1 + pedestrian_width
        y2 = y1 + pedestrian_height
        if 0 < x1 < base_width and 0 < x2 < base_width and 0 < y1 < base_height and 0 < y2 < base_height and pedestrian_height != 0 and pedestrian_width != 0:
            break
        
    assert rate != 0
    return x, y, index, rate

def createOneImage(args, base_image, total_pedestrians, pesestrians_info, pedestrians, probability_map=None):
    position_cache = np.zeros_like(base_image[:,:,0],dtype=np.int64)
    
    pesestrians_info = sorted(pesestrians_info.items(), key= lambda x: x[1]["height"]*x[1]["width"], reverse=False)#所有挑选出来行人的信息按面积排�?
    pesestrians_info = np.array(pesestrians_info)
    count=0
    ground_truth = []
    for i in range(total_pedestrians):
        print("{0}/{1}".format(count,total_pedestrians))
        x, y, index, scale_rate = randomPlacePedestrian(args, pesestrians_info, base_image, probability_map)
        pedestrain_info = pesestrians_info[index]
        id = pedestrain_info[0]
        head_x = pedestrain_info[1]["ann"]["x"]
        head_y = pedestrain_info[1]["ann"]["y"]
        pedestrian = pedestrians[id]
        base_image ,position_cache = paste(base_image, pedestrian, x, y, head_x, head_y, position_cache, count, scale_rate)
        count += 1
        ground_truth.append([x,y])

    ground_truth = np.array(ground_truth)
    position_cache = position_cache.astype(bool)
    position_cache = position_cache.astype(np.uint8)*255
    return base_image, ground_truth, position_cache