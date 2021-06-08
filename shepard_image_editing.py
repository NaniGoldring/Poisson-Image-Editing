import numpy as np
import cv2
import scipy

def blend_image(source, target, mask, offset, f):

    M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])

    image_solution = np.empty_like(target)

    for ch in range(3):
        ch_source = cv2.warpAffine(source[:,:,ch],M,(target.shape[1],target.shape[0]))
        ch_mask = mask 
        ch_target = target[:,:,ch]
        
        image_solution[:,:,ch] = shepard_blending(ch_source, ch_mask, ch_target, f)
        
    return image_solution


def shepard_blending(source, mask, target, f):
    
    copy_target = np.zeros(target.shape)

    Hs,Ws = mask.shape
    
    boundary = {}

    for i in range(1,Hs-1):
        for j in range(1,Ws-1):
            if(mask[i,j] == 0 and (mask[i-1,j] != 0 or mask[i+1,j] != 0 or mask[i,j-1] != 0 or mask[i,j+1] != 0)):
                boundary[(i,j)] = target[i,j] - source[i,j]

    for i in range(1,Hs-1):
        for j in range(1,Ws-1):
            weight = 0
            weighted_avg = 0
            if(mask[i,j] != 0):
                for item in list(boundary.items()):
                    dist = f((i,j),item[0])
                    weight = weight + dist
                    weighted_avg = weighted_avg + dist*item[1]

                copy_target[i,j] = source[i,j] + weighted_avg/weight 
                
            else:
                copy_target[i,j] = target[i,j]
    return copy_target