import cv2
import numpy as np

from paint_mask import MaskPainter
from move_mask import MaskMover
import poisson_image_editing
import shepard_image_editing

#import argparse
import getopt
import sys
from os import path

def f(p1,p2):
    return 1/(np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))**3


def usage():
    print("Usage: python main.py [options] \n\n\
    Options: \n\
    \t-h\tPrint a brief help message and exits..\n\
    \t-s\t(Required) Specify a source image.\n\
    \t-t\t(Required) Specify a target image.\n\
    \t-m\t(Optional) Specify a mask image with the object in white and other part in black, ignore this option if you plan to draw it later.\n\
    \t-k\t(Required) Specify a kind of blending.\n\
    \t-g\t(Optional) Specify the type of poisson blending.")


if __name__ == '__main__':
    # parse command line arguments
    args = {}
    
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hs:t:m:k:g:p:")
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        print("See help: main.py -h")
        exit(2)
    for o, a in opts:
        if o in ("-h"):
            usage()
            exit()
        elif o in ("-s"):
            args["source"] = a
        elif o in ("-t"):
            args["target"] = a
        elif o in ("-m"):
            args["mask"] = a
        elif o in ("-k"):
            args["kind"] = a     
        elif o in ("-g"):
            args["GRAD_MIX"] = a  
        else:
            assert False, "unhandled option"
    
    #     
    if ("source" not in args) or ("target" not in args):
        usage()
        exit()
    
    #    
    source = cv2.imread(args["source"]).astype('float')
    target = cv2.imread(args["target"]).astype('float')
    
    if source is None or target is None:
        print('Source or target image not exist.')
        exit()

    if source.shape[0] > target.shape[0] or source.shape[1] > target.shape[1]:
        print('Source image cannot be larger than target image.')
        exit()

    # draw the mask
    mask_path = ""
    if "mask" not in args:
        print('Please highlight the object to disapparate.\n')
        mp = MaskPainter(args["source"])
        mask_path = mp.paint_mask() 
    else:
        mask_path = args["mask"]
    
    # adjust mask position for target image
    print('Please move the object to desired location to apparate.\n')
    mm = MaskMover(args["target"], mask_path)
    offset_x, offset_y, target_mask_path = mm.move_mask()            

    # blend
    print('Blending ...')
    target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE) 
    offset = offset_x, offset_y
    GRAD_MIX = True
    poisson_blend_result = 0
    if(args["kind"] == 'p'):
        if(args["GRAD_MIX"] == 'n'):
            GRAD_MIX = False    
        poisson_blend_result = poisson_image_editing.blend_image(source, target, target_mask, offset, GRAD_MIX)
    elif(args["kind"] == 's'):
        poisson_blend_result = shepard_image_editing.blend_image(source, target, target_mask, offset, f)
    
    cv2.imwrite(path.join(path.dirname(args["source"]), 'target_result.png'), 
                poisson_blend_result)
    cv2.destroyAllWindows()
    cv2.imshow("blend_result", poisson_blend_result/255)
    cv2.waitKey()

    print('Done.\n')
