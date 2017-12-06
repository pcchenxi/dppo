
# ros images
import matplotlib.pyplot as plt
import sys, os, os.path, numpy as np
import time
import cv2

TF_L = []
MAP_W = 7
MAP_H = 7
MAP_RESOLUTION = 0.02
H_RESOLUTION = 0.05

MAP_ROW = int(MAP_W/MAP_RESOLUTION)
MAP_COL = int(MAP_H/MAP_RESOLUTION)

WIN_SIZE = 2
WIN_SIZE_RATE = 3

edge_map = np.zeros((MAP_ROW, MAP_COL), np.uint8)

def line_detection(img):
    line_img = np.zeros((MAP_ROW, MAP_COL, 3), np.uint8)
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    minLineLength = 25
    maxLineGap = 4
    lines = cv2.HoughLinesP(img, 1, np.pi/90, 40, minLineLength, maxLineGap)
    # if lines == None:
    #     return
    print(len(lines[0]))
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),2)


    # lines1 = cv2.HoughLines(img,1,np.pi/180, 55)
    # lines1 = lines1[0]

    # print(len(lines1[0]))
    # r_list = lines1[:, 0]
    # theta_list = lines1[:, 1]

    # print(np.sort(r_list))
    # for rho,theta in lines1:
    #     print ('Rho and theta:',rho,theta)
    #     # if rho > 130 and rho < 160:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a))

    #     # print (x1,y1)
    #     # print (x2,y2)

    #     cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('houghlines5',line_img)  
    cv2.waitKey(10)  

def check_stair(sub_img):
    flat_img = sub_img.flatten()
    size = int(len(flat_img)/2)
    # center = flat_img[size]
    # sub_img = sub_img - center
    # flat_img = flat_img - center

    rate = 0
    for i in range(size):
        h = flat_img[i]
        h_ = flat_img[-(i+1)]

        h_check = False
        h__check = False
        if abs(h) > 0:
            h_check = True
        if abs(h_) > 0:
            h__check = True            
        if abs(h - h_) < 0.05 and h_check and h__check:
            print(h, h_, h+h_)
            rate += 1

            print(sub_img)
            print(rate, size)

    return rate

def callback_map():
    height_map = np.zeros((MAP_ROW, MAP_COL), np.float32)
    height_map_8 = np.zeros((MAP_ROW, MAP_COL), np.uint8)
    grad_map_8 = np.zeros((MAP_ROW, MAP_COL), np.uint8)
    grad_map = np.zeros((MAP_ROW, MAP_COL), np.float32)
    rate_map = np.zeros((MAP_ROW, MAP_COL), np.uint8)

    height_map_8 = cv2.imread('./heightmap.png', 0)
    grad_map_8 = cv2.imread('./edge.png', 0)

    # for row in range(WIN_SIZE_RATE, MAP_ROW-WIN_SIZE_RATE, 1):
    #     for col in range(WIN_SIZE_RATE, MAP_COL-WIN_SIZE_RATE, 1):
    #         h = grad_map[row, col]
    #         if h != 0:
    #             continue
    #         sub_map = grad_map[row-WIN_SIZE_RATE:row+WIN_SIZE_RATE+1, col-WIN_SIZE_RATE:col+WIN_SIZE_RATE+1]
    #         rate = check_stair(sub_map)
    #         rate_map[row, col] = rate 
    #         if (rate > 0):
    #             print(row, col)

    # grad_map = np.absolute(np.uint8(grad_map))
    line_detection(grad_map_8)
    cv2.imshow('height_map_8 map', height_map_8)
    # cv2.imshow('rate map', rate_map * 255)
    cv2.imshow('grad_map_8 map', grad_map_8)
    cv2.waitKey(0)


def main(args):
    callback_map()

if __name__ == '__main__':
    main(sys.argv)