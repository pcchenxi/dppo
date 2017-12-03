
# ros images
import rospy
import tf
import sensor_msgs.point_cloud2 as pc2
import sensor_msgs
from sensor_msgs.msg import PointCloud2, PointField
from roslib import message


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
edge_map = np.zeros((MAP_ROW, MAP_COL), np.uint8)

def asMatrix(target_frame, hdr):
    print(hdr.frame_id)
    translation,rotation = TF_L.lookupTransform(target_frame, hdr.frame_id, hdr.stamp)
    return TF_L.fromTranslationRotation(translation, rotation)

def transformPointCloud(target_frame, point_cloud):
    r = sensor_msgs.msg.PointCloud()
    r.header.stamp = point_cloud.header.stamp
    r.header.frame_id = target_frame
    # r.channels = point_cloud.channels

    mat44 = asMatrix(target_frame, point_cloud.header)
    def xf(p):
        xyz = tuple(numpy.dot(mat44, numpy.array([p[0], p[1], p[2], 1.0])))[:3]
        print(xyz)
        return geometry_msgs.msg.Point(*xyz)
    r.points = [xf(p) for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z"), skip_nans=True)]
    return r

def line_detection(img):
    line_img = np.zeros((MAP_ROW, MAP_COL, 3), np.uint8)
    # edges = cv2.Canny(img,50,150,apertureSize = 3)
    minLineLength = 20
    maxLineGap = 2
    lines = cv2.HoughLinesP(img, 1, np.pi/90, 40, minLineLength, maxLineGap)
    # if lines == None:
    #     return
    print(len(lines[0]))
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),2)

    lines1 = cv2.HoughLines(img,1,np.pi/180, 40)
    lines1 = lines1[0]

    # print(len(lines1[0]))
    # for rho,theta in lines1:
    #     print ('Rho and theta:',rho,theta)
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

    cv2.imshow('line input',img)  
    cv2.waitKey(10)  

    cv2.imshow('houghlines5',line_img)  
    cv2.waitKey(0)  

def callback_map(data):
    print('map in')
    height_map = np.zeros((MAP_ROW, MAP_COL), np.uint8)
    grad_map = np.zeros((MAP_ROW, MAP_COL), np.uint8)

    for p in pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True):
        height = (p[2]+2.2)/H_RESOLUTION
        # print(p[2], height)
        if abs(p[0]) < MAP_W/2 and abs(p[1]) < MAP_H/2 and height<255 and p[2] < 1:
            row = int(p[0]/MAP_RESOLUTION) + int(MAP_ROW/2)
            col = int(p[1]/MAP_RESOLUTION) + int(MAP_COL/2)
            # print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])
            # print(row, col, height, p[2])
            value_pre = height_map[row, col]
            if value_pre == 0 or height > value_pre:
                height_map[row, col] = height
    laplacian = cv2.Laplacian(height_map,cv2.CV_64F)
    cv2.imshow('height map_ ori', height_map)
    cv2.imshow('laplacian', laplacian)
    # cv2.waitKey(0)

    for row in range(3, MAP_ROW-3, 1):
        for col in range(3, MAP_COL-3, 1):
            h = height_map[row, col]
            if h == 0:
                continue
            sub_map = height_map[row-1:row+1, col-1:col+1]
            sub_nanzero = sub_map[np.nonzero(sub_map)]
            sub_count = len(sub_nanzero) -1
            if sub_count > 0:
                # sub_sum = sub_nanzero.sum() - h
                # grad = -float(sub_sum) + sub_count * h
                # grad_map[row, col] = grad
                # print(sub_count, h, grad)

                minval = np.min(sub_nanzero)
                maxval = np.max(sub_nanzero)
                diff = maxval - minval
                if diff > 0.15/H_RESOLUTION and diff < 0.3/H_RESOLUTION:
                    grad_map[row, col] = 255#diff


    # height_map = (height_map)/0.3 * 256
    # edge_map = edge_map + height_map
    # grad_map = (grad_map - grad_map.min())/(grad_map.max() - grad_map.min()) * 256
    # grad_map = np.absolute(np.uint8(grad_map))
    line_detection(grad_map)
    cv2.imshow('grad map', grad_map)
    cv2.waitKey(0)


def callback(data):
    global edge_map
    height_map = np.zeros((MAP_ROW, MAP_COL), np.uint8)
    grad_map = np.zeros((MAP_ROW, MAP_COL), np.uint8)
    
    # TF_L.waitForTransform("/velodyne", "/base_link", rospy.Time(0),rospy.Duration(4.0))
    # data = transformPointCloud("base_link", data)

    for p in pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True):
        height = (p[2]+1.2)/H_RESOLUTION
        # print(p[2], height)
        if abs(p[0]) < MAP_W/2 and abs(p[1]) < MAP_H/2 and height<255 and p[2] < 1:
            row = int(p[0]/MAP_RESOLUTION) + int(MAP_ROW/2)
            col = int(p[1]/MAP_RESOLUTION) + int(MAP_COL/2)
            # print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])
            # print(row, col, height, p[2])
            value_pre = height_map[row, col]
            if value_pre == 0 or height > value_pre:
                height_map[row, col] = height

    # height_map = cv2.cvtColor(height_map, cv2.COLOR_BGR2GRAY)
    cv2.imshow('height map_ ori', height_map)
    cv2.waitKey(10)
    for row in range(WIN_SIZE, MAP_ROW-WIN_SIZE, 1):
        for col in range(WIN_SIZE, MAP_COL-WIN_SIZE, 1):
            h = height_map[row, col]
            if h == 0:
                continue
            sub_map = height_map[row-1:row+1, col-1:col+1]
            sub_nanzero = sub_map[np.nonzero(sub_map)]
            sub_count = len(sub_nanzero) -1
            if sub_count > 0:
                # sub_sum = sub_nanzero.sum() - h
                # grad = -float(sub_sum) + sub_count * h
                # grad_map[row, col] = grad

                minval = np.min(sub_nanzero)
                maxval = np.max(sub_nanzero)
                diff = maxval - minval
                if diff > 0.15/H_RESOLUTION and diff < 0.3/H_RESOLUTION:
                    grad_map[row, col] = 255#diff

    edge_map = cv2.max(edge_map, grad_map)
    line_detection(edge_map)
    cv2.imshow('height map', edge_map)
    cv2.waitKey(10)

def main(args):
    global TF_L
    rospy.init_node('stair', anonymous=True)

    TF_L = tf.TransformListener()
    # rospy.Subscriber("/velodyne_points", PointCloud2, callback)
    rospy.Subscriber("/map_nodelet/pointcloud", PointCloud2, callback_map)

    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)