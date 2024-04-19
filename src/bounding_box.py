import os
import cv2
import numpy as np
from read_files import ReadImageCube
from copy import deepcopy

class BoundingBoxWidget(object):
    def __init__(self, thumbnail_image):
        self.original_image = thumbnail_image
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.box_coordinates = []
        self.boxes_coordinates = []
        self.box_idx = 0

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.box_coordinates = [[x,y]]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.box_coordinates.append([x,y])
            print('top left: {}, bottom right: {}'.format(self.box_coordinates[0], self.box_coordinates[1]))
            print('x,y,w,h : ({}, {}, {}, {})'.format(self.box_coordinates[0][0], self.box_coordinates[0][1],
                                                      self.box_coordinates[1][0] - self.box_coordinates[0][0],
                                                      self.box_coordinates[1][1] - self.box_coordinates[0][1]))

            # Draw rectangle

            self.order_coord()
            print('top left: {}, bottom right: {}'.format(self.box_coordinates[0], self.box_coordinates[1]))
            cv2.rectangle(self.clone, self.box_coordinates[0], self.box_coordinates[1], (36,255,12), 1)
            cv2.imshow("image", self.clone)

            self.boxes_coordinates.append({"box_{}".format(self.box_idx):tuple(self.box_coordinates[0]+self.box_coordinates[1])})
            self.box_idx += 1
        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.cancel_all_the_boxes()

    def cancel_all_the_boxes(self):
        self.clone = self.original_image.copy()
        self.box_coordinates = []
        self.boxes_coordinates = []
        self.box_idx = 0

    def order_coord(self):
        """Order coordinates of a box such that first point is top_left, second - bottow_right"""
        coord = deepcopy(self.box_coordinates)
        if coord[0][0]>coord[1][0]:
            self.box_coordinates[0][0] = coord[1][0]
            self.box_coordinates[1][0] = coord[0][0]
        if coord[0][1]>coord[1][1]:
            self.box_coordinates[0][1] = coord[1][1]
            self.box_coordinates[1][1] = coord[0][1]

    def show_image(self):
        return self.clone


if __name__ == '__main__':
    image_dir = r"C:\Data\PhD\palimpsest\Victor_data"
    img = ReadImageCube(image_dir, [], 270)
    im = img.thumbnail(14,10)
    im = im / np.amax(im)
    boundingbox_widget = BoundingBoxWidget(im)
    while True:
        cv2.imshow('image', boundingbox_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            print(boundingbox_widget.boxes_coordinates)
            exit(1)


