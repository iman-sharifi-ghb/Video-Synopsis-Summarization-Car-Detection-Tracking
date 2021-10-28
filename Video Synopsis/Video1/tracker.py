#
# Performs Kalman Filter and Munkres (Hungarian) Algorithm on Bounding Boxes.
# Keeps track of the Bounding Boxes.
# TODO: min_tracker_distance (px) -> Parameter
# TODO: Max aging -> Parameter

import cv2
import math
import numpy as np
from munkres import Munkres, print_matrix
from random import randint

class Tracker:

    tracks = []
    next_id = 1
    min_tracker_distance = 100
    max_aging = 10
    max_assignment_cost = 2000

    def track(self, bounding_boxes):
        
        if len(self.tracks) == 0:
            for bounding_box in bounding_boxes:
                self.tracks.append(self.create_new_track(bounding_box))
                self.next_id += 1
        elif len(bounding_boxes) > 0:
            # Kalman Prediction & Munkres Matrix
            munkres_matrix = []
            for bounding_box in bounding_boxes:
                x2 = bounding_box.rect.center.x
                y2 = bounding_box.rect.center.y
                munkres_matrix_row = []
                for track in self.tracks:
                    prediction = track.kalman_filter.predict()
                    x1 = prediction[0]
                    y1 = prediction[1]
                    distance = math.hypot(x2 - x1, y2 - y1)
                    munkres_matrix_row.append(distance)
                munkres_matrix.append(munkres_matrix_row)

            munkres = Munkres()
            #print(munkres_matrix)
            indexes = munkres.compute(munkres_matrix)

            assignments = [None] * len(bounding_boxes)
            # Rows: BoundingBoxes, Columns: Trackers
            for row, column in indexes:
                value = munkres_matrix[row][column]
                #print('(%d, %d) -> %d' % (row, column, value))
                if value < self.max_assignment_cost:
                    assignments[row] = (column, self.tracks[column])

            for track in self.tracks:
                track.age += 1

            index = 0
            for assignment in assignments:
                if assignment is None:
                    self.tracks.append(self.create_new_track(bounding_box))
                    self.next_id += 1
                else:
                    tracker_index, tracker = assignment
                    bounding_box = bounding_boxes[index]
                    track = self.tracks[tracker_index]
                    track.bounding_box = bounding_box
                    track.kalman_filter.correct(np.array([[np.float32(bounding_box.rect.center.x)], [np.float32(bounding_box.rect.center.y)]]))
                    track.age = 0
                index += 1

            for track in list(self.tracks):
                if track.age > self.max_aging:
                    self.tracks.remove(track)
            return self.tracks

    def create_new_track(self, bounding_box):
        track = self.Track()
        kalman = self.init_kalman_filter()
        kalman.correct(np.array([[np.float32(bounding_box.rect.center.x)], [np.float32(bounding_box.rect.center.y)]]))
        kalman.predict()
        track.id = self.next_id
        track.bounding_box = bounding_box
        track.kalman_filter = kalman
        track.color = (randint(0, 255), randint(0, 255), randint(0, 255))
        return track

    def init_kalman_filter(self):
        kalman = cv2.KalmanFilter(4,2)
        kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
        return kalman

    class Track:
        id = None
        bounding_box = None
        color = None
        kalman_filter = None
        age = 0
