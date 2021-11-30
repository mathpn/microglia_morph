#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:27:49 2018

@author: mathpn
"""

import math
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from math import pi
from numba import jit


class Morphology:

    def __init__(self, Z, threshold):
        self.coord = np.array(np.where(Z > threshold))
        self.coord = self.coord.T
        self.hull = ConvexHull(self.coord)
        self.perimeter = self.hull.area
        self.area = self.hull.volume
        self.dist = self.distances(self.hull)
        self.hull_coord = self.coord[self.hull.vertices, :]
        self.sphericity = self.get_sphericity()
        return None


    def get_circularity(self):
        sph = (4 * pi * self.area) / self.perimeter
        return sph


    def distances(self, hull):
	#PRE-ALLOCATE TO REMOVE APPEND
        simplices = []
        for simplex in hull.simplices:
            simplices.append(simplex)
        simplices = np.array(simplices)
        dist = pdist(simplices)
        return dist


    def centroid(self):
        cx = np.mean(self.hull.points[self.hull.vertices, 0])
        cy = np.mean(self.hull.points[self.hull.vertices, 1])
        return np.array([[cx, cy]], dtype = float)


    def rotate_vector(self, vector, degrees):
        return np.array(math.cos(vector[0]*degrees) - math.sin(vector[1]*degrees),\
                        math.sin(vector[0]*degrees) - math.cos(vector[1]*degrees))
    
    (cache = True)
    def span(self, method = 'min'):
        centroid = np.reshape(self.centroid(), (-1))
        k = 0
        dist = np.zeros(len(self.hull_coord)**2)
        points0 = np.zeros((len(self.hull_coord)**2, 3))
        points1 = np.zeros((len(self.hull_coord)**2, 3))
        minima = np.amin(self.hull_coord, axis = 1)
        maxima = np.amax(self.hull_coord, axis = 1)

        for i in range(len(self.hull_coord)):
            point = self.hull_coord[i, :]
            vector = centroid - point
            plane_mid = np.append((point), (-np.sum((point * centroid))))
            side1 = np.dot(plane_mid, np.append(point, 1))

            for j in range(len(self.hull.simplices)):
                dist_temp = np.full(len(self.hull.simplices), 1e10)

                #Rotate a unit vector and create parametric equations, than find the intersection
                #with every line and calculate the distance, do sanity check

                vector *= 1/np.linalg.norm(vector)
                point1 = point + vector

                line = self.coord[self.hull.simplices[j], :]

                point_1 = line[0, :]
                point1_1 = line[1, :]

                #Write determinant solution to intersection
                mat1 = np.array([point[0], point[1]], [point1[0], point1[1]])

                mat2 = np.array([point_1[0], point_1[1]], [point1_1[0], point1_1[1]])

                mat3 = np.array([[point[0] - point1[0], point[1] - point1[1]],\
                                 [point_1[0] - point1_1[0], point_1[1] - point1_1[1]]])

                mat4 = np.array([[np.linalg.det(mat1), point[0] - point1[0]],\
                                 [np.linalg.det(mat2), point_1[0] - point1_1[0]]])

                mat5 = np.array([[np.linalg.det(mat1), point[1] - point1[1]],\
                                 [np.linalg.det(mat2), point_1[1] - point1_1[1]]])

                #x, y of intersection
                x = np.linalg.det(mat4)/np.linalg.det(mat3)
                y = np.linalg.det(mat5)/np.linalg.det(mat3)

                new_point = [x, y]
                #Problem: opposite direction! Maybe param > 0 is a solution is the original point is on the line! But it'd have to be recalculated for each line!
                if minima[0] <= new_point[0] <= maxima[0]\
                and minima[1] <= new_point[1] <= maxima[1]:

                    side2 = np.dot(plane_mid, np.append(point1, 1))
                    if side1 * side2 < 0:
                        temp = np.linalg.norm(point1 - point)
                        if dist_temp < np.min(dist_temp):
                            dist_temp[j] = temp
                dist[k] = np.min(dist_temp)
                points0[k] = point
                points1[k] = point1
                k += 1

        dist_i, points0_i, points1_i = np.nonzero(dist)[0], np.nonzero(points0)[0], np.nonzero(points1)[0]
        return dist[dist_i], points0[points0_i, :], points1[points1_i, :]
