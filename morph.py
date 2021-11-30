#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        self.area = self.hull.area
        self.volume = self.hull.volume
        self.dist = self.distances(self.hull)
        self.hull_coord = self.coord[self.hull.vertices, :]
        self.sphericity = self.get_sphericity()
        return None


    def get_sphericity(self):
        sph = (pi**(1/3)) * ((6 * self.volume)**(2/3))
        sph *= 1/self.area
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
        cz = np.mean(self.hull.points[self.hull.vertices, 2])
        return np.array([[cx, cy, cz]], dtype = float)

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
            param = centroid - point
            plane_mid = np.append((point), (-np.sum((point * centroid))))
            side1 = np.dot(plane_mid, np.append(point, 1))

            for j in range(len(self.hull.simplices)):
                dist_temp = np.full(len(self.hull.simplices), 1e10)

                plane = self.coord[self.hull.simplices[j], :]

                mat1 = np.array([[1, 1, 1, 1], [plane[0, 0], plane[1, 0], plane[2, 0], point[0]],
                                 [plane[0, 1], plane[1, 1], plane[2, 1], point[1]],\
                                 [plane[0, 2], plane[1, 2], plane[2, 2], point[2]]])
                mat2 = np.array([[1, 1, 1, 0], [plane[0, 0], plane[1, 0], plane[2, 0], param[0]],
                                 [plane[0, 1], plane[1, 1], plane[2, 1], param[1]],\
                                 [plane[0, 2], plane[1, 2], plane[2, 2], param[2]]])
                t = - ((np.linalg.det(mat1))/(np.linalg.det(mat2)))
                point1 = point + param*t

                if minima[0] <= point1[0] <= maxima[0]\
                and minima[1] <= point1[1] <= maxima[1]\
                and minima[2] <= point1[2] <= maxima[2]:

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
