#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import skfmm
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from branch import find_branch_points

from skimage.segmentation import random_walker, watershed
from skimage.transform import resize
from skimage.exposure import equalize_adapthist
from skimage.measure import label, regionprops

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.lil import lil_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from scipy.ndimage import distance_transform_edt

from skimage.draw import line

from skimage.morphology import remove_small_objects

from numba import jit

import multiprocessing as mp


#CHANGE DE REGIONPROPS LOOPS WITH REMOVE SMALL OBJECTS AND BITWISE XOR

class Image_Processing():

    def __init__(self, file_name):

        self.image = self.read_image(file_name)
        self.t = self.custom_th()


    def read_image(self, file_name):
        print('Step 1: pre-processing image')
        image = ndi.imread(file_name, flatten = True)
        image *= 1/255
        image = resize(image, (779, 1019))
        image = equalize_adapthist(image)
        image *= 255
        image = np.round(image).astype(int)
        plt.imsave('pre_processed_image.png', image, format = 'png', cmap = 'Greys_r')
        print('Pre-processed image saved!')
        return image


    def custom_th(self):
        hist, _ = np.histogram(self.image, bins = np.arange(0, 256, 1))
        hist_p = hist / (self.image.shape[0] * self.image.shape[1])
        criterion = 0
        t = []
        for i in range(255):
            criterion += 1 - (hist_p[i] * 80)
            if criterion > 10:
                t.append(i)
                criterion = 0
        return t


class Single_cells():

    def __init__(self, Image_processing, side, th = 100, min_rate = 50, size = 20):
        self.image = Image_processing.image
        self.shape = self.image.shape
        self.t = Image_processing.t
        self.th = th
        self.min_rate = min_rate
        self.side = side
        self.size = size
        self.soma_t = 0
        self.count_all = self.count_objects()
        self.back_t = self.get_background()
        self.centroids = self.get_soma(th)
        self.centroids_more = self.get_soma(int(th/2), sanity = False)
        self.single_cells, self.single_masks = self.get_single_cells()


    def get_obj(self, region):

        count = 0
        props = regionprops(region)
        labels = np.zeros(len(props))
        i = 0

        for obj in props:
            if obj.area > self.th:
                count += 1
                labels[i] = obj.label
                i += 1

        labels = np.trim_zeros(labels)
        new_region = self.remove_selected_objs(labels, region)

        return count, new_region


    def get_soma(self, th, sanity = True):

        diff = np.diff(self.count_all)
        diff = np.insert(diff, 0, 0)

        for i in range(len(diff)):
            if diff[i] > self.min_rate:
                self.soma_t = diff[i]
                region = label(self.image <= self.t[i])
                props = regionprops(region)
                j = 0

                centroids = np.zeros((len(props), 2))

                for obj in props:
                    if obj.area > th:
                        centroids[j, :] = obj.centroid
                        j += 1

                centroids = centroids[~np.all(centroids == 0, axis = 1)]

                if sanity:
                    centroids = self.select_centroids(centroids)

                return centroids.astype(int)

        print('Oops... the min_rate was too high!')
        return None


    def select_centroids(self, centroids):
        dist = squareform(pdist(centroids))
        dist[np.triu_indices(dist.shape[0])] = 100
        select = np.argwhere(dist < self.size)[:, 1]
        centroids = np.delete(centroids, select, axis = 0)
        return centroids



    def remove_selected_objs(self, labels, region):

        for i in labels:
            region[region == i] = 0

        return region


    def count_objects(self):

        count_all = [0] * len(self.t)
        region = np.zeros(self.image.shape)

        for i in range(len(self.t)):
            region = label(self.image <= self.t[i] + region)
            count_all[i], region = self.get_obj(region)

        return count_all

    #PENSAR EM FORMA DE SEPARAR FUNDO!

    def get_background(self):
        diff = np.diff(self.count_all)
        for i in range(1, len(diff)):
            if diff[i-1] < 0 and (diff[i:] <= 2).all():
                return i

        print('Oops... that didn\'t work well...')
        return len(self.count_all) - 1


    def remove_spurious_region(self, mask, cell, found, centroid, low_x, low_y):

        markers = np.zeros((self.side, self.side))

        #dist = distance_transform_edt(mask)

        for k in found:
            #dist = np.linalg.norm([(k[0] - centroid[0]), (k[1] - centroid[1])])
            #if dist == 0 or dist > self.size:
            markers[(k[0] - low_x), (k[1] - low_y)] = 1

        markers = label(markers)

        #labels_ws = watershed(-dist, markers, mask = mask)

        labels_rw = random_walker(mask, markers)

        choice = labels_rw[(centroid[0] - low_x), (centroid[1] - low_y)]

        #choice = labels_ws[(centroid[0] - low_x), (centroid[1] - low_y)]

        correct_mask = labels_rw == choice

        #correct_mask = labels_ws == choice

        mask[~correct_mask] = False

        cell[~correct_mask] = 255

        return cell, mask


    def sanity_check(self, x, y):

        count = 0
        found = []

        for i in range(len(self.centroids_more)):

            in_x = self.centroids_more[i, 0] in list(range(x[0], x[1]))
            in_y = self.centroids_more[i, 1] in list(range(y[0], y[1]))

            if in_x & in_y:
                count += 1
                found.append(self.centroids_more[i, :])

        if count > 1:
            return True, found

        return False, found

    #MAKE MP COMPATIBLE

    def get_single_cells(self):

        cell_pixels = self.image.copy()
        cell_pixels[self.image > self.t[self.back_t]] = 255
        region = label(self.image <= self.t[self.back_t])
        single_cells_mask = np.zeros((self.side, self.side, len(self.centroids)))
        single_cells = np.zeros((self.side, self.side, len(self.centroids)))

        #MAKE THIS BETTER
        for i in range(len(self.centroids)):
            low_x = np.clip(int(self.centroids[i, 0] - (self.side/2)), 0, self.shape[0])
            high_x = np.clip(int(self.centroids[i, 0] + (self.side/2)), 0, self.shape[0])

            low_y = np.clip(int(self.centroids[i, 1] - (self.side/2)), 0, self.shape[1])
            high_y = np.clip(int(self.centroids[i, 1] + (self.side/2)), 0, self.shape[1])

            problem, found = self.sanity_check((low_x, high_x), (low_y, high_y))

            size_x, size_y = (high_x - low_x), (high_y - low_y)
            single_cells_mask[:size_x, :size_y, i] = region[low_x:high_x, low_y:high_y]
            single_cells[:size_x, :size_y, i] = cell_pixels[low_x:high_x, low_y:high_y]
            mask_out = single_cells_mask[:, :, i] != region[self.centroids[i, 0], self.centroids[i, 1]]
            single_cells_mask[mask_out, i] = 0
            single_cells[mask_out, i] = 255

            if problem:
                single_cells[:, :, i], single_cells_mask[:, :, i] =\
                self.remove_spurious_region(single_cells_mask[:, :, i], single_cells[:, :, i], found,\
                                            self.centroids[i, :], low_x, low_y)

        return single_cells, single_cells_mask


class Tracing():

    def __init__(self, Single_cells, th = 10):
        self.shape = Single_cells.shape
        self.single_cells = Single_cells.single_cells
        self.single_masks = Single_cells.single_masks
        self.t = Single_cells.t
        self.back_t = Single_cells.back_t
        self.th = th
        self.soma_t = Single_cells.soma_t
        self.n = len(Single_cells.centroids)


    def to_index_vec(self, points, shape):
        mut = np.array([[shape[1], 1]])
        return np.inner(points, mut)


    def to_coordinates_vec(self, index, shape):
        return np.column_stack((np.floor(index / shape[1]), np.remainder(index, shape[1]))).astype(int)


    def geodesic_transform(self, image_mask, center):
        mask = ~image_mask
        m = np.ones((image_mask.shape))
        m[tuple(center)] = 0
        m = np.ma.masked_array(m, mask)
        dist = skfmm.distance(m)
        dist = np.ma.getdata(dist)
        return dist


    def to_csr(self, dok):
        return csr_matrix(dok)


    def get_distances(self, point1, point2, dist):
        return abs(dist[point1[0], point1[1]] - dist[point2[0], point2[1]])


    def build_graph(self, image_mask, all_sources):

        adjacency = lil_matrix((len(all_sources),\
                                len(all_sources)), dtype=float)

        for i in range(len(all_sources)):
            dist = self.geodesic_transform(image_mask, all_sources[i])
            for j in range(i, len(all_sources)):
                point_dist = self.get_distances(all_sources[i], all_sources[j], dist)
                adjacency[i, j] = point_dist

        return self.to_csr(adjacency)


    def get_sources(self, image):

        samples = {}

        for i in range(self.back_t):
            key = 'sample_{}'.format(i)
            sample = self.sample_pixels(image, i+1)
            points = np.array(np.where(sample > 0)).T

            if i == 0:
                samples[key] = points
            else:
                samples[key] = np.concatenate((samples['sample_{}'.format(i-1)], points), axis = 0)

        all_sources = samples['sample_{}'.format(self.back_t-1)]

        return samples, all_sources


    def sample_pixels(self, image, step):

        factor = (step+1)
        #Should get square images with even number of pixels on each side
        if step >= 2:
            mask = (self.t[step-2] <= image) & (image < self.t[step-1])
        else:
            mask = image < self.t[step-1]

        region = image.copy()
        region[~mask] = 255

        sample = np.full(image.shape, False)

        for i in range(int(image.shape[0]/factor)):
            for j in range(int(image.shape[0]/factor)):
                temp = region[(j*factor):(j*factor)+factor, (i*factor):(i*factor)+factor]
                temp_mask = mask[(j*factor):(j*factor)+factor, (i*factor):(i*factor)+factor]

                if temp_mask.any():
                    indexes = np.unravel_index(temp.argmin(), temp.shape)
                    indexes = indexes[0] + (j*factor), indexes[1] + (i*factor)
                    sample[indexes] = True

        return sample


    def spanning_tree(self, graph, samples):

        temp_graph = graph.copy()
        temp_graph = temp_graph.toarray()
        mask = np.full(graph.shape, True)

        for i in range(len(samples)):
            full_graph = graph.copy()
            full_graph = full_graph.toarray()
            full_graph[~mask] = 0
            #print('Iteration {}'.format(i))
            mask = np.full((graph.shape), True)
            points = samples['sample_{}'.format(i)]

            for j in points:
                mask[tuple(j)] = False

            temp_graph[mask] = 0

            if i > 0:
                temp_graph = temp_graph + full_graph

            temp_graph = minimum_spanning_tree((temp_graph)).toarray()

        return temp_graph



    def image_from_graph(self, graph, shape, all_sources):

        points = np.array(np.where(graph > 0)).T
        path_points = np.empty((points.shape[0], 2, 2))
        path_points[:, :, 0] = all_sources[points[:, 0], :]
        path_points[:, :, 1] = all_sources[points[:, 1], :]

        return path_points.astype(int)



    def prune(self, draw):

        branch = find_branch_points(draw)
        temp = draw.copy()
        disconnect = np.array(np.where(branch > 0)).T

        for i in range(disconnect.shape[0]):

            temp[disconnect[i, 0], disconnect[i, 1]] = False
            temp = remove_small_objects(temp, min_size = self.th, connectivity = 2)
            temp[disconnect[i, 0], disconnect[i, 1]] = True

        return temp


    def draw_cell(self, path, shape, original_img):

        cell = np.full((shape), False)

        for i in range(len(path)):
            point0 = path[i, :, 0]
            point1 = path[i, :, 1]

            if np.linalg.norm(abs(point0 - point1)) > 0:
                rr, cc = line(path[i, 0, 0], path[i, 1, 0], path[i, 0, 1], path[i, 1, 1])
                cell[rr, cc] = True

        cell = self.prune(cell)
        cell = self.add_soma(cell, original_img)

        return cell


    def add_soma(self, image, original):

        region = original <= self.soma_t
        region = remove_small_objects(region, min_size = 100, connectivity = 1)
        new = np.logical_or(image, region)

        return new


    def get_all(self, i):
        print(i)
        draw = np.zeros((120, 120))
        mask = self.single_masks[:, :, i] > 0
        cell = self.single_cells[:, :, i]
        samples, all_sources = self.get_sources(cell)
        graph = self.build_graph(mask, all_sources)
        tree = self.spanning_tree(graph, samples)
        path = self.image_from_graph(tree, mask.shape, all_sources)
        draw = self.draw_cell(path, cell.shape, cell)
        return draw

    def Process(self):
        pool = mp.Pool(processes = 4)
        results = pool.map(self.get_all, range(self.n))

        for i in range(self.n):
            plt.imsave('cell_{}.png'.format(i), results[i], format = 'png', cmap = 'Greys')

        return results
