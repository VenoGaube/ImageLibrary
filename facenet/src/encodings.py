"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import normpath, basename
from sklearn.cluster import DBSCAN
from imutils import build_montages

import argparse
import pickle
import cv2
import pathlib
import config
import progressbar
import tensorflow as tf
import numpy as np
import argparse
import facenet.src.facenet as facenet
import os
import sys
import math
import pickle
import time
import cv2

from sklearn.svm import SVC
from shutil import copy2
from pathlib import Path
from facenet.contributed import clustering as clustering
from facenet.contributed import cluster as cluster



class ImageObject:
    def __init__(self, path_to_image):
        self.path_to_image = str(path_to_image)
        self.boundingbox = {'bbox': list(), 'path': list(), 'cluster': list(), 'embedding': list()}

    def append_bb(self, bounding_boxes):
        self.boundingbox["bbox"].append(bounding_boxes)

    def append_path(self, path):
        self.boundingbox["path"].append(path)

    def append_cluster(self, cluster):
        self.boundingbox["cluster"].append(cluster)

    def append_emb(self, embedding):
        self.boundingbox["embedding"].append(embedding)


class ImageEncoding:
    def __init__(self, path_to_image, encoding):
        self.path_to_image = str(path_to_image)
        self.encoding = encoding


class AlignArguments:
    def __init__(self, path_raw_folder, path_aligned_folder):
        self.input_dir = str(path_raw_folder)
        self.output_dir = str(path_aligned_folder)
        self.image_size = 182
        self.margin = 44
        self.random_order = True
        self.detect_multiple_faces = True
        self.gpu_memory_fraction = 1.0


def check_if_multi(image_path):
    imageObjects = config.multiples
    for i in range(len(imageObjects)):
        original_name, ending = os.path.splitext(imageObjects[i].path_to_image)
        name = original_name
        if name[:-1] == '_':
            name = str(name[:-2])
        name = basename(normpath(name))
        if name == Path(image_path).stem:
            return imageObjects[i]
    return None


def euclideanMeanDistance(x, y):
    dist = tf.sqrt(tf.reduce_sum(tf.square(x - y)))
    return dist


def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                dataset = train_set
            else:
                dataset = facenet.get_dataset(args.data_dir)
            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            bar = progressbar.ProgressBar(maxval=nrof_batches_per_epoch,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                bar.update(i + 1)

            bar.finish()
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            encodings = list()
            for i in range(len(paths)):
                encodings.append(ImageEncoding(paths[i], emb_array[i]))

            clusters = clustering.cluster_facial_encodings(encodings)

            # set clusters
            for cluster in range(len(clusters)):
                # if len(clusters[cluster]) < 5:
                #   continue
                for data in clusters[cluster]:
                    flag = False
                    for i in range(len(config.data)):
                        cluster_name, ending = os.path.splitext(data)
                        for j in range(len(config.data[i].boundingbox['path'])):
                            if Path(config.data[i].boundingbox['path'][j]).stem.split('.')[0] == Path(cluster_name).stem:
                                config.data[i].boundingbox['cluster'][j] = cluster

            # set embedding
            for i in range(len(encodings)):
                for j in range(len(config.data)):
                    encodings_name, ending = os.path.splitext(encodings[i].path_to_image)
                    config_name, konec = os.path.splitext(config.data[j].path_to_image)
                    name = encodings_name
                    for k in range(len(config.data[j].boundingbox['path'])):
                        if Path(config.data[j].boundingbox['path'][k]).stem.split('.')[0] == Path(name).stem:
                            config.data[j].boundingbox['embedding'][k] = list(encodings[i].encoding)

            for i in range(len(paths)):
                for j in range(len(config.data)):
                    for k in range(len(config.data[j].boundingbox['path'])):
                        if Path(config.data[j].boundingbox['path'][k]).stem.split('.')[0] == Path(paths[i]).stem.split('.')[0]:
                            labels[i] = config.data[j].boundingbox['cluster'][k]

            results_path = os.getcwd()
            if not os.path.exists('results'):
                os.mkdir('results')
                results_path = os.path.join(results_path, 'results')
            else:
                results_path = os.path.join(results_path, 'results')
            config.result_path = results_path

            for i in range(len(clusters)):
                if len(clusters[i]) < 0:
                    for o in range(len(clusters[i])):
                        for j in range(len(config.data)):
                            for k in range(len(config.data[j].boundingbox["path"])):
                                if Path(config.data[j].boundingbox["path"][k]).stem == Path(clusters[i][o]).stem:
                                    config.data[j].boundingbox["cluster"][k] = 99999
                    continue
                else:
                    for j in range(len(clusters[i])):
                        try:
                            new_dir = Path(results_path) / Path(str(i))
                            pathlib.Path(new_dir).mkdir(parents=True, exist_ok=True)
                            copy2(str(clusters[i][j]), new_dir)
                        except IndexError:
                            pass



def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set
