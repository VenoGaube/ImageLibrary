"""Performs face alignment and stores face thumbnails in the output directory."""
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

import io

import cv2
import exifread
import scipy
from matplotlib import cm
from scipy import misc
from pathlib import Path
from PIL import Image

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet.src.facenet as fn
import facenet.src.align.detect_face as align
import random
import progressbar
import config
import findImages

import matplotlib.pyplot as plt


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


def get_rotation(img_path):
    f = open(img_path, 'rb')
    # Return Exif tags
    tags = exifread.process_file(f)
    if "Image Orientation" in tags or "Orientation" in tags:
        orientation = tags["Image Orientation"].values[0]
        return orientation


def main(args):
    # sleep(random.random())  # Zakaj je to tu?
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    fn.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = fn.get_dataset(args.input_dir)

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.create_mtcnn(sess, None)
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    # Dodan Progress Bar

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)

        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            bar = progressbar.ProgressBar(maxval=len(cls.image_paths),
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            j = 0

            for image_path in cls.image_paths:
                if Path(image_path).stem == "text":
                    continue
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                # print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = cv2.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            # print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % output_filename)
                            continue
                        if img.ndim == 2:
                            img = fn.to_rgb(img)

                        orientation = get_rotation(image_path)
                        if orientation == 3:
                            img = cv2.rotate(img, cv2.ROTATE_180)
                        elif orientation == 6:
                            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        elif orientation == 8:
                            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        # scipy.misc.imsave(image_path, img, format='JPEG')

                        img = img[:, :, 0:3]
                        cv2.imwrite(str(image_path), img)
                        img = misc.imread(image_path)
                        bounding_boxes, _ = align.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                              factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces > 0:
                            # Append Path to Image
                            config.data.append(ImageObject(str(image_path)))
                            det = bounding_boxes[:, 0:4]
                            det_arr = []
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces > 1:
                                if args.detect_multiple_faces:
                                    config.multiples.append(ImageObject(Path(image_path)))
                                    # print(config.multiples[-1].path_to_image)
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                    index = np.argmax(
                                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                    det_arr.append(det[index, :])
                            else:
                                det_arr.append(np.squeeze(det))

                            for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bounding = np.zeros(4)
                                bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                                bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                                bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                                bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
                                bounding[0] = int(np.maximum(det[0] - args.margin / 2, 0))
                                bounding[1] = int(np.maximum(det[1] - args.margin / 2, 0))
                                bounding[2] = int(np.minimum(det[2] + args.margin / 2, img_size[1]))
                                bounding[3] = int(np.minimum(det[3] + args.margin / 2, img_size[0]))
                                # Append BoundingBox of Image
                                config.data[-1].append_bb(list(bounding))
                                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                                scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                # print('\rLoading: \\', end="")
                                if args.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                    # Dodaj poleg boundingboxa še ime od patha tega obraza, da lahko pol cluster assignaš
                                    config.data[-1].append_path(output_filename_n)
                                    config.data[-1].append_cluster(99999)
                                    config.data[-1].append_emb(list())
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                misc.imsave(output_filename_n, scaled)
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            # print('\rLoading: \\', end="")
                            # print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % output_filename)
                bar.update(j + 1)
                j += 1
            bar.finish()
    print()
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
