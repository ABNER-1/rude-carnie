from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
# from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import ImageCoder, make_batch, RESIZE_FINAL
from data import inputs, standardize_image
# from detect import face_detection_model
import os
import json
import csv
import random
from guess import resolve_file, classify, batchlist, FLAGS
import cv2

RESIZE_FINAL = 227
GENDER_LIST = ['M', 'F']
AGE_LIST = [
    '(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)'
]


class RudeCarnie():
    def decode(self, raw_bytes):
        dec = tf.image.decode_jpeg(raw_bytes, channels=3)
        res = tf.image.resize_images(dec, (RESIZE_FINAL, RESIZE_FINAL))
        stand = standardize_image(res)
        return stand

    def __init__(self,
                 model_dir='/usr/src/app/deps/rude-carnie/inception_gender_checkpoint',
                 model_type='inception',
                 class_type='gender'):
        '''
        Just a wrapper around guess.py.
        '''
        self.model_dir = model_dir
        self.model_type = model_type
        self.class_type = class_type
        self.sess = tf.Session()
        model_fn = select_model(self.model_type)
        self.label_list = AGE_LIST if self.class_type == 'age' else GENDER_LIST
        nlabels = len(self.label_list)
        self.images = tf.placeholder(tf.string, [None])
        standardize = tf.map_fn(self.decode, self.images, dtype=tf.float32)
        logits = model_fn(nlabels, standardize, 1, False)
        init = tf.global_variables_initializer()

        requested_step = FLAGS.requested_step if FLAGS.requested_step else None

        checkpoint_path = '%s' % (self.model_dir)
        model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step,
                                                            FLAGS.checkpoint)

        saver = tf.train.Saver()
        saver.restore(self.sess, model_checkpoint_path)

        self.softmax_output = tf.nn.softmax(logits)

        self.coder = ImageCoder()

    def get_gender(self, files):
        '''
        This is functionally equivalent to the guess.py file in the original rude carnie directory.
        '''
        with self.sess.as_default():
            best_choices = []
            for f in files:
                image_file = resolve_file(f)
                if image_file is None: continue
                try:
                    best_choices.append(
                        classify(self.sess, self.label_list, self.softmax_output, self.coder,
                                 self.images, image_file))
                except Exception as e:
                    best_choices.append(None)
                    continue

            return best_choices

    def get_gender_batch(self, imgs):
        imgs = [cv2.imencode('.jpg', im) for im in imgs]
        batch_size = 100
        results = []
        with self.sess.as_default():
            for i in range(0, len(imgs), batch_size):
                end_index = min(i + batch_size, len(imgs))
                batch_data = imgs[i:end_index]
                batch_results = self.sess.run(
                    self.softmax_output, feed_dict={
                        self.images: batch_data
                    })
                for result in batch_results:
                    best = np.argmax(result)
                    best_choice = (self.label_list[best], result[best])
                    results.append(best_choice)
                print(end_index)
        return results
