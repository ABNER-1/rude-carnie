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
from utils import ImageCoder, make_batch
# from detect import face_detection_model
import os
import json
import csv
import random
from guess import resolve_file, classify, batchlist, FLAGS

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

class RudeCarnie():
    def __init__(self, model_dir='/Users/parimarjann/Downloads/21936', model_type='inception',class_type='gender'):
        ''' 
        Just a wrapper around guess.py.
        '''
        self.model_dir = model_dir
        self.model_type = model_type
        self.class_type = class_type

    def get_gender(self, files):
        '''
        This is functionally equivalent to the guess.py file in the original rude carnie directory.
        '''
        # files = random.sample(files, 10)

        with tf.Session() as sess:

            #tf.reset_default_graph()
            label_list = AGE_LIST if self.class_type == 'age' else GENDER_LIST
            nlabels = len(label_list)

            print('Executing on %s' % FLAGS.device_id)
            model_fn = select_model(self.model_type)

            with tf.device(FLAGS.device_id):
                
                images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
                logits = model_fn(nlabels, images, 1, False)
                init = tf.global_variables_initializer()
                
                requested_step = FLAGS.requested_step if FLAGS.requested_step else None
                
                checkpoint_path = '%s' % (self.model_dir)
                model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
                
                saver = tf.train.Saver()
                saver.restore(sess, model_checkpoint_path)
                            
                softmax_output = tf.nn.softmax(logits)

                coder = ImageCoder() 
                best_choices = []
                for f in files:
                    image_file = resolve_file(f) 
                    if image_file is None: continue

                    try:
                        best_choices.append(classify(sess, label_list,
                            softmax_output, coder, images, image_file))
                    except Exception as e:
                        print(e)
                        print('Failed to run image %s ' % image_file)
                        best_choices.append(None)
                        continue

                return best_choices
