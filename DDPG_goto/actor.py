import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Dense as k_dense
from tensorflow.compat.v1.keras.layers import Conv2D as k_conv2d
from tensorflow.compat.v1.keras.activations import relu as k_relu
from tensorflow.compat.v1.keras.layers import Flatten as k_flatten
from tensorflow.compat.v1.keras.optimizers import Adam as k_adam
from tensorflow.compat.v1.keras.layers import BatchNormalization as k_batch_norm
from tensorflow.compat.v1.keras.initializers import random_uniform as k_rand_uniform

tf.disable_v2_behavior()


class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims,
                 fc2_dims, action_bound, batch_size=64, ckpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.net_name = name
        self.input_dims = input_dims

        self.conv1_filters = 32
        self.conv1_kernel_size = 3
        c1 = 1/np.sqrt(self.conv1_filters*(self.conv1_kernel_size+1))
        self.conv_layer_1 = k_conv2d(filters=self.conv1_filters,
                                kernel_size=self.conv1_kernel_size,
                                kernel_initializer=k_rand_uniform(-c1, c1),
                                bias_initializer=k_rand_uniform(-c1, c1))

        self.conv2_filters = 32
        self.conv2_kernel_size = 3
        c2 = 1/np.sqrt(self.conv2_filters*(self.conv2_kernel_size+1))
        self.conv_layer_2 = k_conv2d(filters=self.conv2_filters,
                                kernel_size=self.conv2_kernel_size,
                                kernel_initializer=k_rand_uniform(-c2, c2),
                                bias_initializer=k_rand_uniform(-c2, c2))

        self.conv3_filters = 32
        self.conv3_kernel_size = 3
        c3 = 1/np.sqrt(self.conv3_filters*(self.conv3_kernel_size+1))
        self.conv_layer_3 = k_conv2d(filters=self.conv3_filters,
                                kernel_size=self.conv3_kernel_size,
                                kernel_initializer=k_rand_uniform(-c3, c3),
                                bias_initializer=k_rand_uniform(-c3, c3))

        self.fc1_dims = fc1_dims
        f4 = 1/np.sqrt(self.fc1_dims)
        self.dense_layer_4_img = k_dense(units=self.fc1_dims,
                                kernel_initializer=k_rand_uniform(-f4, f4),
                                bias_initializer=k_rand_uniform(-f4, f4))
        self.dense_layer_4_num = k_dense(units=self.fc1_dims, input_shape=(13,), #
                                kernel_initializer=k_rand_uniform(-f4, f4),
                                bias_initializer=k_rand_uniform(-f4, f4))

        self.fc2_dims = fc2_dims
        f5 = 1/np.sqrt(self.fc2_dims)
        self.dense_layer_5 = k_dense(units=self.fc2_dims,
                                kernel_initializer=k_rand_uniform(-f5, f5),
                                bias_initializer=k_rand_uniform(-f5, f5))

        f6 = 0.0003
        self.dense_layer_6 = k_dense(units=self.n_actions,
                                     kernel_initializer=k_rand_uniform(-f6, f6),
                                     bias_initializer=k_rand_uniform(-f6, f6))

        # for i, d in enumerate(["/GPU:0", "/GPU:1"]):
        #     with tf.device(d):
        self.sess = sess
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.action_bound = action_bound
        self.build_network()
        self.params = tf.trainable_variables(scope=self.net_name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(
            self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        self.optimize = k_adam(self.lr).\
                    apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):

        with tf.variable_scope(self.net_name):
            self.num_input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims[0]],
                                        name='num_inputs')
            self.img_input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims[1]],
                                        name='img_inputs')

            self.action_gradient = tf.placeholder(tf.float32,
                                        shape=[None, self.n_actions],
                                        name='action_gradient')
            
            #Conv2D 1
            conv1 = self.conv_layer_1(self.img_input)
            batch1 = k_batch_norm()(conv1)
            layer1_activation = k_relu(batch1)

            #Conv2D 2
            conv2 = self.conv_layer_2(layer1_activation)
            batch2 = k_batch_norm()(conv2)
            layer2_activation = k_relu(batch2)
            
            #Conv2D 3
            conv3 = self.conv_layer_3(layer2_activation)
            batch3 = k_batch_norm()(conv3)
            layer3_activation = k_relu(batch3)
            layer3_flat = k_flatten()(layer3_activation)

            #Dense 4 Flattened Image Input + Numeric Input
            img_in = self.dense_layer_4_img(layer3_flat)
            img_batch = k_batch_norm()(img_in)
            num_in = self.dense_layer_4_num(self.num_input)
            batch4 = k_batch_norm()(num_in)
            input_batch = k_relu(tf.add(batch4, img_batch))
            layer4_activation = k_relu(input_batch)

            #Dense 5
            dense5 = self.dense_layer_5(layer4_activation)
            batch5 = k_batch_norm()(dense5)
            layer5_activation = k_relu(batch5)

            #Dense 6 Actions
            mu = self.dense_layer_6(layer5_activation)
            self.mu = tf.math.multiply(mu, self.action_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.net_input: inputs[0],
                                                 self.img_input: inputs[1]})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize,
                            feed_dict={self.net_input: inputs[0],
                                       self.img_input: inputs[1],
                                       self.action_gradient: gradients})

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)