import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.initializers import random_uniform
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
        self.conv2_filters = 32
        self.conv2_kernel_size = 3
        self.conv3_filters = 32
        self.conv3_kernel_size = 3
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.sess = sess
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.build_network()
        self.ckpt_dir = ckpt_dir
        self.params = tf.trainable_variables(scope=self.net_name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(
            self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(tf.keras.layers.Lambda(lambda x:\
                                                tf.math.divide(x, self.batch_size)),
                                        self.unnormalized_actor_gradients))

        self.optimize = tf.train.AdamOptimizer(self.lr).\
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
            c1 = 1/np.sqrt(self.conv1_filters*(self.conv1_kernel_size+1))
            conv1 = tf.keras.layers.Conv2D(filters=self.conv1_filters,
                                     kernel_size=self.conv1_kernel_size,
                                     kernel_initializer=random_uniform(-c1, c1),
                                     bias_initializer=random_uniform(-c1, c1))(self.img_input)
            batch1 = tf.layers.batch_normalization(conv1)
            layer1_activation = tf.nn.relu(batch1)

            #Conv2D 2
            c2 = 1/np.sqrt(self.conv2_filters*(self.conv2_kernel_size+1))
            conv2 = tf.keras.layers.Conv2D(filters=self.conv2_filters,
                                     kernel_size=self.conv2_kernel_size,
                                     kernel_initializer=random_uniform(-c2, c2),
                                     bias_initializer=random_uniform(-c2, c2))(layer1_activation)
            batch2 = tf.layers.batch_normalization(conv2)
            layer2_activation = tf.nn.relu(batch2)
            
            #Conv2D 3
            c3 = 1/np.sqrt(self.conv3_filters*(self.conv3_kernel_size+1))
            conv3 = tf.keras.layers.Conv2D(filters=self.conv3_filters,
                                     kernel_size=self.conv3_kernel_size,
                                     kernel_initializer=random_uniform(-c3, c3),
                                     bias_initializer=random_uniform(-c3, c3))(layer2_activation)
            batch3 = tf.layers.batch_normalization(conv1)
            layer3_activation = tf.nn.relu(batch3)
            layer3_flat = tf.layers.flatten(layer3_activation)
            
            #Dense 4a Flat Image Input
            f1 = 1/np.sqrt(self.fc1_dims)
            img_in = tf.layers.dense(layer3_flat, units=self.fc1_dims,
                                        kernel_initializer=random_uniform(-f1, f1),
                                        bias_initializer=random_uniform(-f1, f1))
            img_batch = tf.layers.batch_normalization(img_in)
            
            #Dense 4b Numeric Input
            dense1 = tf.layers.dense(self.num_input, units=self.fc1_dims,
                                    kernel_initializer=random_uniform(-f1, f1),
                                    bias_initializer=random_uniform(-f1, f1))
            batch4 = tf.layers.batch_normalization(dense1)
            
            #Dense 4a + 4b
            input_batch = tf.nn.relu(tf.add(batch4, img_batch))
            layer4_activation = tf.nn.relu(input_batch)

            #Dense 5
            f2 = 1/np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer4_activation, units=self.fc2_dims,
                                    kernel_initializer=random_uniform(-f2, f2),
                                    bias_initializer=random_uniform(-f2, f2))
            batch5 = tf.layers.batch_normalization(dense2)
            layer5_activation = tf.nn.relu(batch5)

            #Dense 6 Actions
            f3 = 0.0003
            mu = tf.layers.dense(layer5_activation, units=self.n_actions,
                                kernel_initializer=random_uniform(-f3, f3),
                                bias_initializer=random_uniform(-f3, f3))
            self.mu = tf.multiply(mu, self.action_bound)

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