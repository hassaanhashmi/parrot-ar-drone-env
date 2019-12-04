import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Dense as k_dense
from tensorflow.compat.v1.keras.layers import Conv2D as k_conv2d
from tensorflow.compat.v1.keras.layers import Flatten as k_flatten
from tensorflow.compat.v1.keras.layers import BatchNormalization as k_batch_norm
from tensorflow.compat.v1.keras.activations import relu as k_relu
from tensorflow.compat.v1.keras.initializers import random_uniform as k_rand_uniform    
from tensorflow.compat.v1.keras.regularizers import l2 as k_l2
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.train import AdamOptimizer as v1_Adam
from tensorflow.compat.v1.losses import mean_squared_error as v1_MSE
tf.disable_v2_behavior()


class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess,
                 fc1_dims, fc2_dims, batch_size=64, ckpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.net_name = name
        self.input_dims = input_dims
        self.sess = sess
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir
        self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg.ckpt')
        self.conv1_filters = 32
        self.conv1_kernel_size = 3
        self.conv2_filters = 32
        self.conv2_kernel_size = 3
        self.conv3_filters = 32
        self.conv3_kernel_size = 3
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc1_dims
        
        with tf.device('/device:GPU:0'):
            self.build_network()
            self.params = tf.trainable_variables(scope=self.net_name)
            self.optimize_loss = v1_Adam(self.lr).minimize(self.loss)
        self.saver = tf.train.Saver()

    def build_network(self):
        with tf.variable_scope(self.net_name):
            self.num_input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims[0]],
                                        name='num_inputs')
            self.img_input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims[1]],
                                        name='img_inputs')
            self.img_size = tf.constant([45,80],dtype=tf.int32)

            self.actions_ph = tf.placeholder(tf.float32,
                                                shape=[None, self.n_actions],
                                                name='actions')
            self.q_target = tf.placeholder(tf.float32,
                                            shape=[None, 1],
                                            name='targets')

        
            c1 = 1/np.sqrt(self.conv1_filters*(self.conv1_kernel_size+1))
            self.conv_layer_1 = k_conv2d(filters=self.conv1_filters,
                                    kernel_size=self.conv1_kernel_size,
                                    kernel_initializer=k_rand_uniform(-c1, c1),
                                    bias_initializer=k_rand_uniform(-c1, c1))

            c2 = 1/np.sqrt(self.conv2_filters*(self.conv2_kernel_size+1))
            self.conv_layer_2 = k_conv2d(filters=self.conv2_filters,
                                    kernel_size=self.conv2_kernel_size,
                                    kernel_initializer=k_rand_uniform(-c2, c2),
                                    bias_initializer=k_rand_uniform(-c2, c2))

            c3 = 1/np.sqrt(self.conv3_filters*(self.conv3_kernel_size+1))
            self.conv_layer_3 = k_conv2d(filters=self.conv3_filters,
                                    kernel_size=self.conv3_kernel_size,
                                    kernel_initializer=k_rand_uniform(-c3, c3),
                                    bias_initializer=k_rand_uniform(-c3, c3))

            
            f4 = 1/np.sqrt(self.fc1_dims)
            self.dense_layer_4_img = k_dense(units=self.fc1_dims,
                                    kernel_initializer=k_rand_uniform(-f4, f4),
                                    bias_initializer=k_rand_uniform(-f4, f4))

            self.dense_layer_4_num = k_dense(units=self.fc1_dims, input_shape=(13,), #
                                    kernel_initializer=k_rand_uniform(-f4, f4),
                                    bias_initializer=k_rand_uniform(-f4, f4))

            f5 = 1/np.sqrt(self.fc2_dims)
            self.dense_layer_5 = k_dense(units=self.fc2_dims,
                                    kernel_initializer=k_rand_uniform(-f5, f5),
                                    bias_initializer=k_rand_uniform(-f5, f5))

            f6 = 0.0003
            self.dense_layer_6 = k_dense(units=1,
                                        kernel_initializer=k_rand_uniform(-f6, f6),
                                        bias_initializer=k_rand_uniform(-f6, f6),
                                        kernel_regularizer=k_l2(0.01))


            #Conv2D 1
            img_cnn = tf.image.resize(images=self.img_input, size=self.img_size,
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                      preserve_aspect_ratio=True)
            conv1 = self.conv_layer_1(img_cnn)
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
            num_in = self.dense_layer_4_num(self.num_input)
            img_batch = k_batch_norm()(img_in)
            batch4 = k_batch_norm()(num_in)
            input_batch = k_relu(tf.add(batch4, img_batch))
            layer4_activation = k_relu(input_batch)

            #Dense 5
            dense5 = self.dense_layer_5(layer4_activation)
            batch5 = k_batch_norm()(dense5)
            layer5_activation = k_relu(batch5)

            action_in = k_dense(units=self.fc2_dims, input_shape=(self.n_actions,),
                                activation='relu')(self.actions_ph)
            state_actions = k_relu(tf.add(batch5, action_in))

            #Dense 6 Q-values
            self.q = self.dense_layer_6(state_actions)

            self.loss = v1_MSE(self.q_target, self.q)

            self.action_gradients = tf.gradients(self.q, self.actions_ph)

    def predict(self, inputs, actions): #actions from actor
        return self.sess.run(self.q,
                             feed_dict={self.num_input: inputs[0],
                                        self.img_input: inputs[1],
                                        self.actions_ph: actions})

    def train(self, inputs, actions, q_target):
        with tf.device('/device:GPU:0'):
            return self.sess.run(self.optimize_loss,
                             feed_dict={self.num_input: inputs[0],
                                        self.img_input: inputs[1],
                                        self.actions_ph: actions,
                                        self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        with tf.device('/device:GPU:0'):
            return self.sess.run(self.action_gradients,
                             feed_dict={self.num_input: inputs[0],
                                        self.img_input: inputs[1],
                                        self.actions_ph: actions})

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)