import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.initializers import random_uniform
tf.disable_v2_behavior()

class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess,
                 fc1_dims, fc2_dims, batch_size=64, ckpt_dir='tmp/ddpg'):
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
        self.ckpt_dir = ckpt_dir
        self.build_network()
        self.params = tf.trainable_variables(scope=self.net_name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg.ckpt')
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimize_loss = self.optimizer.minimize(self.loss)

    def build_network(self):
        with tf.variable_scope(self.net_name):
            self.num_input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims[0]],
                                        name='num_inputs')
            self.img_input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims[1]],
                                        name='img_inputs')
            self.actions_ph = tf.placeholder(tf.float32,
                                                shape=[None, self.n_actions],
                                                name='actions')
            self.q_target = tf.placeholder(tf.float32,
                                            shape=[None, 1],
                                            name='targets')
            
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

            action_in = tf.layers.dense(self.actions_ph, units=self.fc2_dims,
                                        activation='relu')
            state_actions = tf.nn.relu(tf.add(batch2, action_in))

            #Dense 6 Q-values
            f3 = 0.0003
            self.q = tf.layers.dense(state_actions, units=1,
                                    kernel_initializer=random_uniform(-f3, f3),
                                    bias_initializer=random_uniform(-f3, f3),
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01))

            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

            self.action_gradients = tf.gradients(self.q, self.actions_ph)

    def predict(self, inputs, actions): #actions from actor
        return self.sess.run(self.q,
                             feed_dict={self.num_input: inputs[0],
                                        self.img_input: inputs[1],
                                        self.actions_ph: actions})

    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize_loss,
                             feed_dict={self.num_input: inputs[0],
                                        self.img_input: inputs[1],
                                        self.actions_ph: actions,
                                        self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
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