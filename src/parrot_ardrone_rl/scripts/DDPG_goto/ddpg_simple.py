"""
# Need a replay buffer class --> [1 class]
# Need a class for a target Q network (not same as a critic)
# We will use batch norm
# The policy is deterministic, so how to handle explore exploit dilemma:
# We use a stochastic policy
#to learn the greedy policy
# We will need a way to bound the actions to the env limits
# The Q action is not chosen according to an argmax but by the output of
# the other network
# We have two actor (on policy) and two critic (off-policy) networks,
# a target for each --> [2 classes]
# Target updates are soft, according to theta_prime = tau*theta +
# (1-tau)*theta_prime with tau << 1
# The target actor is just a the evaluation actor plus some noise
# They used Ornstein Uhlenback noise (Temporal Noise process that models
# the motion of brownian particles with) --> [1 class]
# 1 class for DNN
# 1 agent class as an interface between agent and DNN
"""
import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.initializers import random_uniform
tf.disable_v2_behavior()

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # use of __call__ in python
        # e.g. noise  = OUActionNoise()
        # our noise = noise()
        x = self.x_prev + self.theta*(self.mu -self.x_prev)*self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np. zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - int(done) # zero when done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        #mem_cntr is ever increasing increasing
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims,
                 fc2_dims, action_bound, batch_size=64, ckpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.net_name = name
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.sess = sess
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.ckpt_dir = ckpt_dir
        self.build_network()
        self.params = tf.trainable_variables(scope=self.net_name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg.ckpt')
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params,
                                                         -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        self.optimize = self.optimizer.apply_gradients(zip(self.actor_gradients,
                                                               self.params))
        

    def build_network(self):
        with tf.variable_scope(self.net_name):
            self.net_input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')
            self.action_gradient = tf.placeholder(tf.float32,
                                                  shape=[None, self.n_actions])

            #Layer 1 Inputs
            f1 = 1/np.sqrt(self.fc1_dims)
            dense1 = tf.layers.dense(self.net_input, units=self.fc1_dims,
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            #Layer 2 Hidden
            f2 = 1/np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims,
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.relu(batch2)

            #Layer 3 Actions
            f3 = 0.003
            mu = tf.layers.dense(layer2_activation, units=self.n_actions,
                                 kernel_initializer=random_uniform(-f3, f3),
                                 bias_initializer=random_uniform(-f3, f3))
            self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.net_input: inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize,
                      feed_dict={self.net_input: inputs,
                                 self.action_gradient: gradients})

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)


class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess,
                 fc1_dims, fc2_dims, batch_size=64, ckpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.net_name = name
        self.input_dims = input_dims
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
            self.net_input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')
            self.actions_ph = tf.placeholder(tf.float32,
                                             shape=[None, self.n_actions],
                                             name='actions')
            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None, 1],
                                           name='targets')



            f1 = 1/np.sqrt(self.fc1_dims)
            dense1 = tf.layers.dense(self.net_input, units=self.fc1_dims,
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1/np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims,
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)

            action_in = tf.layers.dense(self.actions_ph, units=self.fc2_dims,
                                        activation='relu')
            state_actions = tf.nn.relu(tf.add(batch2, action_in))

            f3 = 0.003
            self.q = tf.layers.dense(state_actions, units=1,
                                     kernel_initializer=random_uniform(-f3, f3),
                                     bias_initializer=random_uniform(-f3, f3),
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01))

            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

            self.action_gradients = tf.gradients(self.q, self.actions_ph)

    def predict(self, inputs, actions): #actions from actor
        return self.sess.run(self.q,
                             feed_dict={self.net_input: inputs,
                                        self.actions_ph: actions})

    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize_loss,
                             feed_dict={self.net_input: inputs,
                                        self.actions_ph: actions,
                                        self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.net_input: inputs,
                                        self.actions_ph: actions})

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.action_bound = 1 # to be edited

        self.actor  = Actor(alpha, n_actions, 'Actor', input_dims, self.sess,
                            layer1_size, layer2_size, env.action_space.high, 
                            self.batch_size)

        self.critic = Critic(beta, n_actions, 'Critic', input_dims, self.sess,
                             layer1_size, layer2_size, self.batch_size)

        self.target_actor  = Actor(alpha, n_actions, 'TargetActor', input_dims,
                                   self.sess,layer1_size, layer2_size, 
                                   env.action_space.high, self.batch_size)

        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims,
                                    self.sess, layer1_size, layer2_size, 
                                    self.batch_size)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_critic = [self.target_critic.params[i].assign(
            tf.multiply(self.critic.params[i], self.tau) +
            tf.multiply(self.target_critic.params[i], 1. -self.tau))
                              for i in range(len(self.target_critic.params))]

        self.update_actor = [self.target_actor.params[i].assign(
            tf.multiply(self.actor.params[i], self.tau) +
            tf.multiply(self.target_actor.params[i], 1. -self.tau))
                              for i in range(len(self.target_actor.params))]
        
        self.sess.run(tf.global_variables_initializer())

        self.update_target_network_parameters(first=True)
    
    def update_target_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state)
        noise = self.noise()
        mu_prime = mu + noise

        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                                     self.memory.sample_buffer(self.batch_size)
        #target q-value(new_state) with actor's bounded action forward pass
        critic_value_ = self.target_critic.predict(new_state,
                                          self.target_actor.predict(new_state))

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])

        target = np.reshape(target, (self.batch_size, 1))

        _ = self.critic.train(state, action, target) #s_i, a_i and y_i

        # a = mu(s_i)
        a_outs = self.actor.predict(state)
        # ∇_a Q(s, a|θ^Q)
        grads = self.critic.get_action_gradients(state, a_outs)

        self.actor.train(state, grads[0])

        self.update_target_network_parameters

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()