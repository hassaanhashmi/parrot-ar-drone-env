import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = [np.zeros((self.mem_size, *input_shape[0])),
                             np.zeros((self.mem_size, *input_shape[1]), dtype=np.uint8)]
        self.new_state_memory = [np.zeros((self.mem_size, *input_shape[0])),
                                 np.zeros((self.mem_size, *input_shape[1]), dtype=np.uint8)]
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np. zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        # print("new_state[0]: ",type(new_state[0]))
        # print("new_state[1]: ",type(new_state[1]))
        # print("self.new_state_memory[1]:",type(self.new_state_memory[1]))
        self.state_memory[0][index] = state[0]
        self.state_memory[1][index] = state[1]
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[0][index] = new_state[0]
        self.new_state_memory[1][index] = new_state[1]
        self.terminal_memory[index] = 1 - int(done) # zero when done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        #mem_cntr is ever increasing increasing
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states0 = self.state_memory[0][batch]
        states1 = self.state_memory[1][batch]
        actions = self.action_memory[batch]
        new_states0 = self.new_state_memory[0][batch]
        new_states1 = self.new_state_memory[1][batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return [states0, states1], actions, rewards, [new_states0, new_states1], terminal