import numpy as np
import torch
import torch.nn as nn

class ReplayBuffer(object):
    def __init__(self, state_dim, goal_dim, max_size=int(5e5), dim_reward=1, dim_action=2, \
            save_goal_flag=False, state_im_dim=None, goal_im_dim=None):
        self.max_size = max_size
        self.ptr = 0 
        self.size = 0
        self.save_goal_flag = save_goal_flag
        self.save_img = not (state_im_dim is None)
        self.dim_action = dim_action

        self.numblocks = np.zeros((max_size, 1))
        self.next_numblocks = np.zeros((max_size, 1))
        self.state = np.zeros([max_size] + list(state_dim))
        self.next_state = np.zeros([max_size] + list(state_dim))
        self.action = np.zeros((max_size, dim_action))
        self.reward = np.zeros((max_size, dim_reward))
        self.not_done = np.zeros((max_size, 1))
        self.goal = np.zeros([max_size] + list(goal_dim))
        self.next_goal = np.zeros([max_size] + list(goal_dim))

        if self.save_goal_flag:
            self.goal_flag = np.zeros((max_size, goal_dim[0]))
            self.next_goal_flag = np.zeros((max_size, goal_dim[0]))
        if self.save_img:
            self.state_im = np.zeros([max_size] + list(state_im_dim))
            self.next_state_im = np.zeros([max_size] + list(state_im_dim))
            self.goal_im = np.zeros([max_size] + list(goal_im_dim))
            self.next_goal_im = np.zeros([max_size] + list(goal_im_dim))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done, goal, next_goal, goal_flag=None, next_goal_flag=None):
        if self.save_img:
            self.state_im[self.ptr] = state[1]
            self.next_state_im[self.ptr] = next_state[1]
            self.goal_im[self.ptr] = goal[1]
            self.next_goal_im[self.ptr] = next_goal[1]
            state = state[0]
            next_state = next_state[0]
            goal = goal[0]
            next_goal = next_goal[0]

        n_blocks = len(state)
        next_n_blocks = len(next_state)
        self.numblocks[self.ptr] = n_blocks
        self.next_numblocks[self.ptr] = next_n_blocks
        if n_blocks > 0:
            self.state[self.ptr][:n_blocks] = state
            self.goal[self.ptr][:n_blocks] = goal
        if next_n_blocks > 0:
            self.next_state[self.ptr][:next_n_blocks] = next_state
            self.next_goal[self.ptr][:next_n_blocks] = next_goal
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        if self.save_goal_flag:
            self.goal_flag[self.ptr][:len(goal_flag)] = goal_flag
            self.next_goal_flag[self.ptr][:len(next_goal_flag)] = next_goal_flag

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        data_batch = [
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.next_goal[ind]).to(self.device),
            torch.LongTensor(self.numblocks[ind]).to(self.device),
            torch.LongTensor(self.next_numblocks[ind]).to(self.device),
        ]
        if self.save_img:
            data_bath.append(torch.FloatTensor(self.state_im[ind]).to(self.device))
            data_bath.append(torch.FloatTensor(self.next_state_im[ind]).to(self.device))
            data_bath.append(torch.FloatTensor(self.goal_im[ind]).to(self.device))
            data_bath.append(torch.FloatTensor(self.next_goal_im[ind]).to(self.device))
        if self.save_goal_flag:
            data_batch.append(torch.FloatTensor(self.goal_flag[ind]).to(self.device))
            data_batch.append(torch.FloatTensor(self.next_goal_flag[ind]).to(self.device))

        return data_batch


class PER(object):
    def __init__(self, state_dim, goal_dim, max_size=int(5e5), dim_reward=1, dim_action=2, \
            save_goal_flag=False, state_im_dim=None, goal_im_dim=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.save_goal_flag = save_goal_flag
        self.save_img = not (state_im_dim is None)
        self.dim_action = dim_action

        self.tree = np.zeros(2 * max_size - 1)
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

        self.numblocks = np.zeros((max_size, 1))
        self.next_numblocks = np.zeros((max_size, 1))
        self.state = np.zeros([max_size] + list(state_dim))
        self.next_state = np.zeros([max_size] + list(state_dim))
        self.action = np.zeros((max_size, dim_action))
        self.reward = np.zeros((max_size, dim_reward))
        self.not_done = np.zeros((max_size, 1))
        self.goal = np.zeros([max_size] + list(goal_dim))
        self.next_goal = np.zeros([max_size] + list(goal_dim))

        if self.save_goal_flag:
            self.goal_flag = np.zeros((max_size, goal_dim[0]))
            self.next_goal_flag = np.zeros((max_size, goal_dim[0]))
        if self.save_img:
            self.state_im = np.zeros([max_size] + list(state_im_dim))
            self.next_state_im = np.zeros([max_size] + list(state_im_dim))
            self.goal_im = np.zeros([max_size] + list(goal_im_dim))
            self.next_goal_im = np.zeros([max_size] + list(goal_im_dim))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## tree functions ##
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def update_tree(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get_tree(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.max_size + 1
        return (idx, self.tree[idx], data_idx)

    ## replay buffer functions ##
    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, state, action, next_state, reward, done, goal=None, next_goal=None, goal_flag=None, next_goal_flag=None):
        p = self._get_priority(error)
        idx = self.ptr + self.max_size - 1

        if self.save_img:
            self.state_im[self.ptr] = state[1]
            self.next_state_im[self.ptr] = next_state[1]
            self.goal_im[self.ptr] = goal[1]
            self.next_goal_im[self.ptr] = next_goal[1]
            state = state[0]
            next_state = next_state[0]
            goal = goal[0]
            next_goal = next_goal[0]

        n_blocks = len(state)
        next_n_blocks = len(next_state)
        self.numblocks[self.ptr] = n_blocks
        self.next_numblocks[self.ptr] = next_n_blocks
        if n_blocks > 0:
            self.state[self.ptr][:n_blocks] = state
            self.goal[self.ptr][:n_blocks] = goal
        if next_n_blocks > 0:
            self.next_state[self.ptr][:next_n_blocks] = next_state
            self.next_goal[self.ptr][:next_n_blocks] = next_goal
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        if self.save_goal_flag:
            self.goal_flag[self.ptr][:len(goal_flag)] = goal_flag
            self.next_goal_flag[self.ptr][:len(next_goal_flag)] = next_goal_flag

        self.update_tree(idx, p)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n):
        priorities = []
        data_idxs = []
        idxs = []
        segment = self.total() / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i+1)
            s = np.random.uniform(a, b)
            (idx, p, data_idx) = self.get_tree(s)
            priorities.append(p)
            data_idxs.append(data_idx)
            idxs.append(idx)

        ind = np.array(data_idxs)
        data_batch = [
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.next_goal[ind]).to(self.device),
            torch.LongTensor(self.numblocks[ind]).to(self.device),
            torch.LongTensor(self.next_numblocks[ind]).to(self.device),
        ]
        if self.save_img:
            data_bath.append(torch.FloatTensor(self.state_im[ind]).to(self.device))
            data_bath.append(torch.FloatTensor(self.next_state_im[ind]).to(self.device))
            data_bath.append(torch.FloatTensor(self.goal_im[ind]).to(self.device))
            data_bath.append(torch.FloatTensor(self.next_goal_im[ind]).to(self.device))
        if self.save_goal_flag:
            data_batch.append(torch.FloatTensor(self.goal_flag[ind]).to(self.device))
            data_batch.append(torch.FloatTensor(self.next_goal_flag[ind]).to(self.device))

        sampling_probabilities = priorities / self.total()
        is_weight = np.power(self.size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return data_batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.update_tree(idx, p)


class GATReplayBuffer(object):
    def __init__(self, feature_dim, sdf_dim, max_size=int(5e5), dim_reward=1):
        self.max_size = max_size
        self.ptr = 0 
        self.size = 0
        dim_action = 2

        self.nb_st = np.zeros((max_size, 1))
        self.nb_ns = np.zeros((max_size, 1))
        self.nb_g = np.zeros((max_size, 1))
        self.feature_st = np.zeros([max_size] + list(feature_dim))
        self.feature_ns = np.zeros([max_size] + list(feature_dim))
        self.feature_g = np.zeros([max_size] + list(feature_dim))
        self.sdf_st = np.zeros([max_size] + list(sdf_dim))
        self.sdf_ns = np.zeros([max_size] + list(sdf_dim))
        self.sdf_g = np.zeros([max_size] + list(sdf_dim))
        self.action = np.zeros((max_size, dim_action))
        self.reward = np.zeros((max_size, dim_reward))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, feature_st, feature_ns, feature_g, sdf_st, sdf_ns, sdf_g, action, reward, done):
        nb_st = len(sdf_st)
        nb_ns = len(sdf_ns)
        nb_g = len(sdf_g)
        self.nb_st[self.ptr] = nb_st 
        self.nb_ns[self.ptr] = nb_ns
        self.nb_g[self.ptr] = nb_g
        if nb_st > 0:
            self.feature_st[self.ptr][:nb_st] = feature_st
            self.sdf_st[self.ptr][:nb_st] = sdf_st 
        if nb_ns > 0:
            self.feature_ns[self.ptr][:nb_ns] = feature_ns
            self.sdf_ns[self.ptr][:nb_ns] = sdf_ns
        if nb_g > 0:
            self.feature_g[self.ptr][:nb_g] = feature_g
            self.sdf_g[self.ptr][:nb_g] = sdf_g 

        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        data_batch = [
            torch.FloatTensor(self.feature_st[ind]).to(self.device),
            torch.FloatTensor(self.feature_ns[ind]).to(self.device),
            torch.FloatTensor(self.feature_g[ind]).to(self.device),
            torch.FloatTensor(self.sdf_st[ind]).to(self.device),
            torch.FloatTensor(self.sdf_ns[ind]).to(self.device),
            torch.FloatTensor(self.sdf_g[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.LongTensor(self.nb_st[ind]).to(self.device),
            torch.LongTensor(self.nb_ns[ind]).to(self.device),
            torch.LongTensor(self.nb_g[ind]).to(self.device),
        ]
        #data_batch = [
        #    torch.FloatTensor(self.sdf_st[ind]).to(self.device),
        #    torch.FloatTensor(self.sdf_ns[ind]).to(self.device),
        #    torch.FloatTensor(self.action[ind]).to(self.device),
        #    torch.FloatTensor(self.reward[ind]).to(self.device),
        #    torch.FloatTensor(self.not_done[ind]).to(self.device),
        #    torch.FloatTensor(self.sdf_g[ind]).to(self.device),
        #    torch.FloatTensor(self.sdf_ng[ind]).to(self.device),
        #    torch.LongTensor(self.numblocks[ind]).to(self.device),
        #    torch.LongTensor(self.next_numblocks[ind]).to(self.device),
        #]
        return data_batch
