import cv2
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn


dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
criterion = nn.SmoothL1Loss(reduction='mean').type(dtype)

def smoothing_log(log_data, log_freq):
    return np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq

def smoothing_log_same(log_data, log_freq):
    return np.concatenate([np.array([np.nan] * (log_freq-1)), np.convolve(log_data, np.ones(log_freq), 'valid') / log_freq])

def combine_batch(minibatch, data):
    try:
        combined = []
        if minibatch is None:
            for i in range(len(data)):
                combined.append(data[i].unsqueeze(0))
        else:
            for i in range(len(minibatch)):
                combined.append(torch.cat([minibatch[i], data[i].unsqueeze(0)]))
    except:
        print(i)
        print(data[i].shape)
        print(minibatch[i].shape)
        print(data[i])
        print(minibatch[i])
    return combined

def sample_her_transitions(env, info):
    _info = deepcopy(info)
    move_threshold = 0.005
    range_x = env.block_range_x
    range_y = env.block_range_y

    pre_poses = info['pre_poses']
    poses = info['poses']
    pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
    if np.linalg.norm(poses - pre_poses) < move_threshold:
        return []

    for i in range(env.num_blocks):
        if pos_diff[i] < move_threshold:
            continue
        ## 1. archived goal ##
        archived_goal = poses[i]

        ## clipping goal pose ##
        x, y = archived_goal
        _info['goals'][i] = np.array([x, y])

    ## recompute reward  ##
    reward_recompute, done_recompute, block_success_recompute = env.get_reward(_info)

    return [[reward_recompute, _info['goals'], done_recompute, block_success_recompute]]

def sample_ig_transitions(env, info, num_samples=1):
    if env.detection:
        return []
    move_threshold = 0.005
    range_x = env.block_range_x
    range_y = env.block_range_y
    n_blocks = env.num_blocks

    pre_poses = info['pre_poses']
    poses = info['poses']
    pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
    if np.linalg.norm(poses - pre_poses) < move_threshold:
        return []

    transitions = []
    for s in range(num_samples):
        _info = deepcopy(info)
        for i in range(n_blocks):
            if pos_diff[i] < move_threshold:
                continue
            ## 1. archived goal ##
            gx = np.random.uniform(*range_x)
            gy = np.random.uniform(*range_y)
            archived_goal = np.array([gx, gy])
            _info['goals'][i] = archived_goal

        ## recompute reward  ##
        reward_recompute, done_recompute, block_success_recompute = env.get_reward(_info)
        transitions.append([reward_recompute, _info['goals'], done_recompute, block_success_recompute])

    return transitions


# HER with predicted R
def sample_her_transitions_withR_sa(R, env, info):
    _info = deepcopy(info)
    move_threshold = 0.005
    range_x = env.block_range_x
    range_y = env.block_range_y

    pre_poses = info['pre_poses']
    poses = info['poses']
    pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
    if np.linalg.norm(poses - pre_poses) < move_threshold:
        return []

    for i in range(env.num_blocks):
        if pos_diff[i] < move_threshold:
            continue
        ## 1. archived goal ##
        archived_goal = poses[i]

        ## clipping goal pose ##
        x, y = archived_goal
        _info['goals'][i] = np.array([x, y])

    ## recompute reward  ##
    states = _info['pre_poses']
    goals = _info['goals']
    action = _info['action']
    state = torch.tensor(states).type(dtype).unsqueeze(0)
    goal = torch.tensor(goals).type(dtype).unsqueeze(0)
    r_hat = R([state, goal])[0, action[0], action[1]]

    reward_recompute = r_hat.detach().cpu().numpy().item()
    done_recompute = False
    block_success_recompute = [False] * env.num_blocks

    if _info['out_of_range']:
        if env.seperate:
            reward_recompute = [-1.] * env.num_blocks
        else:
            reward_recompute = -1.

    return [[reward_recompute, _info['goals'], done_recompute, block_success_recompute]]

def sample_ig_transitions_withR_sa(R, env, info, num_samples=1):
    move_threshold = 0.005
    range_x = env.block_range_x
    range_y = env.block_range_y
    n_blocks = env.num_blocks

    pre_poses = info['pre_poses']
    poses = info['poses']
    pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
    if np.linalg.norm(poses - pre_poses) < move_threshold:
        return []

    transitions = []
    for s in range(num_samples):
        _info = deepcopy(info)
        for i in range(n_blocks):
            if pos_diff[i] < move_threshold:
                continue
            ## 1. archived goal ##
            gx = np.random.uniform(*range_x)
            gy = np.random.uniform(*range_y)
            archived_goal = np.array([gx, gy])
            _info['goals'][i] = archived_goal

        ## recompute reward  ##
        states = _info['pre_poses']
        goals = _info['goals']
        action = _info['action']
        state = torch.tensor(states).type(dtype).unsqueeze(0)
        goal = torch.tensor(goals).type(dtype).unsqueeze(0)
        r_hat = R([state, goal])[0, action[0], action[1]]

        reward_recompute = r_hat.detach().cpu().numpy().item()
        done_recompute = False
        block_success_recompute = [False] * env.num_blocks
        if _info['out_of_range']:
            if env.seperate:
                reward_recompute = [-1.] * env.num_blocks
            else:
                reward_recompute = -1.
        transitions.append([reward_recompute, _info['goals'], done_recompute, block_success_recompute])

    return transitions


def sample_her_transitions_withR_sns(R, env, info):
    _info = deepcopy(info)
    move_threshold = 0.005
    range_x = env.block_range_x
    range_y = env.block_range_y

    pre_poses = info['pre_poses']
    poses = info['poses']
    pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
    if np.linalg.norm(poses - pre_poses) < move_threshold:
        return []

    for i in range(env.num_blocks):
        if pos_diff[i] < move_threshold:
            continue
        ## 1. archived goal ##
        archived_goal = poses[i]

        ## clipping goal pose ##
        x, y = archived_goal
        _info['goals'][i] = np.array([x, y])

    ## recompute reward  ##
    states = _info['pre_poses']
    next_states = _info['poses']
    goals = _info['goals']
    state = torch.tensor(states).type(dtype).unsqueeze(0)
    goal = torch.tensor(goals).type(dtype).unsqueeze(0)
    nextstate = torch.tensor(next_states).type(dtype).unsqueeze(0)
    r_hat = R([state, goal, nextstate])[0]

    reward_recompute = r_hat.detach().cpu().numpy().item()
    done_recompute = False
    block_success_recompute = [False] * env.num_blocks

    if _info['out_of_range']:
        if env.seperate:
            reward_recompute = [-1.] * env.num_blocks
        else:
            reward_recompute = -1.

    return [[reward_recompute, _info['goals'], done_recompute, block_success_recompute]]

def sample_ig_transitions_withR_sns(R, env, info, num_samples=1):
    move_threshold = 0.005
    range_x = env.block_range_x
    range_y = env.block_range_y
    n_blocks = env.num_blocks

    pre_poses = info['pre_poses']
    poses = info['poses']
    pos_diff = np.linalg.norm(poses - pre_poses, axis=1)
    if np.linalg.norm(poses - pre_poses) < move_threshold:
        return []

    transitions = []
    for s in range(num_samples):
        _info = deepcopy(info)
        for i in range(n_blocks):
            if pos_diff[i] < move_threshold:
                continue
            ## 1. archived goal ##
            gx = np.random.uniform(*range_x)
            gy = np.random.uniform(*range_y)
            archived_goal = np.array([gx, gy])
            _info['goals'][i] = archived_goal

        ## recompute reward  ##
        states = _info['pre_poses']
        next_states = _info['poses']
        goals = _info['goals']
        state = torch.tensor(states).type(dtype).unsqueeze(0)
        goal = torch.tensor(goals).type(dtype).unsqueeze(0)
        nextstate = torch.tensor(next_states).type(dtype).unsqueeze(0)
        r_hat = R([state, goal, nextstate])[0]

        reward_recompute = r_hat.detach().cpu().numpy().item()
        done_recompute = False
        block_success_recompute = [False] * env.num_blocks
        if _info['out_of_range']:
            if env.seperate:
                reward_recompute = [-1.] * env.num_blocks
            else:
                reward_recompute = -1.
        transitions.append([reward_recompute, _info['goals'], done_recompute, block_success_recompute])

    return transitions


## RewardNet Loss ##
def reward_loss_sa(minibatch, R):
    state = minibatch[0]
    #next_state = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    #not_done = minibatch[4]
    goal = minibatch[5]
    batch_size = state.size()[0]

    state_goal = [state, goal]
    r_hat = R(state_goal)
    pred = r_hat[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)
    y_target = rewards

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def reward_loss_sns(minibatch, R):
    state = minibatch[0]
    next_state = minibatch[1]
    #actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    #not_done = minibatch[4]
    goal = minibatch[5]
    batch_size = state.size()[0]

    state_goal_nextstate = [state, goal, next_state]
    r_hat = R(state_goal_nextstate)
    pred = r_hat[torch.arange(batch_size)]
    y_target = rewards

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

## DQN Loss ##
def calculate_loss_origin(minibatch, Q, Q_target, gamma=0.5):
    state = minibatch[0]
    next_state = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal = minibatch[5]
    batch_size = state.size()[0]

    state_goal = [state, goal]
    next_state_goal = [next_state, goal]

    next_q = Q_target(next_state_goal)
    next_q_max = next_q.max(1)[0].max(1)[0]
    y_target = rewards + gamma * not_done * next_q_max

    q_values = Q(state_goal)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_double(minibatch, Q, Q_target, gamma=0.5):
    state = minibatch[0]
    next_state = minibatch[1]
    rewards = minibatch[3]
    actions = minibatch[2].type(torch.long)
    not_done = minibatch[4]
    goal = minibatch[5]
    batch_size = state.size()[0]

    state_goal = [state, goal]
    next_state_goal = [next_state, goal]

    def get_a_prime():
        next_q = Q(next_state_goal)
        obj = next_q.max(2)[0].max(1)[1]
        theta = next_q.max(1)[0].max(1)[1]
        return obj, theta

    a_prime = get_a_prime()

    next_q_target = Q_target(next_state_goal)
    q_target_s_a_prime = next_q_target[torch.arange(batch_size), a_prime[0], a_prime[1]].unsqueeze(1)
    y_target = rewards + gamma * not_done * q_target_s_a_prime

    q_values = Q(state_goal)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## GCN-nsdf Loss ##
def calculate_loss_gcn_nsdf_origin(minibatch, Q, Q_target, gamma=0.5):
    state = minibatch[0]
    next_state = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal = minibatch[5]
    next_goal = minibatch[6]
    nsdf = minibatch[7].squeeze()
    next_nsdf = minibatch[8].squeeze()
    batch_size = state.size()[1]

    state_goal = [state, goal]
    next_state_goal = [next_state, next_goal]

    next_q = Q_target(next_state_goal, next_nsdf)
    empty_mask = (next_state_goal[0].sum((2, 3))==0)
    next_q[empty_mask] = next_q.min()

    next_q_max = next_q.max(1)[0].max(1)[0]
    y_target = rewards + gamma * not_done * next_q_max

    q_values = Q(state_goal, nsdf)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_gcn_nsdf_double(minibatch, Q, Q_target, gamma=0.5):
    state = minibatch[0]
    next_state = minibatch[1]
    rewards = minibatch[3]
    actions = minibatch[2].type(torch.long)
    not_done = minibatch[4]
    goal = minibatch[5]
    next_goal = minibatch[6]
    nsdf = minibatch[7].squeeze()
    next_nsdf = minibatch[8].squeeze()
    batch_size = state.size()[0]

    state_goal = [state, goal]
    next_state_goal = [next_state, next_goal]

    def get_a_prime():
        next_q = Q(next_state_goal, next_nsdf)
        empty_mask = (next_state_goal[0].sum((2, 3))==0)
        next_q[empty_mask] = next_q.min()

        obj = next_q.max(2)[0].max(1)[1]
        theta = next_q.max(1)[0].max(1)[1]
        return obj, theta

    a_prime = get_a_prime()
    next_q_target = Q_target(next_state_goal, next_nsdf)
    q_target_s_a_prime = next_q_target[torch.arange(batch_size), a_prime[0], a_prime[1]].unsqueeze(1)
    y_target = rewards + gamma * not_done * q_target_s_a_prime

    q_values = Q(state_goal, nsdf)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_pose_nsdf_double(minibatch, Q, Q_target, gamma=0.5):
    state = minibatch[0]
    next_state = minibatch[1]
    rewards = minibatch[3]
    actions = minibatch[2].type(torch.long)
    not_done = minibatch[4]
    goal = minibatch[5]
    next_goal = minibatch[6]
    nsdf = minibatch[7].squeeze()
    next_nsdf = minibatch[8].squeeze()
    batch_size = state.size()[0]

    state_goal = [state, goal]
    next_state_goal = [next_state, next_goal]

    def get_a_prime():
        next_q = Q(next_state_goal, next_nsdf)
        empty_mask = (next_state_goal[0].sum(2)==0)
        next_q[empty_mask] = next_q.min()

        obj = next_q.max(2)[0].max(1)[1]
        theta = next_q.max(1)[0].max(1)[1]
        return obj, theta

    a_prime = get_a_prime()
    next_q_target = Q_target(next_state_goal, next_nsdf)
    q_target_s_a_prime = next_q_target[torch.arange(batch_size), a_prime[0], a_prime[1]].unsqueeze(1)
    y_target = rewards + gamma * not_done * q_target_s_a_prime

    q_values = Q(state_goal, nsdf)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## GCN Loss ##
def calculate_loss_gcn_origin(minibatch, Q, Q_target, gamma=0.5):
    state = minibatch[0]
    next_state = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal = minibatch[5]
    next_goal = minibatch[6]
    batch_size = state.size()[1]

    state_goal = [state, goal]
    next_state_goal = [next_state, next_goal]

    next_q = Q_target(next_state_goal)
    empty_mask = (next_state_goal[0].sum((2, 3))==0)
    next_q[empty_mask] = next_q.min()

    next_q_max = next_q.max(1)[0].max(1)[0]
    y_target = rewards + gamma * not_done * next_q_max

    q_values = Q(state_goal)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_gcn_double(minibatch, Q, Q_target, gamma=0.5):
    state = minibatch[0]
    next_state = minibatch[1]
    rewards = minibatch[3]
    actions = minibatch[2].type(torch.long)
    not_done = minibatch[4]
    goal = minibatch[5]
    next_goal = minibatch[6]
    batch_size = state.size()[0]

    state_goal = [state, goal]
    next_state_goal = [next_state, next_goal]

    def get_a_prime():
        next_q = Q(next_state_goal)
        empty_mask = (next_state_goal[0].sum((2, 3))==0)
        next_q[empty_mask] = next_q.min()

        obj = next_q.max(2)[0].max(1)[1]
        theta = next_q.max(1)[0].max(1)[1]
        return obj, theta

    a_prime = get_a_prime()
    next_q_target = Q_target(next_state_goal)
    q_target_s_a_prime = next_q_target[torch.arange(batch_size), a_prime[0], a_prime[1]].unsqueeze(1)
    y_target = rewards + gamma * not_done * q_target_s_a_prime

    q_values = Q(state_goal)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error


## GCN GoalFlag Loss ##
def calculate_loss_gcn_gf_origin(minibatch, Q, Q_target, gamma=0.5):
    state = minibatch[0]
    next_state = minibatch[1]
    actions = minibatch[2].type(torch.long)
    rewards = minibatch[3]
    not_done = minibatch[4]
    goal = minibatch[5]
    next_goal = minibatch[6]
    nsdf = minibatch[7].squeeze()
    next_nsdf = minibatch[8].squeeze()
    goalflag = minibatch[9].squeeze()
    next_goalflag = minibatch[10].squeeze()
    batch_size = state.size()[1]

    state_goal = [state, goal]
    next_state_goal = [next_state, next_goal]

    next_q = Q_target(next_state_goal, next_nsdf, next_goalflag)
    empty_mask = (next_state_goal[0].sum((2, 3))==0)
    next_q[empty_mask] = next_q.min()
    '''
    if len(next_q)<=1:
        next_q[next_nsdf:] = next_q.min()
    else:
        for nq, ns in zip(next_q, next_nsdf):
            nq[ns:] = nq.min()
    '''
    next_q_max = next_q.max(1)[0].max(1)[0]
    y_target = rewards + gamma * not_done * next_q_max

    q_values = Q(state_goal, nsdf, goalflag)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_gcn_gf_double(minibatch, Q, Q_target, gamma=0.5):
    state = minibatch[0]
    next_state = minibatch[1]
    rewards = minibatch[3]
    actions = minibatch[2].type(torch.long)
    not_done = minibatch[4]
    goal = minibatch[5]
    next_goal = minibatch[6]
    nsdf = minibatch[7].squeeze()
    next_nsdf = minibatch[8].squeeze()
    goalflag = minibatch[9].squeeze()
    next_goalflag = minibatch[10].squeeze()
    batch_size = state.size()[0]

    state_goal = [state, goal]
    next_state_goal = [next_state, next_goal]

    def get_a_prime():
        next_q = Q(next_state_goal, next_nsdf, next_goalflag)
        empty_mask = (next_state_goal[0].sum((2, 3))==0)
        next_q[empty_mask] = next_q.min()
        '''
        if len(next_q)<=1:
            next_q[next_nsdf:] = next_q.min()
        else:
            for i in range(len(next_q)):
                next_q[i, next_nsdf[i]:] = next_q[i].min()
        '''
        obj = next_q.max(2)[0].max(1)[1]
        theta = next_q.max(1)[0].max(1)[1]
        return obj, theta

    a_prime = get_a_prime()
    next_q_target = Q_target(next_state_goal, next_nsdf, next_goalflag)
    q_target_s_a_prime = next_q_target[torch.arange(batch_size), a_prime[0], a_prime[1]].unsqueeze(1)
    y_target = rewards + gamma * not_done * q_target_s_a_prime

    q_values = Q(state_goal, nsdf, goalflag)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

## GAT Loss ##
def calculate_loss_gat_origin(minibatch, Q, Q_target, gamma=0.5):
    feature_st = minibatch[0]
    feature_ns = minibatch[1]
    feature_g = minibatch[2]
    sdf_st = minibatch[3]
    sdf_ns = minibatch[4]
    sdf_g = minibatch[5]
    actions = minibatch[6].type(torch.long)
    rewards = minibatch[7]
    not_done = minibatch[8]
    nb_st = minibatch[9].squeeze()
    nb_ns = minibatch[10].squeeze()
    nb_g = minibatch[11].squeeze()
    batch_size = sdf_st.size()[0]

    obs_st = [feature_st, sdf_st]
    obs_ns = [feature_ns, sdf_ns]
    obs_g = [feature_g, sdf_g]

    next_q = Q_target(obs_ns, obs_g, nb_ns, nb_g)
    if len(next_q)<=1:
        next_q[nb_ns:] = next_q.min()
    else:
        for nq, ns in zip(next_q, nb_ns):
            nq[ns:] = nq.min()
    next_q_max = next_q.max(1)[0].max(1)[0]
    y_target = rewards + gamma * not_done * next_q_max

    q_values = Q(obs_st, obs_g, nb_st, nb_g)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

def calculate_loss_gat_double(minibatch, Q, Q_target, gamma=0.5):
    feature_st = minibatch[0]
    feature_ns = minibatch[1]
    feature_g = minibatch[2]
    sdf_st = minibatch[3]
    sdf_ns = minibatch[4]
    sdf_g = minibatch[5]
    actions = minibatch[6].type(torch.long)
    rewards = minibatch[7]
    not_done = minibatch[8]
    nb_st = minibatch[9].squeeze()
    nb_ns = minibatch[10].squeeze()
    nb_g = minibatch[11].squeeze()
    batch_size = sdf_st.size()[0]

    obs_st = [feature_st, sdf_st]
    obs_ns = [feature_ns, sdf_ns]
    obs_g = [feature_g, sdf_g]

    def get_a_prime():
        next_q = Q(obs_ns, obs_g, nb_ns, nb_g)
        if len(next_q)<=1:
            next_q[nb_ns:] = next_q.min()
        else:
            for i in range(len(next_q)):
                next_q[i, nb_ns[i]:] = next_q[i].min()
        obj = next_q.max(2)[0].max(1)[1]
        theta = next_q.max(1)[0].max(1)[1]
        return obj, theta

    a_prime = get_a_prime()
    next_q_target = Q_target(obs_ns, obs_g, nb_ns, nb_g)
    q_target_s_a_prime = next_q_target[torch.arange(batch_size), a_prime[0], a_prime[1]].unsqueeze(1)
    y_target = rewards + gamma * not_done * q_target_s_a_prime

    q_values = Q(obs_st, obs_g, nb_st, nb_g)
    pred = q_values[torch.arange(batch_size), actions[:, 0], actions[:, 1]]
    pred = pred.view(-1, 1)

    loss = criterion(y_target, pred)
    error = torch.abs(pred - y_target)
    return loss, error

