import numpy as np

def reward_touch(self, info):
    if info['collision']:
        return -0.5, False
    pre_poses = np.array(info['pre_poses'])
    poses = np.array(info['poses'])
    if np.linalg.norm(pre_poses - poses) > 1e-3:
        reward = 1.0
        done = False
    else:
        reward = -self.time_penalty
        done = False
    return reward, done

def reward_targetpush(self, info):
    if info['collision']:
        return -0.5, False
    target_idx = list(info['obj_indices']).index(info['target_obj'])
    poses = np.array(info['poses'])
    check_near = np.linalg.norm(poses, axis=1) < 0.05
    info['success'] = False
    if check_near.any():
        done = True
        reached_idx = list(check_near).index(True)
        if reached_idx==target_idx:
            reward = 1.0
            info['success'] = True
        else:
            reward = 0.0
        reached_obj = info['obj_indices'][reached_idx]
        info['reached_obj'] = reached_obj
    else:
        reward = -self.time_penalty
        done = False
        info['reached_obj'] = -1
    return reward, done

def reward_reach(self):
    target_pos = self.env.sim.data.get_body_xpos('target_body_1')
    if np.linalg.norm(target_pos - self.pre_target_pos) > 1e-3:
        reward = 1.0
        done = False #True
    else:
        reward = -self.time_penalty
        done = False
    return reward, done


###################################################################

def reward_push_sparse(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']
    collision = info['collision']
    oor = info['out_of_range']

    reward = 0.0
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            if pre_dist > self.threshold and dist < self.threshold:
                reward += 1.0
            elif pre_dist < self.threshold and dist > self.threshold:
                reward -= 1.0
        success.append(dist<self.threshold)
    success = np.array(success)

    reward -= self.time_penalty
    done = np.array(success).all()
    if success.all():
        reward = 10.0
        done = True
    elif oor:
        reward = -2.0
        done = True
    elif collision:
        reward = -0.5
    return reward, done, success


def reward_push_linear(self, info):
    reward_scale = 30 #100
    min_reward = -1
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']
    collision = info['collision']
    oor = info['out_of_range']

    reward = 0.0
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            reward += reward_scale * (pre_dist - dist)
            #if dist < self.threshold:
            #    reward += 1.0
        success.append(dist<self.threshold)
    success = np.array(success)

    reward -= self.time_penalty
    reward = max(reward, min_reward)
    done = np.array(success).all()
    if success.all():
        reward = 10.0
        done = True
    elif oor:
        reward = -2.0 #-1.0
        done = True
    elif collision:
        reward = -0.5
    return reward, done, success

def reward_push_linear_penalty(self, info):
    reward_scale = 30 #100
    min_reward = -1
    nb = info['num_blocks']
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']
    collision = info['collision']
    oor = info['out_of_range']

    reward = 0.0
    dist = np.linalg.norm(poses.reshape([1, nb, 2]) - goals.reshape([nb, 1, 2]), axis=2)
    pre_dist = np.linalg.norm(pre_poses.reshape([1, nb, 2]) - goals.reshape([nb, 1, 2]), axis=2)
    diff = pre_dist - dist

    weight_mat = (1 + nb/10) * np.eye(nb) - 1/10 * np.ones([nb, nb])
    diff_weighted = weight_mat * diff
    if target!=-1:
        diff_weighted = diff_weighted[target]

    reward = reward_scale * np.sum(diff_weighted)
    reward -= self.time_penalty
    reward = max(reward, min_reward)

    paired_dist = (dist*np.eye(nb)).sum(1)
    success = (paired_dist < self.threshold)
    done = np.array(success).all()
    if success.all():
        reward = 10.0
        done = True
    if oor:
        reward = -2.0 #-1.0
        done = True
    return reward, done, success

def reward_push_linear_maskpenalty(self, info):
    reward_scale = 30 #100
    min_reward = -2
    nb = info['num_blocks']
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    collision = info['collision']
    oor = info['out_of_range']

    reward = 0.0
    dist = np.linalg.norm(poses.reshape([1, nb, 2]) - goals.reshape([nb, 1, 2]), axis=2)
    pre_dist = np.linalg.norm(pre_poses.reshape([1, nb, 2]) - goals.reshape([nb, 1, 2]), axis=2)
    diff = pre_dist - dist

    threshold = 0.2
    mask_near = dist < threshold
    non_eye = np.ones([nb ,nb]) - np.eye(nb)
    mask_near = np.all([mask_near, non_eye], 0)
    weight_mat = np.eye(nb) - 1/5 * mask_near
    diff_weighted = weight_mat * diff

    reward = reward_scale * np.sum(diff_weighted)
    reward -= self.time_penalty
    reward = max(reward, min_reward)

    paired_dist = (dist*np.eye(nb)).sum(1)
    success = np.array(paired_dist < self.threshold)
    done = False
    if success.all():
        reward = 10.0
        done = True
    if oor:
        reward = -2.0 #-5.0
        done = True
    return reward, done, success

def reward_push_inverse(self, info):
    reward_scale = 20
    min_reward = -2
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']
    collision = info['collision']
    oor = info['out_of_range']

    reward = 0.0
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            reward += reward_scale * min(10, (1/dist - 1/pre_dist))
            if dist < self.threshold:
                reward += 1.0
        success.append(dist<self.threshold)

    reward -= self.time_penalty
    reward = max(reward, min_reward)
    done = np.array(success).all()
    if oor:
        reward = -1.0
        done = True
    elif collision:
        reward = -0.5
    return reward, done, success


def reward_push_binary(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    collision = info['collision']
    oor = info['out_of_range']

    reward = 0.0
    success = []
    for obj_idx in range(self.num_blocks):
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if dist < pre_dist - 0.001:
            reward += 1
        if dist > pre_dist + 0.001:
            reward -= 1
        success.append(dist<self.threshold)
    success = np.array(success)

    reward -= self.time_penalty
    done = np.array(success).all()
    if oor:
        reward = -1.0
        done = True
    elif collision:
        reward = -0.5
    return reward, done, success


def reward_push_new(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    contact = info['contact']
    collision = info['collision']
    oor = info['out_of_range']

    rewards = []
    pre_success = []
    success = []
    for obj_idx in range(self.num_blocks):
        reward = 0.0
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        success.append(dist < self.threshold)
        pre_success.append(pre_dist < self.threshold)
        if dist < pre_dist - 0.001:
            reward += 1.
        elif dist > pre_dist + 0.04: #0.001
            reward -= 0.5

        # reach reward
        # reward += 10. * (int(success[-1]) - int(pre_success[-1]))

        # block collision
        if contact[obj_idx]:
            reward -= 1.
        rewards.append(reward)
    success = np.array(success)
    reward = sum(rewards)

    # fail to touch
    if np.linalg.norm(poses - pre_poses) < 0.005:
        reward -= 0.1

    done = np.array(success).all()
    if oor:
        reward = -1.0
        done = True
    elif collision:
        reward = -1.0
    return reward, done, success

def reward_push_seg(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    contact = info['contact']
    collision = info['collision']
    oor = info['out_of_range']
    seg_target = info['seg_target']

    reward = 0.0
    pre_success = []
    success = []
    for obj_idx in range(self.num_blocks):
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        success.append(dist < self.threshold)
        pre_success.append(pre_dist < self.threshold)
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if obj_idx == seg_target or seg_target==-1:
            if dist < pre_dist - 0.001:
                reward += 1.
            elif dist > pre_dist + 0.04: #0.001
                reward -= 0.5
        else:
            if np.linalg.norm(poses[obj_idx] - pre_poses[obj_idx]) > 0.005:
                reward -= 1.

        # reach reward
        # reward += 10. * (int(success[-1]) - int(pre_success[-1]))

        # block collision
        if contact[obj_idx]:
            reward -= 1.

    # fail to touch
    if np.linalg.norm(poses - pre_poses) < 0.005:
        reward -= 0.1

    if seg_target==-1:
        done = np.array(success).all()
    else:
        done = success[seg_target]

    if oor:
        reward = -1.0
        done = True
    elif collision:
        reward = -1.0
    return reward, done, success

####################### seperate rewards ###############################

def reward_sparse_seperate(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']
    collision = info['collision']
    oor = info['out_of_range']

    rewards = []
    success = []
    for obj_idx in range(self.num_blocks):
        reward = 0.0
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            if dist < self.threshold:
                reward += 1.0
            if pre_dist < self.threshold and dist > self.threshold:
                reward -= 1.0
        reward -= self.time_penalty
        rewards.append(reward)
        success.append(dist<self.threshold)

    done = np.array(success).all()
    if oor:
        rewards = [-1.0] * self.num_blocks
        done = True
    elif collision:
        rewards = -np.ones(self.num_blocks)/self.num_blocks
    return rewards, done, success


def reward_linear_seperate(self, info):
    reward_scale = 20
    min_reward = -2
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']
    collision = info['collision']
    oor = info['out_of_range']

    rewards = []
    success = []
    for obj_idx in range(self.num_blocks):
        reward = 0.0
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            reward += reward_scale * np.clip(pre_dist - dist, -0.1, 0.1)
            if dist < self.threshold:
                reward += 1.0
            if pre_dist < self.threshold and dist > self.threshold:
                reward -= 1.0
        reward -= self.time_penalty
        reward = max(reward, min_reward)
        rewards.append(reward)
        success.append(dist<self.threshold)

    done = np.array(success).all()
    if oor:
        rewards = [-1.0] * self.num_blocks
        done = True
    elif collision:
        rewards = -np.ones(self.num_blocks)/self.num_blocks
    return rewards, done, success


def reward_inverse_seperate(self, info):
    reward_scale = 5
    min_reward = -2
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']
    collision = info['collision']
    oor = info['out_of_range']

    rewards = []
    success = []
    for obj_idx in range(self.num_blocks):
        reward = 0.0
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            reward += reward_scale * np.clip(1/(dist+0.1) - 1/(pre_dist+0.1), -2, 2)
            if dist < self.threshold:
                reward += 1.0
            if pre_dist < self.threshold and dist > self.threshold:
                reward -= 1.0
        reward -= self.time_penalty
        reward = max(reward, min_reward)
        rewards.append(reward)
        success.append(dist<self.threshold)

    done = np.array(success).all()
    if oor:
        rewards = [-1.0] * self.num_blocks
        done = True
    elif collision:
        rewards = -np.ones(self.num_blocks)/self.num_blocks
    return rewards, done, success


def reward_binary_seperate(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    target = info['target']
    collision = info['collision']
    oor = info['out_of_range']

    rewards = []
    success = []
    for obj_idx in range(self.num_blocks):
        reward = 0.0
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        if target==-1 or target==obj_idx:
            if dist < pre_dist - 0.001:
                reward += 1
            if dist > pre_dist + 0.001:
                reward -= 1
        reward -= self.time_penalty
        rewards.append(reward)
        success.append(dist<self.threshold)

    done = np.array(success).all()
    if oor:
        rewards = [-1.0] * self.num_blocks
        done = True
    elif collision:
        rewards = -np.ones(self.num_blocks)/self.num_blocks
    return rewards, done, success


def reward_new_seperate(self, info):
    goals = info['goals']
    poses = info['poses']
    pre_poses = info['pre_poses']
    contact = info['contact']
    target = info['target']
    collision = info['collision']
    oor = info['out_of_range']

    rewards = []
    pre_success = []
    success = []
    for obj_idx in range(self.num_blocks):
        reward = 0.0
        dist = np.linalg.norm(poses[obj_idx] - goals[obj_idx])
        pre_dist = np.linalg.norm(pre_poses[obj_idx] - goals[obj_idx])
        success.append(dist < self.threshold)
        pre_success.append(pre_dist < self.threshold)
        if target==-1 or target==obj_idx:
            if dist < pre_dist - 0.001:
                reward += 1.
            elif dist > pre_dist + 0.001:
                reward += 0.3
            else:
                reward -= 1./self.num_blocks

        # reach reward
        reward += 10. * (int(success[-1]) - int(pre_success[-1]))

        # block collision
        if contact[obj_idx]:
            reward -= 1
        rewards.append(reward)

    done = np.array(success).all()
    if oor:
        rewards = [-1.0] * self.num_blocks
        done = True
    elif collision:
        rewards = -np.ones(self.num_blocks)/self.num_blocks
    return rewards, done, success
