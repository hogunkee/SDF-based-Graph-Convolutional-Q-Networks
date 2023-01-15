import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, 'ur5_mujoco'))
from object_env import *

from utils.training_utils import *
from utils.sdf_module import SDFModule
from utils.replay_buffer import ReplayBuffer, PER

import torch
import torch.nn as nn
import argparse
import json

import copy
import time
import datetime
import random
import pylab

from matplotlib import pyplot as plt
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_npy(array):
    positive = array - array.min()
    return positive / positive.max()

def pad_sdf(sdf, nmax, res=96):
    nsdf = len(sdf)
    padded = np.zeros([nmax, res, res])
    if nsdf > nmax:
        padded[:] = sdf[:nmax]
    elif nsdf > 0:
        padded[:nsdf] = sdf
    return padded

def get_action(env, max_blocks, depth, sdf_raw, sdfs, epsilon, with_q=False, sdf_action=False, target_res=96):

    if np.random.random() < epsilon:
        #print('Random action')
        obj = np.random.randint(len(sdf_raw))
        theta = np.random.randint(env.num_bins)
    else:
        nsdf = sdfs[0].shape[0]
        s = pad_sdf(sdfs[0], max_blocks, target_res)
        g = pad_sdf(sdfs[1], max_blocks, target_res)
        nonempty = np.where(np.sum(s, (1,2))!=0)[0]

        check_reach = True
        for _ in range(50):
            obj = np.random.choice(nonempty)
            sx, sy = env.get_center_from_sdf(s[obj], depth)
            gx, gy = env.get_center_from_sdf(g[obj], depth)
            check_reach = (np.linalg.norm([sx-gx, sy-gy])<0.05)
            if not check_reach:
                break
        theta = np.arctan2(gx-sx, gy-sy)
        theta = np.round(theta / np.pi / 0.25) % 8

    action = [obj, theta]
    sdf_target = sdf_raw[obj]
    cx, cy = env.get_center_from_sdf(sdf_target, depth)
    mask = None
    if sdf_action:
        masks = []
        for s in sdf_raw:
            m = copy.deepcopy(s)
            m[m<0] = 0
            m[m>0] = 1
            masks.append(m)
        mask = np.sum(masks, 0)

    print(action)
    if with_q:
        return action, [cx, cy, theta], mask, None
    else:
        return action, [cx, cy, theta], mask

def evaluate(env,
        sdf_module,
        n_actions=8,
        num_trials=100,
        visualize_q=False,
        clip_sdf=False,
        sdf_action=False,
        max_blocks=5,
        oracle_matching=False,
        round_sdf=False,
        tracker=False, 
        segmentation=False,
        scenario=-1
        ):

    if sdf_module.resize:
        sdf_res = 96
    else:
        sdf_res = 480

    log_returns = []
    log_eplen = []
    log_out = []
    log_success = []
    log_distance = []
    log_success_block = [[] for i in range(env.num_blocks)]

    if visualize_q:
        plt.rc('axes', labelsize=6)
        plt.rc('font', size=8)

        cm = pylab.get_cmap('gist_rainbow')
        fig = plt.figure(figsize=(6,5))
        ax0 = fig.add_subplot(221)
        ax1 = fig.add_subplot(222)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(224)
        ax0.set_title('Goal')
        ax1.set_title('Observation')
        ax2.set_title('Goal SDF')
        ax3.set_title('Current SDF')
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])

        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

    epsilon = 0.1
    for ne in range(num_trials):
        ep_len = 0
        episode_reward = 0.

        check_env_ready = False
        while not check_env_ready:
            (state_img, goal_img), info = env.reset(scenario=scenario)
            if segmentation:
                sdf_st, sdf_raw, feature_st = sdf_module.get_seg_features_with_ucn(state_img[0], state_img[1], env.num_blocks, clip=clip_sdf)
                sdf_g, _, feature_g = sdf_module.get_seg_features_with_ucn(goal_img[0], goal_img[1], env.num_blocks, clip=clip_sdf)
            else:
                sdf_st, sdf_raw, feature_st = sdf_module.get_sdf_features_with_ucn(state_img[0], state_img[1], env.num_blocks, clip=clip_sdf)
                sdf_g, _, feature_g = sdf_module.get_sdf_features_with_ucn(goal_img[0], goal_img[1], env.num_blocks, clip=clip_sdf)
            sdfimg = ((sdf_g > 0).astype(int)*255).astype(np.uint8).transpose([1,2,0])
            if round_sdf:
                sdf_g = sdf_module.make_round_sdf(sdf_g)
            check_env_ready = (len(sdf_g)==env.num_blocks) & (len(sdf_st)==env.num_blocks)

        n_detection = len(sdf_st)
        # target: st / source: g
        if oracle_matching:
            sdf_st_align = sdf_module.oracle_align(sdf_st, info['pixel_poses'])
            sdf_raw = sdf_module.oracle_align(sdf_raw, info['pixel_poses'], scale=1)
            sdf_g = sdf_module.oracle_align(sdf_g, info['pixel_goals'])
        else:
            matching = sdf_module.object_matching(feature_st, feature_g)
            sdf_st_align = sdf_module.align_sdf(matching, sdf_st, sdf_g)
            sdf_raw = sdf_module.align_sdf(matching, sdf_raw, np.zeros([env.num_blocks, *sdf_raw.shape[1:]]))

        masks = []
        for s in sdf_raw:
            masks.append(s>0)
        sdf_module.init_tracker(state_img[0], masks)

        if visualize_q:
            if env.env.camera_depth:
                ax0.imshow(goal_img[0])
                ax1.imshow(state_img[0])
            else:
                ax0.imshow(goal_img)
                ax1.imshow(state_img)
            # goal sdfs
            vis_g = norm_npy(0*sdf_g + 2*(sdf_g>0).astype(float))
            goal_sdfs = np.zeros([sdf_res, sdf_res, 3])
            for _s in range(len(vis_g)):
                goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
            ax2.imshow(norm_npy(goal_sdfs))
            # current sdfs
            vis_c = norm_npy(0*sdf_st_align + 2*(sdf_st_align>0).astype(float))
            current_sdfs = np.zeros([sdf_res, sdf_res, 3])
            for _s in range(len(vis_c)):
                current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
            ax3.imshow(norm_npy(current_sdfs))
            fig.canvas.draw()
            fig.canvas.flush_events()

        for t_step in range(env.max_steps):
            ep_len += 1
            action, pose_action, sdf_mask, q_map = get_action(env, max_blocks, \
                    state_img[1], sdf_raw, [sdf_st_align, sdf_g], epsilon=epsilon,  \
                    with_q=True, sdf_action=sdf_action, target_res=sdf_res)

            (next_state_img, _), reward, done, info = env.step(pose_action, sdf_mask)

            if tracker:
                if segmentation:
                    sdf_ns, sdf_raw, feature_ns = sdf_module.get_seg_features(next_state_img[0], next_state_img[1], env.num_blocks, clip=clip_sdf)
                else:
                    sdf_ns, sdf_raw, feature_ns = sdf_module.get_sdf_features(next_state_img[0], next_state_img[1], env.num_blocks, clip=clip_sdf)
            else:
                if segmentation:
                    sdf_ns, sdf_raw, feature_ns = sdf_module.get_seg_features_with_ucn(next_state_img[0], next_state_img[1], env.num_blocks, clip=clip_sdf)
                else:
                    sdf_ns, sdf_raw, feature_ns = sdf_module.get_sdf_features_with_ucn(next_state_img[0], next_state_img[1], env.num_blocks, clip=clip_sdf)
            pre_n_detection = n_detection
            n_detection = len(sdf_ns)
            if oracle_matching:
                sdf_ns_align = sdf_module.oracle_align(sdf_ns, info['pixel_poses'])
                sdf_raw = sdf_module.oracle_align(sdf_raw, info['pixel_poses'], scale=1)
            else:
                matching = sdf_module.object_matching(feature_ns, feature_g)
                sdf_ns_align = sdf_module.align_sdf(matching, sdf_ns, sdf_g)
                sdf_raw = sdf_module.align_sdf(matching, sdf_raw, np.zeros([env.num_blocks, *sdf_raw.shape[1:]]))

            # sdf reward #
            reward += sdf_module.add_sdf_reward(sdf_st_align, sdf_ns_align, sdf_g)
            episode_reward += reward

            # detection failed #
            if n_detection == 0:
                done = True

            if info['block_success'].all():
                info['success'] = True
            else:
                info['success'] = False

            if visualize_q:
                if env.env.camera_depth:
                    ax1.imshow(next_state_img[0])
                else:
                    ax1.imshow(next_state_img)

                # goal sdfs
                vis_g = norm_npy(0*sdf_g + 2*(sdf_g>0).astype(float))
                goal_sdfs = np.zeros([sdf_res, sdf_res, 3])
                for _s in range(len(vis_g)):
                    goal_sdfs += np.expand_dims(vis_g[_s], 2) * np.array(cm(_s/5)[:3])
                ax2.imshow(norm_npy(goal_sdfs))
                # current sdfs
                vis_c = norm_npy(0*sdf_ns_align + 2*(sdf_ns_align>0).astype(float))
                current_sdfs = np.zeros([sdf_res, sdf_res, 3])
                for _s in range(len(vis_c)):
                    current_sdfs += np.expand_dims(vis_c[_s], 2) * np.array(cm(_s/5)[:3])
                ax3.imshow(norm_npy(current_sdfs))
                fig.canvas.draw()
                fig.canvas.flush_events()

            if done:
                break
            else:
                sdf_st_align = sdf_ns_align

        log_returns.append(episode_reward)
        log_eplen.append(ep_len)
        log_out.append(int(info['out_of_range']))
        log_success.append(int(info['success']))
        log_distance.append(info['dist'])
        for o in range(env.num_blocks):
            log_success_block[o].append(int(info['block_success'][o]))# and sdf_success[o]))

        print("EP{}".format(ne+1), end=" / ")
        print("reward:{0:.2f}".format(log_returns[-1]), end=" / ")
        print("eplen:{0:.1f}".format(log_eplen[-1]), end=" / ")
        print("SR:{0:.2f} ({1}/{2})".format(np.mean(log_success),
                np.sum(log_success), len(log_success)), end=" / ")
        for o in range(env.num_blocks):
            print("B{0}:{1:.2f}".format(o+1, np.mean(log_success_block[o])), end=" ")

        #print(" / mean eplen:{0:.1f}".format(np.mean(log_eplen)), end="")
        #print(" / mean error:{0:.1f}".format(np.mean(log_distance)*1e3), end="")
        dist_success = np.array(log_distance)[np.array(log_success)==1] * 1e3 #scale: mm
        print(" / mean error:{0:.1f}".format(np.mean(dist_success)), end="")
        eplen_success = np.array(log_eplen)[np.array(log_success)==1]
        print(" / mean eplen:{0:.1f}".format(np.mean(eplen_success)), end="")
        print(" / oor:{0:.2f}".format(np.mean(log_out)), end="")
        print(" / mean reward:{0:.1f}".format(np.mean(log_returns)))

    print()
    print("="*80)
    print("Evaluation Done.")
    print("Success rate: {}".format(100*np.mean(log_success)))
    #print("Mean episode length: {}".format(np.mean(log_eplen)))
    #print("Mean error: {0:.1f}".format(np.mean(log_distance) * 1e3))
    dist_success = np.array(log_distance)[np.array(log_success)==1] * 1e3 #scale: mm
    print("Error-success: {0:.1f}".format(np.mean(dist_success)))
    eplen_success = np.array(log_eplen)[np.array(log_success)==1]
    print("Mean episode length: {}".format(np.mean(eplen_success)))
    print("Out of range: {}".format(np.mean(log_out)))
    print("Mean reward: {0:.2f}".format(np.mean(log_returns)))
    for o in range(env.num_blocks):
        print("Block {}: {}% ({}/{})".format(o+1, 100*np.mean(log_success_block[o]), np.sum(log_success_block[o]), len(log_success_block[o])))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # env config #
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument("--max_blocks", default=8, type=int)
    parser.add_argument("--threshold", default=0.10, type=float)
    parser.add_argument("--real_object", action="store_false")
    parser.add_argument("--dataset", default="test", type=str)
    parser.add_argument("--max_steps", default=100, type=int)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--scenario", default=-1, type=int)
    # sdf #
    parser.add_argument("--oracle", action="store_true")
    # etc #
    parser.add_argument("--num_trials", default=100, type=int)
    parser.add_argument("--show_q", action="store_true")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    args = parser.parse_args()

    # random seed #
    seed = args.seed
    if seed is not None:
        print("Random seed:", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # env configuration #
    render = args.render
    num_blocks = args.num_blocks
    max_blocks = args.max_blocks
    real_object = args.real_object
    dataset = args.dataset
    threshold = args.threshold
    max_steps = args.max_steps
    gpu = args.gpu
    small = args.small
    scenario = args.scenario

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)

    # evaluate configuration
    num_trials = args.num_trials

    # model configuration
    resize = True
    clip_sdf = False
    round_sdf = False
    sdf_action = True
    depth = False
    mov_dist = 0.06
    camera_height = 480
    camera_width = 480
    tracker = True
    convex_hull = False
    reward_type = 'linear_penalty'

    visualize_q = args.show_q
    oracle_matching = True #args.oracle
    sdf_module = SDFModule(rgb_feature=True, resnet_feature=True, convex_hull=convex_hull, 
            binary_hole=True, using_depth=depth, tracker='medianflow', resize=resize)
    if real_object:
        from realobjects_env import UR5Env
    else:
        from ur5_env import UR5Env
    env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, dataset=dataset,\
            small=small)
    env = objectwise_env(env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
            threshold=threshold, conti=False, detection=True, reward_type=reward_type, \
            delta_action=False)

    evaluate(env=env, sdf_module=sdf_module, n_actions=8, num_trials=num_trials, \
            visualize_q=visualize_q, clip_sdf=False, sdf_action=True, max_blocks=max_blocks, \
            oracle_matching=oracle_matching, round_sdf=round_sdf, tracker=tracker, \
            segmentation=False, scenario=scenario)
