import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import sys
import os

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import maddpg.common.tf_util as U
from maddpg.trainer.m3ddpg import M3DDPGAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="mmmaddpg", help="policy for good agents")
    parser.add_argument("--bad-policy", type=str, default="mmmaddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    parser.add_argument("--adv-eps-s", type=float, default=1e-5, help="small adversarial training rate")
    parser.add_argument("--gpu-frac", type=float, default=0.3, help="Fraction of GPU memory usage.")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-name", type=str, default="", help="name of which training state and model are loaded, leave blank to load seperately")
    parser.add_argument("--load-good", type=str, default="", help="which good policy to load")
    parser.add_argument("--load-bad", type=str, default="", help="which bad policy to load")
    # Evaluation
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    # Add Noise for observation 
    parser.add_argument("--noise-type", type=int, default=0, help="type can be: 0-none, 1-truncated normal distribution, 2-normal distribution")
    parser.add_argument('--noise-std', type=float, default=1.0, help='{0.0, 1.0, 2.0, 3.0, ...}, noise standard deviation')
    parser.add_argument("--d-value", type=float, default=1.0, help="a radius denoting how large the perturbation set is")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = M3DDPGAgentTrainer
    for i in range(num_adversaries):
        print("{} bad agents".format(i))
        policy_name = arglist.bad_policy
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.observation_space, env.action_space, i, arglist,
            policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    for i in range(num_adversaries, env.n):
        print("{} good agents".format(i))
        policy_name = arglist.good_policy
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.observation_space, env.action_space, i, arglist,
            policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    return trainers


def train(arglist):
    if arglist.test:
        np.random.seed(71)
    with U.single_threaded_session(arglist.gpu_frac):
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and bad policy {} with {} adversaries'.format(arglist.good_policy, arglist.bad_policy, num_adversaries))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.test or arglist.display or arglist.restore or arglist.benchmark:
            if arglist.load_name == "":
                # load seperately
                bad_var_list = []
                for i in range(num_adversaries):
                    bad_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trainers[i].scope)
                saver = tf.train.Saver(bad_var_list)
                U.load_state(arglist.load_bad, saver)

                good_var_list = []
                for i in range(num_adversaries, env.n):
                    good_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trainers[i].scope)
                saver = tf.train.Saver(good_var_list)
                U.load_state(arglist.load_good, saver)
            else:
                print('Loading previous state from {}'.format(arglist.load_name))
                U.load_state(arglist.load_name)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            if not arglist.test:
                loss = None
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, global_step = len(episode_rewards), saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("{} vs {} steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(arglist.bad_policy, arglist.good_policy,
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                suffix = '_test.pkl' if arglist.test else '.pkl'
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards' + suffix
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards' + suffix

                if not os.path.exists(os.path.dirname(rew_file_name)):
                    try:
                        os.makedirs(os.path.dirname(rew_file_name))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise

                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
