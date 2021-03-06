import argparse
import cv2
import gym
import copy
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lightsaber.rl.replay_buffer import EpisodeReplayBuffer
from lightsaber.rl.trainer import Trainer
from lightsaber.rl.env_wrapper import EnvWrapper
from lightsaber.tensorflow.log import TfBoardLogger
from network import make_actor_network, make_critic_network
from agent import Agent
from datetime import datetime
# Credit: https://github.com/takuseno/rsvg

def main():
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--log', type=str, default=date)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-steps', type=int, default=10 ** 5)
    parser.add_argument('--episode-update', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/{}'.format(args.log))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logdir = os.path.join(os.path.dirname(__file__), 'logs/{}'.format(args.log))

    env = EnvWrapper(
        env=gym.make(args.env),
        r_preprocess=lambda r: r / 10.0
    )

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    sess = tf.Session()
    sess.__enter__()

    actor = make_actor_network([64, 64])
    critic = make_critic_network([64, 64])
    replay_buffer = EpisodeReplayBuffer(10 ** 3)

    agent = Agent(
        actor, critic, obs_dim, n_actions, replay_buffer,
        episode_update=args.episode_update)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    train_writer = tf.summary.FileWriter(logdir, sess.graph)
    tflogger = TfBoardLogger(train_writer)
    tflogger.register('reward', dtype=tf.float32)

    end_episode = lambda r, s, e: tflogger.plot('reward', r, s)
    def after_action(state, reward, global_step, local_step):
        if global_step > 0 and global_step % 10 * 5 == 0:
            path = os.path.join(outdir, 'model.ckpt')
            saver.save(sess, path, global_step=global_step)

    trainer = Trainer(
        env=env,
        agent=agent,
        render=args.render,
        state_shape=[obs_dim],
        state_window=1,
        final_step=args.final_steps,
        end_episode=end_episode,
        after_action=after_action,
        training=not args.demo
    )

    N = 20
    T = 500
    rewards = np.zeros((N,T))
    env_gym = gym.make(args.env)
    for i in range(N):
        state = env_gym.reset()
        for step_index in range(T):
            a = env_gym.action_space.sample()
            state, r, done, _ = env_gym.step(np.array(a).reshape(-1))
            rewards[i,step_index] = r
            if done:
                print("Finished after iteration: ", step_index)
                break
    rewards = np.mean(rewards, axis=0)
    trainer.start()
    #
    # print('Rendering RSVG trained agent')
    # rewardRSVG = np.zeros((N,T))
    # trainer.training = False
    # trainer.render = True
    # n_envs = trainer.env.get_num_of_envs()
    # for i in range(N):
    #     state = trainer.env.reset()
    #     for step_index in range(T):
    #         for i in range(n_envs):
    #             trainer.before_action_callback(
    #                 state[i],
    #                 trainer.global_step,
    #                 trainer.local_step[i]
    #             )
    #         state, reward, done, info = trainer.move_to_next(state, None, None)
    #         for i in range(n_envs):
    #             trainer.after_action_callback(
    #                 state[i],
    #                 trainer.global_step,
    #                 trainer.local_step[i]
    #             )
    #         rewardRSVG[i,step_index] = reward
    #         if done:
    #             print("Finished after iteration: ", step_index)
    #             break
    #
    # rewardRSVG = np.mean(rewardRSVG, axis=0)
    # print(trainer.is_training_finished())
    # plt.plot(np.cumsum(rewardRSVG), color='green', label='RSVG Agent trained')
    # # plt.plot(np.cumsum(rewards), color='red', label='Untrained Agent')
    # plt.legend()
    # plt.savefig('rewards_trained_rsvg.png')
    # plt.show()
    # plt.clf()

if __name__ == '__main__':
    main()
