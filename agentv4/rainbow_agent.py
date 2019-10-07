#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.

Refer to :  anyrl-py/anyrl/algos/dqn.py

"""

import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from sonic_util import AllowBacktracking, make_env

def main():
    """Run DQN until the environment throws an exception."""
    env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        
        """
        Create a TF Op that optimizes the objective.
        Args:
          learning_rate: the Adam learning rate.
          epsilon: the Adam epsilon.
        """
        optimize = dqn.optimize(learning_rate=6.25e-5,epsilon=1.5e-4)

        sess.run(tf.global_variables_initializer())
        
        """
        Run an automated training loop.
        This is meant to provide a convenient way to run a
        standard training loop without any modifications.
        You may get more flexibility by writing your own
        training loop.
        Args:
          num_steps: the number of timesteps to run.
          player: the Player for gathering experience.
          replay_buffer: the ReplayBuffer for experience.
          optimize_op: a TF Op to optimize the model.
          train_interval: timesteps per training step.
          target_interval: number of timesteps between
            target network updates.
          batch_size: the size of experience mini-batches.
          min_buffer_size: minimum replay buffer size
            before training is performed.
          tf_schedules: a sequence of TFSchedules that are
            updated with the number of steps taken.
          handle_ep: called with information about every
            completed episode.
          timeout: if set, this is a number of seconds
            after which the training loop should exit.
        """
        dqn.train(num_steps=1000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)
        

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
