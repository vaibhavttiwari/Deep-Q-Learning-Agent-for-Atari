import tensorflow as tf
import numpy as np
import gym
import json
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from collections import deque
import random
import warnings
import click

from DQNetwork import DQN
from Memory import Memory

warnings.filterwarnings('ignore')

def preprocess_frame(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[8:-12,4:-12]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [110,84])
    
    return preprocessed_frame


def stack_frames(stacked_frames, state, is_new_episode, stack_size):
    frame = preprocess_frame(state)
    
    if is_new_episode:
        stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

@click.command()
@click.option('--to_train',
              type=click.BOOL,
              default=False,
              help='Whether you want to train or not.')
@click.option('--summary_dir',
              type=click.STRING,
              default="tensorboard/dqn",
              help='Location for saving the summary.')
@click.option('--ckpt_path',
              type=click.STRING,
              default="./models/model.ckpt",
              help='Location of checkpoint.')
@click.option('--env_name',
              type=click.STRING,
              default='SpaceInvaders-v0',
              help='The name of the Atari environment. Default : SpaceInvaders-v0')
@click.option('--render',
              type=click.BOOL,
              default=False,
              help='Enable playback or not.')
              
def main(to_train, summary_dir, ckpt_path, env_name, render):
    
    with open("config.json") as config_file:
        config = json.load(config_file)
   
    learning_rate =  config['learning_rate']
    total_episodes = config['total_episodes']
    max_steps = config['max_steps']
    batch_size = config['batch_size']
    explore_start = config['explore_start']
    explore_stop = config['explore_stop']
    decay_rate = config['decay_rate']
    gamma = config['gamma']
    pretrain_length = config['pretrain_length']
    memory_size = config['memory_size']
    stack_size = config['stack_size']

    env = gym.make(env_name)

    print("The size of our frame is: ", env.observation_space)
    print("The action size is : ", env.action_space.n)
    possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
    
    stacked_frames  =  deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
    state_size = [110, 84, 4]
    action_size = env.action_space.n
    
    training = to_train

    episode_render = render

    tf.reset_default_graph()

    DQNetwork = DQN(state_size, action_size, learning_rate)
    
    memory = Memory(max_size = memory_size)
    for i in range(pretrain_length):
    
        if i == 0:
            state = env.reset()        
            state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)

        choice = random.randint(1,len(possible_actions))-1
        action = possible_actions[choice]
        next_state, reward, done, _ = env.step(choice)
    
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)
        
        if done:    
            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state, done))
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)
        
        else:
            memory.add((state, action, reward, next_state, done))
            state = next_state
            
    writer = tf.summary.FileWriter(summary_dir)
    tf.summary.scalar("Loss", DQNetwork.loss)
    write_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    if training == True:
        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            decay_step = 0
        
            for episode in range(total_episodes):
                step = 0
                episode_rewards = []
            
                state = env.reset()
          
                state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)
            
                while step < max_steps:
                    step += 1      
                    decay_step +=1
                    exp_exp_tradeoff = np.random.rand()

                    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
                    if (explore_probability > exp_exp_tradeoff):        
                        choice = random.randint(1,len(possible_actions))-1
                        action = possible_actions[choice]
        
                    else:
                        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
                        choice = np.argmax(Qs)
                        action = possible_actions[choice]
                
                    next_state, reward, done, _ = env.step(choice)
                
                    if episode_render:
                        env.render()
                
                    episode_rewards.append(reward)
                
                    if done:
                        next_state = np.zeros((110,84), dtype=np.int)
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)
                        step = max_steps
                        total_reward = np.sum(episode_rewards)

                        print('Episode: {}'.format(episode),
                                  'Total reward: {}'.format(total_reward),
                                  'Explore P: {:.4f}'.format(explore_probability),
                                'Training Loss {:.4f}'.format(loss))

                        memory.add((state, action, reward, next_state, done))

                    else:                    
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)
                        memory.add((state, action, reward, next_state, done))
                        state = next_state
                        
                    batch = memory.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch]) 
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []

                    Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                                
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]
    
                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])
                        
                        else:
                            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)
                        

                    targets_mb = np.array([each for each in target_Qs_batch])

                    loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                        feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb})

                
                    summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                       DQNetwork.target_Q: targets_mb,
                                                       DQNetwork.actions_: actions_mb})

                    writer.add_summary(summary, episode)
                    writer.flush()
            
                if episode % 5 == 0:
                    save_path = saver.save(sess, ckpt_path)
                    print("Model Saved")
                
    with tf.Session() as sess:
        total_test_rewards = []    
        saver.restore(sess, ckpt_path)
    
        for episode in range(1):
            total_rewards = 0
        
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)
        
            print("EPISODE ", episode)
        
            while True:
                state = state.reshape((1, *state_size))
                Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state})
                choice = np.argmax(Qs)
                action = possible_actions[choice]
            
                next_state, reward, done, _ = env.step(choice)
                env.render()
            
                total_rewards += reward

                if done:
                    print ("Score", total_rewards)
                    total_test_rewards.append(total_rewards)
                    break
                      
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)
                state = next_state
            
    env.close()
        
if __name__ == '__main__':
    main()
