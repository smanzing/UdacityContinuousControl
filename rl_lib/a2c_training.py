import torch
import numpy as np
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_batch(env, brain_name, agent, nr_agents, t_max, init_states):
    batch_states = list()
    batch_actions = list()
    batch_actions_log_prob = list()
    batch_actions_entropy = list()
    batch_rewards = list()
    batch_dones = list()
    scores = np.zeros(nr_agents)

    states = init_states
    for t in range(t_max):
        actions, actions_log_prob, entropy = agent.act(states)
        env_info = env.step(actions)[brain_name]
        # get next state, reward, and check if episode has finished
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        batch_states.append(states)
        batch_actions.append(actions)
        batch_actions_log_prob.append(actions_log_prob)
        batch_actions_entropy.append(entropy)
        batch_rewards.append(np.array(rewards))
        batch_dones.append(np.array(dones))

        scores += rewards
        if np.any(dones):
            break
        states = next_states

    next_states = env_info.vector_observations

    return batch_states, next_states, batch_actions, batch_actions_log_prob, \
        batch_actions_entropy, batch_rewards, batch_dones, scores


def a2c_training(env, brain_name, agent, n_episodes=2000, t_min=10, update_t_every_n_episode=10, t_max=1000,
                 min_avg_score=13.0, continue_learning=False,
                 filename='checkpoint'):
    """
    @param env: unity environment
    @param brain_name: name of the brain that we control from Python
    @param agent: the RL agent that we train
    @param n_episodes: maximum number of training episodes
    @param t_min: minimum number of timesteps per episode
    @param update_t_every_n_episode: increase t by 1 every n episode
    @param t_max: maximum number of timesteps per episode
    @param min_avg_score: minimum average score over 100 episodes that the agent must achieve to consider the task fulfilled
    @param continue_learning: if true, the agent continues to learn after reaching min_avg_score until reaching n_episodes
    @param filename: name for the file that contains the trained network parameters
    @return:
    """

    scores = []  # list containing scores from each episode
    scores_window = deque([0], maxlen=100)  # last 100 scores
    min_score_achieved = False
    env_info = env.reset(train_mode=True)[brain_name]
    init_states = env_info.vector_observations
    nr_agents = len(env_info.agents)

    i_episode = 1
    accumulated_reward_per_episode = np.zeros(nr_agents)
    t = t_min
    while True:
        batch_states, next_states, batch_actions, batch_actions_log_prob, batch_actions_entropy, batch_rewards, batch_dones, batch_scores = collect_batch(
            env, brain_name, agent, nr_agents, t, init_states)
        agent.learn(batch_states, next_states, batch_actions_log_prob, batch_actions_entropy, batch_rewards, batch_dones)

        accumulated_reward_per_episode += batch_scores
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        init_states = next_states
        if np.any(batch_dones[-1]):
            env_info = env.reset(train_mode=True)[brain_name]
            init_states = env_info.vector_observations

            i_episode += 1

            average_score_episode = np.sum(accumulated_reward_per_episode) / nr_agents
            scores.append(average_score_episode)
            scores_window.append(average_score_episode)
            accumulated_reward_per_episode = np.zeros(nr_agents)

            if i_episode % update_t_every_n_episode == 0:
                t = min(t + 1, t_max)

            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= min_avg_score and not min_score_achieved:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             np.mean(scores_window)))
                agent.save_networks(filename + 'min_score_achieved_')
                min_score_achieved = True
                if not continue_learning:
                    break
        if i_episode == n_episodes:
            break

    agent.save_networks(filename + 'final_')
    return scores
