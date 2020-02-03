#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from s_q_agent import s_q_agent
from approximate_sarsa import approximate_sarsa_agent
from supply_distribution import SupplyDistribution
from reinforce import REINFORCE_agent

# ========================= 0. Function definitions ========================= #


def print_step(step, state, action, reward, state_new, total_reward, freq = 100):
    print("========= Step: %3s =========" % step)
    print("State:         ", state)
    print("Action:        ", action)
    print("Reward:        ", round(reward, 2))
    print("Next state:    ", state_new)
    print("Episode reward:", round(total_reward,2))


def evaluate_agent(agent, env, n_episodes, max_steps, output, status_freq, print_freq, log_freq, results_folder_path, result_file_name):

    # Set seed
    np.random.seed(10108)

    # Initialize array with rewards
    rewards = np.zeros(n_episodes)
    stocks = np.zeros((int(n_episodes / log_freq), max_steps, env.n_stores+1))
    demands = np.zeros((int(n_episodes / log_freq), max_steps, env.n_stores))
    actions = np.zeros((int(n_episodes / log_freq), max_steps, env.n_stores+1))
    reward_log = np.zeros(int(n_episodes / log_freq))
    maxreward = -100000 # save the maximum reward episode

    current_stocks = np.zeros(( max_steps+1, env.n_stores+1))
    current_demands = np.zeros(( max_steps+1, env.n_stores))
    current_actions = np.zeros(( max_steps+1, env.n_stores+1))
    # Simulate n_episodes with max_steps each
    for episode in np.arange(n_episodes):
        # Reset environment, select initial action and reset episode reward
        state = env.reset()
        action = agent.get_action(state)
        episode_reward = 0

        # save some stuff
        current_stocks[0,:] = env.s
        current_demands[0,:] = env.demand
        current_actions[0,:] = action

        for step in np.arange(max_steps):

            # Update environment
            state_new, reward, done, info = env.step(action)

            # Select a new action
            action_new = agent.get_action(state_new)

            # Update episode reward
            episode_reward += np.power(env.gamma, step) * reward

            # Update agent
            agent.update(state, action, reward, state_new, action_new)

            # Print information
            if output and episode % print_freq == 0:
                print_step(step, state, action, reward, state_new, episode_reward,print_freq)

            # Log environment
            current_stocks[step,:] = env.s
            current_demands[step,:] = env.demand
            current_actions[step,:] = action_new

            if (episode+1) % log_freq == 0:
                stocks[int((episode+1) / log_freq - 1), step, :] = env.s
                demands[int((episode+1) / log_freq - 1), step, :] = env.demand
                actions[int((episode+1) / log_freq - 1), step, :] = action_new
                reward_log[int((episode+1) / log_freq - 1)] = episode_reward
            # Update state
            state = state_new
            action = action_new

        # Add episodes reward to rewards list
        rewards[episode] = episode_reward

        # Save for best reward actions
        if episode_reward >= max(episode_reward, maxreward):
            best_stocks = current_stocks.copy()
            best_demands= current_demands.copy()
            best_actions= current_actions.copy()
            maxreward = episode_reward

        # Print number current episode each 100 episodes
        if episode % status_freq == 0:
            print("Episode ", episode," Reward: ", episode_reward)

    # ============================ 3. Output results ============================ #

    # Output results
    print("Average reward: ", round(np.mean(rewards),2))

    # Save all data:
    pd.DataFrame(best_stocks).to_csv(results_folder_path + result_file_name + "_best_stocks.csv", header=None, index=None)
    pd.DataFrame(best_demands).to_csv(results_folder_path + result_file_name + "_best_demands.csv", header=None, index=None)
    pd.DataFrame(best_actions).to_csv(results_folder_path + result_file_name + "_best_actions.csv", header=None, index=None)
    pd.DataFrame(rewards).to_csv(results_folder_path + result_file_name + "_rewards.csv", header=None, index=None)
    pd.DataFrame(current_stocks).to_csv(results_folder_path + result_file_name + "_last_stock.csv", header=None, index=None)
    pd.DataFrame(current_demands).to_csv(results_folder_path + result_file_name + "_last_demand.csv", header=None, index=None)
    pd.DataFrame(current_actions).to_csv(results_folder_path + result_file_name + "_last_actions.csv", header=None, index=None)

    data_sets = np.array([])
    try:
        data_sets = pd.read_csv(results_folder_path + "data_sets.csv", header=None).values.flatten()
    except:
        data_sets = np.array([])
        print("files file not created, creating it")
    finally:
        pd.DataFrame(np.unique(np.append(data_sets, result_file_name))).to_csv(results_folder_path + "data_sets.csv", header=None, index=None)
        print(pd.read_csv(results_folder_path + "data_sets.csv", header=None).values.flatten())



