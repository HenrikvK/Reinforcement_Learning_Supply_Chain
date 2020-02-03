#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from s_q_agent import s_q_agent
from approximate_sarsa import approximate_sarsa_agent
from supply_distribution import SupplyDistribution
from reinforce import REINFORCE_agent
from evaluate_agent import evaluate_agent
from evaluate_agent import print_step

###
# Testing file.
# To run a test, append the environment to the "environments" list and append the test name to env_names list
# Then append a boolean variable to test_to_run. True to run the test, False if not.
# There are already 10 environment. Look at them to understand how to add them a which parameters to use
# To choose an agents, set the agent boolean variables of the agents to run to true
# The results will be added in the folder at the path "results_folder_path"
# This results will be loaded from the graph_creation.py file
###

# results path name
results_folder_path = "../results/"

# Simulation parameters
n_episodes = 15000
max_steps = 24 # 24 Months

# Visualization parameters
output=1
status_freq = int(n_episodes/100) # Print status (current episode) every X episodes
print_freq = int(n_episodes/5) # Print current step every X episodes
log_freq = int(n_episodes / 10) # Helper variable for when

environments = []
env_names = []

# Agent Variables
add_s_q = True
add_sarsa = True
add_reinforce_1 = True  # Linear
add_reinforce_2 = True  # Quadratic
add_reinforce_3 = True  # RBF

# Simple2,3,4, Medium,2,3,4, weird,2, Difficult
test_to_run = [False, False, False, True, False, False, False, False, True, False, False]

# Instantiate environment
environments.append(SupplyDistribution(n_stores=1,
                                    cap_truck=3,
                                    prod_cost=1,
                                    max_prod=2,
                                    store_cost=np.array([0, 0]),
                                    truck_cost=np.array([0]),
                                    cap_store=np.array([30, 10]),
                                    penalty_cost=1,
                                    price=1,
                                    gamma=1,
                                    max_demand = 4,
                                    episode_length = max_steps))
env_names.append("simple_environment_2")

environments.append(SupplyDistribution(n_stores=1,
                         cap_truck=4,
                         prod_cost=0.5,
                         max_prod=1,
                         store_cost=np.array([0, 0.1]),
                         truck_cost=np.array([0]),
                         cap_store=np.array([10, 10]),
                         penalty_cost=1,
                         price=1,
                         gamma=1,
                         max_demand = 3,
                         episode_length = max_steps))
env_names.append("simple_environment_3")

environments.append(SupplyDistribution(n_stores=1,
                         cap_truck=2,
                         prod_cost=0.5,
                         max_prod=1,
                         store_cost=np.array([0, 0.1]),
                         truck_cost=np.array([0]),
                         cap_store=np.array([10, 10]),
                         penalty_cost=1,
                         price=1,
                         gamma=1,
                         max_demand = 3,
                         episode_length = max_steps))
env_names.append("simple_environment_4")

environments.append(SupplyDistribution(n_stores=1,
                         cap_truck=3,
                         prod_cost=1,
                         max_prod=3,
                         store_cost=np.array([0, 0.1]),
                         truck_cost=np.array([1]),
                         cap_store=np.array([50, 10]),
                         penalty_cost=1,
                         price=2,
                         gamma=1,
                         max_demand = 5,
                         episode_length = max_steps))
env_names.append("medium_environment")

environments.append(SupplyDistribution(n_stores=1,
                                       cap_truck=3,
                                       prod_cost=1,
                                       max_prod=2,
                                       store_cost=np.array([0, 1]),
                                       truck_cost=np.array([3]),
                                       cap_store=np.array([30, 10]),
                                       penalty_cost=1,
                                       price=1,
                                       gamma=1,
                                       max_demand = 4,
                                       episode_length = max_steps))
env_names.append("medium_environment_2")

environments.append(SupplyDistribution(n_stores=3,
                         cap_truck=3,
                         prod_cost=1,
                         max_prod=6,
                         store_cost=np.array([0, 0.4, 0.5, 0.6]),
                         truck_cost=np.array([2, 3, 4]),
                         cap_store=np.array([30, 6, 6, 6]),
                         penalty_cost=1,
                         price=3,
                         gamma=1,
                         max_demand = 4,
                         episode_length = max_steps))
env_names.append("medium_environment_3stores")

environments.append(SupplyDistribution(n_stores=3,
                         cap_truck=2,
                         prod_cost=0.5,
                         max_prod=5,
                         store_cost=np.array([0, 0.1, 0.1, 0.1]),
                         truck_cost=np.array([0, 0, 0]),
                         cap_store=np.array([30, 10, 10, 10]),
                         penalty_cost=1,
                         price=1,
                         gamma=1,
                         max_demand = 3,
                         episode_length = max_steps))
env_names.append("medium_environment_3stores2")

environments.append(SupplyDistribution(n_stores=3,
                                       cap_truck=3,
                                       prod_cost=1,
                                       max_prod=6,
                                       store_cost=np.array([0, 0.5, 0.5, 0.5]),
                                       truck_cost=np.array([3, 3, 3]),
                                       cap_store=np.array([90, 10, 10, 10]),
                                       penalty_cost=1,
                                       price=2.5,
                                       gamma=1,
                                       max_demand = 4,
                                       episode_length = max_steps))
env_names.append("medium_hard_environment_2")

environments.append(SupplyDistribution(n_stores=3,
                                       cap_truck=2,
                                       prod_cost=1,
                                       max_prod=3,
                                       store_cost=np.array([0, 2, 0, 0]),
                                       truck_cost=np.array([3, 3, 0]),
                                       cap_store=np.array([50, 10, 10, 10]),
                                       penalty_cost=1,
                                       price=2.5,
                                       gamma=1,
                                       max_demand = 3,
                                       episode_length = max_steps))
env_names.append("special_environment")


environments.append(SupplyDistribution(n_stores=3,
                                       cap_truck=2,
                                       prod_cost=1,
                                       max_prod=3,
                                       store_cost=np.array([0, 2, 0, 0]),
                                       truck_cost=np.array([3, 3, 0]),
                                       cap_store=np.array([50, 10, 10, 10]),
                                       penalty_cost=1,
                                       price=2.5,
                                       gamma=1,
                                       max_demand = 4,
                                       episode_length = max_steps))
env_names.append("special_environment_2")

environments.append(SupplyDistribution(n_stores=3,
                         cap_truck=2,
                         prod_cost=1,
                         max_prod= 5,
                         store_cost=np.array([0.1, 0.5, 0, 1]),
                         truck_cost=np.array([2, 4, 6]),
                         cap_store=np.array([30, 5, 10, 20]),
                         penalty_cost=1,
                         price=5,
                         gamma=1,
                         max_demand = 3,
                         episode_length = max_steps))
env_names.append("difficult_environment")





for test_num in range(len(test_to_run)):
    if test_to_run[test_num]:
        result_file_names = []
        agents = []
        env = environments[test_num]
        test_name = env_names[test_num]
        if add_s_q:
            result_file_names.append(test_name + "_s_q")
            agents.append(s_q_agent(threshold=np.array(env.cap_store/2), reorder_quantity=np.array([env.max_prod, env.cap_truck, env.cap_truck, env.cap_truck])))

        if add_sarsa:
            result_file_names.append(test_name + "_sarsa_V1")
            agents.append(approximate_sarsa_agent(env))

        if add_reinforce_1:
            result_file_names.append(test_name + "_reinforce")
            agents.append(REINFORCE_agent(env, actions_per_store=3, max_steps=max_steps, type_of_phi=1))

        if add_reinforce_2:
            result_file_names.append(test_name + "_reinforce_phi2")
            agents.append(REINFORCE_agent(env, actions_per_store=3, max_steps=max_steps, type_of_phi=2))

        if add_reinforce_3:
            result_file_names.append(test_name + "_reinforce_phi3")
            agents.append(REINFORCE_agent(env, actions_per_store=3, max_steps=max_steps, type_of_phi=0))


        for i in range(len(agents)):
            evaluate_agent(agents[i], env, n_episodes, max_steps, output, status_freq, print_freq, log_freq,
                           results_folder_path, result_file_names[i])
