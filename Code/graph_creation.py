import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def add_graph_to_show(agent_code_name, agent_label_name):
    agents.append(agent_code_name)
    agent_labels.append(agent_label_name)

###
# Graph creation file
# To create a graph for an agent, set the agent boolean variable to true.
# To select the environment, put the correct boolean variable to true in "tests_to_graph"
# If you want to run your own environment, append the name to "tests" and a boolean to "test to graph"


# Variables
add_s_q = True
add_sarsa = True
add_reinforce_1 = True  # Linear
add_reinforce_2 = True  # Quadratic
add_reinforce_3 = True  # RBF
tests = ["simple_environment_2", "simple_environment_3", "simple_environment_4", "medium_environment", "medium_environment_3stores", "medium_environment_3stores2", "medium_hard_environment_2", "special_environment", "special_environment_2", "difficult_environment"]
tests_to_graph = [False, False, False, True, False, False, False, True, False, False]
results_folder_path = "../results/"
#reward_plot_step = 10
# Create auxiliary lists
agents = []
agent_labels = []
marks = ['*', '+', 'o', '^', 'v', '+', 'p', 'h']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
add_s_q and add_graph_to_show("s_q", "(ς,Q)-Policy")
add_sarsa and add_graph_to_show("sarsa_V1", "Sarsa")
add_reinforce_1 and add_graph_to_show("reinforce3", "REINFORCE - Linear")
add_reinforce_2 and add_graph_to_show("reinforce3_phi2", "REINFORCE - Quadratic")
add_reinforce_3 and add_graph_to_show("reinforce3_phi3", "REINFORCE - RBF")
for test_num in range(len(tests)):
    test = tests[test_num]
    if tests_to_graph[test_num]:
        fig1 = plt.figure(figsize=(10, 4), dpi=120)
        for agent_num in range(len(agents)):
            agent = agents[agent_num]
            rewards = pd.read_csv(results_folder_path + test + "_" + agent + "_rewards.csv", header=None).values.flatten()
            sum100 = 0
            rewards_window = []
            for i in range(len(rewards)):
                sum100 += rewards[i]
                if i >= 100:
                    sum100-= rewards[i-100]
                rewards_window.append(sum100/min(i+1,100))
            #plt.title("Rewards for " + test)
            plt.plot(range(0, 15000, 10), rewards_window[:15000:10], marks[agent_num]+"-", markevery=30,label=agent_labels[agent_num])
            plt.xlabel('episodes', fontsize=18)
            plt.ylabel('reward', fontsize=18)
            plt.tight_layout()
            plt.legend(loc=4, prop={'size': 15})
            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.tick_params(axis='both', which='minor', labelsize=15)
        plt.show()
        if False:
            fig2 = plt.figure(figsize=(10, 4), dpi=120)
            for agent_num in range(len(agents)):
                agent = agents[agent_num]
                best_actions = pd.read_csv(results_folder_path + test + "_" + agent + "_best_actions.csv", header=None).values
                plt.title("Action of best case scenario for " + agent_labels[agent_num])
                plt.plot(best_actions[:-1,0], label='production')
                plt.plot(best_actions[:-1,1], label='sending to warehouse 1')
                if best_actions.shape[1] > 2:
                    plt.plot(best_actions[:-1, 2], label='sending to warehouse 2')
                if best_actions.shape[1] > 3:
                    plt.plot(best_actions[:-1, 3], label='sending to warehouse 1')
                plt.xlabel('steps', fontsize=18)
                plt.ylabel('stock', fontsize=18)
                plt.tight_layout()
                plt.legend(loc=4, prop={'size': 15})
                plt.tick_params(axis='both', which='major', labelsize=15)
                plt.tick_params(axis='both', which='minor', labelsize=15)
                plt.show()
        if True:
            for agent_num in range(len(agents)):
                fig3 = plt.figure(figsize=(10, 4), dpi=120)
                agent = agents[agent_num]
                best_stocks = pd.read_csv(results_folder_path + test + "_" + agent + "_best_stocks.csv", header=None).values
                #plt.title("Stocks for the best policy for " + agent_labels[agent_num])
                plt.plot(best_stocks[:-1,1], 'o-', label='Stock warehouse 1')
                if best_stocks.shape[1] > 2:
                    plt.plot(best_stocks[:-1, 2], '+-', label='Stock warehouse 2')
                if best_stocks.shape[1] > 3:
                    plt.plot(best_stocks[:-1, 3], 'x-', label='Stock warehouse 3')
                plt.plot(best_stocks[:-1,0], '*b-', label='Stock factory')
                plt.plot(np.zeros(len(best_stocks)), 'k')
                plt.xlabel('steps', fontsize=18)
                plt.ylabel('stock', fontsize=18)
                plt.tight_layout()
                plt.legend(loc=1, prop={'size': 15})
                plt.tick_params(axis='both', which='major', labelsize=15)
                plt.tick_params(axis='both', which='minor', labelsize=15)
            plt.show()