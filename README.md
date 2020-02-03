# Reinforcement Learning for Supply Chain Optimization
Reinforcement Learning implementation in Python to optimize a supply chain. COmparison of Sarsa and REINFORCE algorithms against a simple baseline.

Projects includes code for the paper: 
Kemmer, L., von Kleist, H., de Rochebouët, D., Tziortziotis, N., & Read, J. (2018). Reinforcement learning for supply chain optimization. In European Workshop on Reinforcement Learning (Vol. 14, No. 10).



### How to run: <br>
In test_notebook (or separately) call:
* Run testing.py to do multiple runs with the different environments already created or to create new ones. The results are stored in results/
* Run graph_creation.py to create the graphs used in the paper to compare different environments and agents


### Code functionality: <br>
#### Running scripts
1. testing.py
* define here which environment to use
* define here which agent to use
* stores computation results in results/ folder
2. graph_creation.py
* plot output for the stored results

#### Environment:
supply_distribution.py
* implements the environment with variables (can be set in testing.py):
    - number of stores
    - capacity of trucks
    - production cost
    - maximum production at factory
    - storage cost
    - truck cost
    - capacity to store
    - penalty cost 
    - price of product
    - gamma (discounting factor)
    - maximum demand
    - episode length
evaluate_agent.py
* evaluates the performance of an agent in the environment

#### Agents:
1. s_q_agent.py 
    * baseline agent (refills warehouse when storage drops below threshold)
2. approximate_sarsa.py
    * approximate sarsa algorithm 
    * choose phi carefully (which variables to include)
3. reinforce.py
    * implements REINFORCE agent
    * choose alpha carefully
    * choose phi (the Kernel) carefully (quadratic and RBF kernels might work better than linear kernels) <br>
        options (for `type_of_phi`): 0 = linear, 1 = quadratic, 2 = RBF
   

