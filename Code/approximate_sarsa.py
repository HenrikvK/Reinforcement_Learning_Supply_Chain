#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from supply_distribution import SupplyDistribution
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.stats import linregress

class approximate_sarsa_agent(object):
    
    def __init__(self, env):
        # Set environment
        self.env = env
        
        # Initialize theta random
        self.theta_len = 15*env.n_stores+15
        self.theta = np.zeros(self.theta_len)#np.ones(env.n_stores+2)#
        # Initialize the stepsize alpha
        self.alpha = 1e-10
        # Initialize Epsilon for epsilon greedy
        self.epsilon = 0.1
        # Initialize agent parameters for stepsize rule        
        self.n=1
        # Initialize environment params
        self.env_params = (env.prod_cost, env.store_cost, env.price, env.n_stores)
        # Initialize status logger
        self.log = []
    
    def phi(self, state, action):
    
        # Copy variables for easier to read code
        n_stores = self.env.n_stores
        price = self.env.price
        prod_cost = self.env.prod_cost
        store_cost = self.env.store_cost.reshape(n_stores+1,1)
        penalty_cost = self.env.penalty_cost
        cap_truck = self.env.cap_truck
        truck_cost = self.env.truck_cost.reshape(n_stores,1)
        action_dim = action.ndim
        cap_store = self.env.cap_store.reshape(n_stores+1, 1)
        
        # Initialize phi
        if action_dim==1:
            phi = np.zeros((self.theta_len, 1))
            action = action.reshape(1, n_stores+1) # reshape action so matrix operation is possible
        else:
            phi = np.zeros((self.theta_len, action.shape[0]))
            
        # Create simple estimates for demand and storage levels in the next episode
        d_next = (2*state[n_stores+1:2*n_stores+1] - state[2*n_stores+1:3*n_stores+1])
        s_next = ((state[0:n_stores+1] - np.hstack((0, d_next))).T + action).T
        s_next[0] -= np.sum(action[:, 1:], axis=1)
        s_next = np.minimum(s_next, cap_store)
        # Save size of s_next
        s_shape = (s_next.shape[0]-1, s_next.shape[1])

        # Create simple scenarios for errors within s_next estimation
        s_next_plus = s_next.copy()
        s_next_plus[1:n_stores+1,:] += 1
        s_next_minus = s_next.copy()
        s_next_minus[1:n_stores+1,:] -= 1

        # Add bias
        phi[0,:] = 1
            
        # 1. Reward from sales
        phi[1,:] = np.sum(d_next)*price
            
        # 2. Production cost
        phi[2,:] = action[:,0]*prod_cost
            
        # 3. Store cost
        phi[3:n_stores+4,:] = -np.maximum(np.zeros(s_next.shape), s_next) * store_cost
        
        # 4. Penalty cost
        phi[n_stores+4:2*n_stores+4,:] = np.minimum(np.zeros(s_shape), s_next[1:,:]) * penalty_cost
            
        # 5. Transportation cost
        phi[2*n_stores+4:3*n_stores+4,:] = -np.ceil(action[:,1:] / cap_truck).T * truck_cost
        
        # 6. Store cost for scenarios
        phi[3*n_stores+4:4*n_stores+5,:] = -np.maximum(np.zeros(s_next.shape), s_next_minus) * store_cost
        phi[4*n_stores+5:5*n_stores+6,:] = - np.maximum(np.zeros(s_next.shape), s_next_plus) * store_cost
        
        # 7. Penalty cost for scenarios
        phi[6*n_stores+6:7*n_stores+6,:] = np.minimum(np.zeros(s_shape), s_next_minus[1:,:]) * penalty_cost
        phi[7*n_stores+6:8*n_stores+6,:] = np.minimum(np.zeros(s_shape), s_next_plus[1:,:]) * penalty_cost
        
        # 8. Reward from scenarios
        phi[8*n_stores+6,:] = (np.sum(d_next)+len(d_next))*price
        phi[8*n_stores+7,:] = (np.sum(d_next)-len(d_next))*price
                
        # 9. Factory stock can satisfy next estimated demand
        phi[8*n_stores+8,:] = (s_next[0] >= np.sum(d_next))*1
        
        # 10. Positive stock in warehouses
        phi[8*n_stores+9:9*n_stores+9,:] = (s_next[1:n_stores+1,:] >= 0)*1
        
        # 11. Estimate of demand in next period and quadratic demand in next period
        phi[9*n_stores+9:10*n_stores+9,:] = d_next.reshape(n_stores, 1)
        phi[10*n_stores+9:11*n_stores+9,:] = np.power(d_next, 2).reshape(n_stores, 1)
        
        # 12. Squared difference from quantiles of store capacity
        phi[11*n_stores+9:12*n_stores+10,:] = np.power(s_next - 1*cap_store/4, 2)
        phi[12*n_stores+10:13*n_stores+11,:] = np.power(s_next - 2*cap_store/4, 2)
        phi[14*n_stores+11:15*n_stores+12,:] = np.power(s_next - 3*cap_store/4, 2)
        
        # 13. Production able to satisfy demand?
        phi[15*n_stores+12,:] = action[:,0] - np.sum(d_next)
        phi[15*n_stores+13,:] = action[:,0] - np.sum(d_next + len(d_next))
        phi[15*n_stores+14,:] = action[:,0] - np.sum(d_next - len(d_next))
        
        # Normalize features
        #phi[:,:] /= np.sum(phi, axis = 0)
        
        # Format output in case of single action input
        if action_dim ==1:
            return phi.reshape((self.theta_len,))
            
        return phi
    
    def get_action(self, state):
        # Find all possible actions
        actions = np.array(self.env.action_space())
        
        # Select allowed actions
        #actions = self.allowed_actions(actions)
        
        # With probability epsilon, select random action
        if np.random.rand() < self.epsilon:
            return actions[np.random.randint(0, len(actions))]

        # With probability 1-epsilon, select greedy action
        return actions[np.argmax(np.dot(self.theta, self.phi(state, actions)))]        
    
    def update(self, state, action, reward, state_new, action_new):
        # Calculate delta
        delta = reward + self.env.gamma * np.dot(self.theta, self.phi(state_new, action_new)) - np.dot(self.theta, self.phi(state, action))
        
        # Clip delta in case of extreme values
        delta = np.clip(delta, -1e100, 1e100)
        
        # Update theta
        self.theta += self.alpha * delta * self.phi(state, action)        
                
        # Update alpha, epsilon and n
        self.epsilon = update_epsilon(self.epsilon, self.n)
        self.alpha = update_alpha(self.alpha, self.n)
        self.n+=1
        
        # Save information for log
        self.log.append([self.n, self.alpha, self.epsilon, delta, self.theta.copy()])

    def create_plots(self, rewards):
        '''
            Plots parameters of agent
        '''
        # Receive information from agent
        ns = [self.log[i][0] for i in range(len(self.log))]
        alphas = [self.log[i][1] for i in range(len(self.log))]
        epsilons = [self.log[i][2] for i in range(len(self.log))]
        deltas = [self.log[i][3] for i in range(len(self.log))]
        thetas = [self.log[i][4] for i in range(len(self.log))]
        
        # Plot parameters
        fig = plt.figure(figsize=(5, 10), dpi=120)
        fig.add_subplot(6, 1, 1)
        plt.plot(rewards)
        fig.add_subplot(6, 1, 2)
        plt.plot(thetas)
        fig.add_subplot(6, 1, 3)
        plt.plot(deltas)
        fig.add_subplot(6, 1, 4)
        plt.plot(alphas)
        fig.add_subplot(6, 1, 5)
        plt.plot(epsilons)
        
        # Plot regression over rewards
        #reg = linregress(np.arange(len(rewards)), rewards)
        #plt.plot(np.arange(len(rewards)), rewards, 'o', label='original data')
        #plt.plot(np.arange(len(rewards)), reg.intercept + reg.slope*np.arange(len(rewards)), 'r', label='fitted line')
        
    def allowed_actions(self, actions):
        result = []
        for i in np.arange(actions.shape[0]):
            if actions[i,0] not in [0, 3, 5]:
                continue
            if actions[i,1] not in [0, 5, 10]:
                continue
            result.append(actions[i,:])
        return np.array(result)
    
@nb.njit(cache=True)
def update_epsilon(epsilon, n):
    return stc_stepsize(n, alpha_0=0.2, a=175, b=75, beta=0.5)
        
@nb.njit(cache=True)
def update_alpha(alpha, n):
    return stc_stepsize(n, alpha_0=1e-5, a=50, b=10, beta=0.95)

@nb.njit(cache=True)
def geometric_stepsize(alpha, beta=0.99):
    return alpha*beta

@nb.njit(cache=True)
def generalized_harmonic_stepsize(n, a=20):
    return a/(a+n-1)

@nb.njit(cache=True)
def mcclains_formular(alpha, target=0.005):
    return alpha/(1+alpha-target)

@nb.njit(cache=True)
def stc_stepsize(n, alpha_0=0.5, a=150, b=50, beta=0.65):
    return alpha_0 * (b/n+a) / (b/n+a+np.power(n, beta))