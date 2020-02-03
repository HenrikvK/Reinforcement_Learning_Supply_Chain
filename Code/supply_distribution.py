import numpy as np
import itertools

class SupplyDistribution:
    """
    The supply distribution environment
    """

    def __init__(self, n_stores=3, cap_truck=2, prod_cost=1, max_prod=8,
                 store_cost=np.array([0.01, 0.1, 0.1, 0.1]), truck_cost=np.array([1, 2, 3]),
                 cap_store=np.array([20, 5, 5, 5]), penalty_cost=2, price=30, gamma=0.90,
                 max_demand = 8, episode_length = 48):
        """
        :param n_stores:
        :param cap_truck:
        :param prod_cost:
        :param store_cost:
        :param truck_cost:
        :param cap_store:
        :param penalty_cost:
        :param price:
        """
        self.n_stores = n_stores
        self.s = np.zeros(self.n_stores + 1, dtype=int)
        self.demand = np.zeros(self.n_stores, dtype=int)
        self.demand_old = np.zeros(self.n_stores, dtype=int)
        self.price = price
        self.max_prod = max_prod
        # capacity
        self.cap_store = np.ones(n_stores + 1, dtype=int)
        self.cap_store = cap_store
        self.cap_truck = cap_truck
        # costs
        self.prod_cost = prod_cost
        self.store_cost = np.array(store_cost)
        self.truck_cost = np.array(truck_cost)
        self.penalty_cost = penalty_cost
        # demand
        self.max_demand = max_demand
        self.episode_length = episode_length
        # other variables
        self.gamma = gamma
        self.t = 0

        self.reset()

    def reset(self):
        """
        Resets the environment to the starting conditions
        """
        self.s = (self.cap_store/2).astype(np.int) #np.zeros(self.n_stores + 1, dtype=int)  # +1 Because the central warehouse is not counted as a store
        #self.s[0] = self.cap_store[0]/2
        self.t = 0
        # Initialize demand and update it directly to avoid jumps in demand of first step
        self.demand = np.zeros(self.n_stores, dtype=int)
        self.update_demand()
        self.demand_old = self.demand.copy() #np.zeros(self.n_stores, dtype=int)
        return np.hstack((self.s.copy(), self.demand.copy(), self.demand_old.copy()))

    def step(self, action):
        # Update state
        self.s[0] = min(self.s[0] + action[0] - sum(action[1:]), self.cap_store[0])
        self.s[1:] = np.minimum(self.s[1:] - self.demand + action[1:], self.cap_store[1:])
        
        # Update reward
        reward = (sum(self.demand) * self.price
                  - action[0] * self.prod_cost
                  - np.sum(np.maximum(np.zeros(self.n_stores+1), self.s[:self.n_stores+1]) * self.store_cost)
                  + np.sum(np.minimum(np.zeros(self.n_stores+1), self.s[:self.n_stores+1])) * self.penalty_cost # Changed to + so that penalty cost actually decrease reward -- Luke 26/02
                  - np.sum(np.ceil(action[1:] / self.cap_truck) * self.truck_cost))
        info = "Demand was: ", self.demand

        # Define state
        state = np.hstack((self.s.copy(), self.demand.copy(), self.demand_old.copy()))

        # Update demand old
        self.demand_old = self.demand.copy()

        # Update t
        self.t += 1

        # Update demand
        self.update_demand()
        
        # Set if done 0 since unused
        done = 0
        return state, reward, done, info

    def update_demand(self):
        """
        Updates the demand using the update demand function
        :return:
        """
        demand = np.zeros(self.n_stores, dtype=int)
        for i in range(self.n_stores):
            # We need an integer so we use the ceiling because if there is demand then we asume the users will buy
            # what they need and keep the rests. We use around to get an integer out of it.
            
            #try not random:
            demand[i] = int(np.floor(.5*self.max_demand * np.sin( np.pi * (self.t + 2*i) / (.5*self.episode_length) - np.pi ) + .5*self.max_demand + np.random.randint(0, 2))) # 2 month cycles
            # demand[i] = int(np.ceil(1.5 * np.sin(2 * np.pi * (self.t + i) / 26) + 1.5 + np.random.randint(0, 2))) 
        self.demand = demand

    def action_space_itertools(self):
        """
        Returns the set of possibles actions that the agent can make
        :return: The posible actions in a list of tuples. Each tuple with (a0, a1, ..., ak) k = n_stores.
        """
        feasible_actions = []
        a_0 = np.arange(0, self.max_prod + 1)

        iterator = [a_0, *[np.arange(0, min(self.s[0], self.cap_store[i] - self.s[i]) + 1) for i in np.arange(1, self.n_stores+1)]]
        for element in itertools.product(*iterator):
            if np.sum(element[1:]) <= self.s[0]:
                feasible_actions.append(element)
        return np.array(feasible_actions)

    def action_space_recur(self):
        feasible_actions_aux = self.action_space_recur_aux(0, [[]], self.s[0])
        feasible_actions = []
        for action in feasible_actions_aux:
            prod_being_send = sum(action)
            s_0 = self.s[0] - prod_being_send
            for production in np.arange(0, min(self.max_prod, self.cap_store[0] - s_0) + 1):
                feasible_actions.append([production] + action)
        return np.array(feasible_actions)

    def action_space_recur_aux(self, store_num, current_actions, prod_left):
        feasible_actions = []
        if store_num == self.n_stores:
            return current_actions
        for prod_being_send in range(0, min(prod_left, self.cap_store[store_num+1] - self.s[store_num+1]) + 1):
            new_actions = []
            for action in current_actions:
                new_action = action + [prod_being_send]
                new_actions.append(new_action)
            feasible_actions.extend(self.action_space_recur_aux(store_num+1, new_actions, prod_left - prod_being_send))
        return feasible_actions

    def action_space(self):
        return self.action_space_recur()

    def observation_space(self):
        """
        Used to observe the current state of the environment
        :return:
        """
        return