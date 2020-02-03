#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class s_q_agent(object):
    
    def __init__(self, threshold, reorder_quantity):
        # Initialize parameters
        self.threshold = threshold
        self.reorder_quantity = reorder_quantity

        # Initialize status logger
        self.log = []

    def get_action(self, state):
        '''
            Heuristic based on s_q policy
        '''
        # Initialize output and helper variable
        a = np.zeros(int((len(state)-1)/3)+1)
        disposable_produce = state[0]
        
        # 1. Set actions for individual warehouses
        for i in np.arange(1, int((len(state)-1)/3)+1):
            # Check if current stock is below replenishment threshold
            if state[i] < self.threshold[i]:
                # Replenish with reorder quantity if possible
                a[i] = min(self.reorder_quantity[i], disposable_produce)

                # Update remaining disposable produce
                disposable_produce -= a[i]
    
        # 2. Set action for factory
        if disposable_produce < self.threshold[0]:
            a[0] = self.reorder_quantity[0]

        return a.astype(np.int)


    def update(self, state, action, reward, state_new, action_new):
        '''
            update function not required for q-s-policy
        '''
        pass
    
    def create_plots(self, rewards):
        '''
            Plots parameters of agent
        '''
        pass