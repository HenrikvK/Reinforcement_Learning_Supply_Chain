from numpy import *
import time

    # not used:
def softmaxvec(Theta, obs):  # computes a vector of the softmax
    ''' The softmax activation function '''

    dotprod = zeros(Theta.shape[1])
    for k in range(Theta.shape[1]):
        dotprod[k] = dot(obs,Theta[:,k])
        
    correction = max(dotprod) - 6
    correction_vec = correction * ones(Theta.shape[1])
    prob = exp(dotprod - correction_vec)
    prob = prob/ sum(prob)
    return prob

def softmax_f(Theta, obs, allowed_actions):
    count_allowed_a = int(sum(allowed_actions))        
    prob = zeros(count_allowed_a)        
    actions_ind = zeros(count_allowed_a).astype(int)   # keep an array of all allowed actions
    dotproduct  = zeros(count_allowed_a)  # save an array with the computed dotproduct of theta and obs
    counter= 0
    count = 0
    for k in range(Theta.shape[1]):
        if allowed_actions[k]==1: 
            actions_ind[count] = k
            dotproduct[count] = dot(obs,Theta[:,k])
            count += 1

    correction = max(dotproduct) - 6 # choose a correction factor       
    correction_vec = correction * ones(count_allowed_a)
    
    exp_dot = exp(dotproduct - correction_vec)
    exp_dot = exp_dot / sum(exp_dot)
    
    prob = zeros(Theta.shape[1])
    prob[actions_ind] = exp_dot 
    return prob

            
    # not used:
def softmax(Theta, obs, index):  # computes an entry of the softmax corresponding to the action(index)
    ''' The softmax activation function '''
    sum_exp = 0
    dotprod = zeros(Theta.shape[1])
    for k in range(Theta.shape[1]):
        dotprod[k] = dot(obs,Theta[:,k])
        
    correction = max(dotprod) - 6
    
    for k in range(Theta.shape[1]):
        sum_exp += exp(dotprod[k] - correction )
    if sum_exp == float('+inf'):
        print("sum_exp is infinity !!!! ", sum_exp)
    return exp(dotprod[index] - correction )/ sum_exp

def phi_size(n_stores, type_of_phi, n_rbf = 3):
    # give back the size of phi, given which phi design you want:
    if type_of_phi == 1: # use squared stocks
        return n_stores+1+1
    elif type_of_phi == 2: # use rbf
        return n_rbf*(n_stores+1)+1
    else:
        return 1
    

        
def RBF(s,c):
    sigma = 1
    phi = np.zeros(len(c))
    for i in range(len(c)):
        phi[i] = np.exp( - (np.linalg.norm( (s-c[i])**2, ord=2)/(2*sigma)))
    return phi

def RBF_vec(state , centers):
    sigma = 50    
    rbf_vec = zeros((centers.shape)) # shape of centers: nbr_rbf, n_stores+1
    for i in range(centers.shape[0]): # for every rbf 
        rbf_vec[i,:] = exp( - (state - centers[i,:])**2/(2*sigma))
    return hstack(rbf_vec[:,i] for i in range(rbf_vec.shape[1]))


class REINFORCE_agent(object):
    '''
        A Policy-Search Method.
    '''
    
    def compute_phi(self,obs,type_of_phi,n_stores, n_rbf = 3):
        obs[-n_stores:] = obs[-2*n_stores:-n_stores] - obs[-n_stores:]
        if type_of_phi == 1: # use squared stocks
            squarred = obs[:n_stores+1]**2
            return hstack(([1],obs,squarred))
        elif type_of_phi == 2: # use rbf
            statemin = zeros(self.env.n_stores+1)
            statemax = self.env.cap_store 
            statemid = (statemax + statemin)/2
            centers = array([statemin,statemid,statemax])
            rbf = RBF_vec(obs[:n_stores+1] , centers)
            #print("State = ", hstack(([1],obs,rbf)))
            return hstack(([1],obs,rbf))
        else:
            return hstack(([1],obs))
    
    
    def allowed_actions(self):  # returns an array indicating which actions are allowed        
        a_allowed = zeros(self.dim_action)
        for i in range(self.dim_action):
            if self.possible_action(self.discrete2continuous[i]):
                a_allowed[i] = 1
                
        # warning message        
        if(sum(a_allowed)) <1:
            print("Warning: we have an action space of zero!!!!!!!!!")
           
        return a_allowed

    
    def choose_action(self, obs, allowed_actions, epsilon= 0):  # returns one of the allowed actions
        sum_exp = 0
        count_allowed_a = int(sum(allowed_actions))        
        prob = zeros(count_allowed_a)        
        actions_ind = zeros(count_allowed_a)   # keep an array of all allowed actions
        dotproduct   = zeros(count_allowed_a)  # save an array with the computed dotproduct of theta and obs
        counter= 0
        for k in range(self.Theta.shape[1]):
            if allowed_actions[k]==1:
                actions_ind[counter]=k             
                dotproduct[counter] = dot(obs,self.Theta[:,k])
                counter +=1
                # print("allowed action is:", self.discrete2continuous[k])
        # epsilon greedy        
        if random.rand() < epsilon:
            action = random.choice(len(prob), size = None)
        else:        
            correction = max(dotproduct) - 6 # choose a correction factor       

            counter= 0
            for k in actions_ind:
                prob[counter]= exp(dotproduct[counter] - correction)
                counter +=1        
            if sum(prob) > 10000000 or sum(prob) < 0.0000001:
                print("Warning : sum_exp = ", sum_exp )
            prob = prob/sum(prob)
            
            # choose action according to probability vector   
            action = random.choice(len(prob), size = None, p = prob)
               
        return actions_ind[action]
              
        
    def __init__(self, environment, actions_per_store, max_steps, output_freq = 1000, type_of_phi = 1):         
        # The step size for the gradient
        self.alpha = 0.0000001    # very sensitive parameter to choose

        self.type_of_phi = type_of_phi
        self.epsilon = 0  # epsilone -greedy (not used at the moment)
        self.t0 = time.time()  # time the algorithm!
        self.max_steps = max_steps  # steps when update is made
        self.env = environment
        
        self.dim_state = self.env.n_stores*3+1 + phi_size(self.env.n_stores, type_of_phi) 
        self.dim_action =  actions_per_store**(self.env.n_stores+1)  # number of actions to choose from
        
        # initialize weights (every action with same probability)
        self.Theta = zeros((self.dim_state , self.dim_action ))
        
        self.output = 0  # print some output every couple of updates
        self.output_freq = output_freq
        
        # To store an episode
        self.episode_allowed_actions = zeros((self.max_steps+1,self.dim_action)) # for storing the allowed episodes
        self.episode = zeros((self.max_steps+1,1+self.dim_state+1)) # for storing (a,s,r) 
        self.t = 0                                   # for counting time steps
        
        # define a matrix that lists the possible actions for each store
        available_actions = zeros( ( actions_per_store ,self.env.n_stores+1 ))   
        available_actions[:,0] = [0,int(self.env.max_prod/2),self.env.max_prod]
        for i in range(self.env.n_stores):
            available_actions[:,i+1] = [0,self.env.cap_truck,self.env.cap_truck*2]
        
        # Discretize the action space: compute all action combinations
        self.discrete2continuous = []
        if self.env.n_stores == 3:
            for i in range(available_actions.shape[0]):
                for j in range(available_actions.shape[0]):
                    for k in range(available_actions.shape[0]):
                        for l in range(available_actions.shape[0]):
                            self.discrete2continuous.append( array([int(available_actions[l,0]), int(available_actions[i,1]), int(available_actions[j,2]), int(available_actions[k,3])]))
                        # We use the l for the a0 so we have then ordered by store action and then by production. So it matches the action space order
        elif self.env.n_stores == 2:
            for i in range(available_actions.shape[0]):
                for k in range(available_actions.shape[0]):
                    for l in range(available_actions.shape[0]):
                        self.discrete2continuous.append( array([int(available_actions[l,0]), int(available_actions[i,1]), int(available_actions[k,3])]))
                        
        elif self.env.n_stores == 1:
            for i in range(available_actions.shape[0]):
                    for l in range(available_actions.shape[0]):
                        self.discrete2continuous.append( array([int(available_actions[l,0]), int(available_actions[i,1])]))
        
        return
        

    #   check if an action is allwoed
    def possible_action(self, action):
        if sum(action[1:]) > self.env.s[0]:
            return False
        if self.env.s[0] + action[0] - sum(action[1:]) > self.env.cap_store[0]:
            return False
        for i in range(1, len(action)):
            if self.env.cap_store[i] - self.env.s[i] < action[i]:
                return False
        return True

    
    def get_action(self,obs):
        """
            Act.
            Parameters
            ----------
            obs : numpy array
                the state observation
            Returns
            -------
            numpy array
                the action to take
        """
        
        # compute the new state!
        phi = self.compute_phi(obs,self.type_of_phi, self.env.n_stores)
        self.episode[self.t,1:self.dim_state+1] = phi   # set observations in log
      
        # save the allowed actions for that state        
        allowed_actions = self.allowed_actions()
        self.episode_allowed_actions[self.t,:] = allowed_actions
        
        # next time step
        self.t = self.t + 1
                
        # choose new action:
        action = int(self.choose_action( phi, allowed_actions, self.epsilon ))
        
        # Save some info to a episode
        self.episode[self.t-1,0] = action

        # Return the action to take
        return array(self.discrete2continuous[action])
    
    
    def update(self,state, action, reward, state_new, action_new):  # update the parameters of the agent
        
        # save reward in episode
        self.episode[self.t-2,-1] = reward
        
        # if we reach the end of an episode: update
        if self.t == self.max_steps+1:
            # compute the last reward:
            state_new, reward, done, info = self.env.step(action_new)
            self.episode[self.t-1,-1] = reward

            self.output += 1
            
            # change epsilon (not used right now)
            self.epsilon = self.epsilon * 0.9995
            
            grad = zeros(( self.dim_action, self.dim_state ))  # initialize empty gradient
            for ts in range(self.t-1):  
                Dt = sum(self.episode[ts:,-1])  # sum up all rewards
                action = int(self.episode[ts,0])
                
                x= self.episode[ts,1:-1]
                
                # compute the softmax vector for every time step
                softmax_vec = softmax_f(self.Theta,x,self.episode_allowed_actions[ts,:])
                for i in range(self.dim_action): 
                                        
                    if i == action:  # different gradient for the weight of the action that was performed                                                
                        grad[i,:] = grad[i,:] + (1 - softmax_vec[i]) * x * Dt
                    else:                        
                        grad[i,:] = grad[i,:] - softmax_vec[i] * x * Dt
            
            # print all desired output here
            if self.output % self.output_freq == 0:
                print("================Episode: ",self.output," ================")
                print("log :",self.episode )
                print("The sum of all gradient entries is: ", sum(sum(absolute(grad))))
                #print("Theta 2 after is : ", self.Theta[:,2])
                print("Algorithm time: ", time.time()- self.t0, " seconds!")
                print("Algorithm time per episode: ", (time.time()- self.t0)/ self.output, " seconds!")
                print("=========================================================")
            for i in range(self.dim_action):
                self.Theta[:,i] = self.Theta[:,i] + self.alpha *  grad[i,:]             
            
            # after episode, set everything to zero!
            self.t = 0
            self.episode_allowed_actions = zeros((self.max_steps+1,self.dim_action)) # for storing the allowed episodes
            self.episode = zeros((self.max_steps+1,1+self.dim_state+1)) # for storing (a,s,r) 
        
        return 
    def create_plots(self, rewards): # plot some useful information about the algorithm:
        print("========================Final Output=====================")
        #print("log :", self.episode)
        #print("The sum of all gradient entries is: ", sum(sum(absolute(grad))))
        #print("Theta 2 after is : ", self.Theta[:,2])
        print("Algorithm time: ", time.time()- self.t0, " seconds!")
        print("Algorithm time per episode: ", (time.time()- self.t0)/ self.output, " seconds!")
        print("=========================================================")       
        return



