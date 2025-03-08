from scipy.stats import norm, uniform

class ProposalTrajectoryDistribution:
    def __init__(self, mean, std, d):
        # Choose values of mean and std that will make it more likely for system to fail
        self.mean = mean
        self.std = std
        self.state_len = 6
        self.depth = d

    """    
    Intitial state is from the reset() function in Grasp.env(). The only probabilistic component of the state
    is the x and y object position. 
    x = 0.5 + 0.05 * random.uniform(-1, 1) 
    y = 0.0 + 0.1 * random.uniform(-1, 1)
    Want to return a distribution of possible x and y values based on these equatinos. 
    Transforming the uniform distribution we get:
    x ~ uniform(0.45, 0.1) and y ~ uniform(-0.1, 0.2) (in form of  uniform(loc, scale))
    We return a distribution for the x and y object position 
    """
    def initial_state_distribution(self):
        # I could add disturbance to the initial state to make failures more likely
        return [uniform(0.45, 0.1), uniform(-0.1, 0.2)]


    """ We add a disturbance only on the system's sensor. 
    The agent and environment are deterministic.
    The sensor is the observation of our environment and we add gaussian noise to 
    imitate imperfect sensor readings in the real world. 
    Return one distribution of sensor noise. 
    (In importance sampling we will sample from this distribution 6 times to get a disturbance 
    for each component of the observation: [x_g, y_g, z_g, x_o, y_o, z_o]) 
    """
    def disturbance_distribution(self, t):
        return norm(self.mean, self.std)

    """
    Return depth of the system
    """
    def depth(self):
        return self.depth
