from scipy.stats import norm, uniform

class NominalTrajectoryDistribution:
    def __init__(self, d):
        # Should define nominal values for mean and std for the sensor noise (try standard normal)
        self.mean = 0
        self.std = 0.0001 # since direct estimation is deterministic
        self.state_len = 6
        self.depth = d


    """
    Intitial state is from the reset() function in Grasp.env(). The only probabilistic component of the state
    is the x and y object position. 
    x = 0.5 + 0.05 * random.uniform(-1, 1)
    y = 0.0 + 0.05 * random.uniform(-1, 1) 
    Want to return a distribution of possible x and y values based on these equatinos. 
    Transforming the uniform distribution we get:
    x ~ uniform(0.45, 0.1) and y ~ uniform(-0.05, 0.1) (in form of  uniform(mean, scale))
    We return a distribution for the x and y object position 
    """
    def initial_state_distribution(self):
        return [uniform(0.45, 0.1), uniform(-0.05, 0.1)]

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
