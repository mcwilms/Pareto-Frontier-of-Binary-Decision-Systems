
import numpy as np
import pandas as pd
import math

class data_distribution:
    
    def __init__(self, alpha0, beta0, alpha1, beta1, num_thresholds): 
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.num_thresholds = num_thresholds
        
    def B_function(self, alpha, beta):
        return (math.gamma(alpha)*math.gamma(beta)) / math.gamma(alpha+beta)

    def beta_function(self, x, alpha, beta):
        return (1/self.B_function(alpha,beta)) * (x**(alpha-1)) * ((1-x)**(beta-1))
    
    def create_dataframe(self):
        x = np.linspace(0, 1, self.num_thresholds + 1)
        x_centers = (x[:-1] + x[1:]) / 2     # center of each bin
        bin_width = x[1] - x[0]
        synth_data1 = pd.DataFrame({
            'left_bound' : x[:-1],                       # left bound of probability interval
            'right_bound' : x[1:],                       # right bound of probability interval
            'p_center' : x_centers                       # probability of Y=1 for interval (NB centered)
            })
        synth_data1['pop0'] = self.beta_function(synth_data1.p_center, self.alpha0, self.beta0) * bin_width # fraction of individuals belonging to group 0 in bin
        synth_data1['pop1'] = self.beta_function(synth_data1.p_center, self.alpha1, self.beta1) * bin_width # fraction of individuals belonging to group 1 in bin
        
        return synth_data1