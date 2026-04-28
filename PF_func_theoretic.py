
import numpy as np


class pareto_front_theoretic():
    def __init__(self,u00,u01,u10,u11,v00,v01,v10,v11,num_thresholds, N_0, N_1, N_tot):
        self.u00 = u00
        self.u01 = u01
        self.u10 = u10
        self.u11 = u11
        self.v00 = v00
        self.v01 = v01
        self.v10 = v10
        self.v11 = v11
        self.num_thresholds = num_thresholds
        self.N_0 = N_0
        self.N_1 = N_1
        self.N_tot = N_tot

    ##FUNCTIONS
    def utility_dm(self, p, dv):
        Udm = self.u00*(1-dv)*(1-p) + self.u01*(1-dv)*p + self.u10*dv*(1-p) + self.u11*dv*p
        return Udm
    
    def utility_ds(self, p, dv):
        Uds = self.v00*(1-dv)*(1-p) + self.v01*(1-dv)*p + self.v10*dv*(1-p) + self.v11*dv*p
        return Uds


    def evaluate_threshold_lb(self, dataset, threshold, column):
        dataset['dv'] = np.where(dataset['p_center'] >= threshold, 1, 0)
        #determine the utility per bin
        dataset['U'] = self.utility_dm(p=dataset['p_center'], dv=dataset['dv'])
        dataset['V'] = self.utility_ds(p=dataset['p_center'], dv=dataset['dv'])
        
        #calculate the expected utility
        U_avg = sum(dataset['U'] * dataset[column])
        V_avg = sum(dataset['V'] * dataset[column])
        return U_avg, V_avg

    def evaluate_threshold_ub(self, dataset, threshold, column):
        dataset['dv'] = np.where(dataset['p_center'] < threshold, 1, 0)
        #determine the utility per bin
        dataset['U'] = self.utility_dm(p=dataset['p_center'], dv=dataset['dv'])
        dataset['V'] = self.utility_ds(p=dataset['p_center'], dv=dataset['dv'])
        
        #calculate the expected utility
        U_avg = sum(dataset['U'] * dataset[column])
        V_avg = sum(dataset['V'] * dataset[column])
        return U_avg, V_avg
    
    def fairness_score_utility_space(self, V0, V1, metric):
        if metric == 'abs_delta_V':
            FS = abs(V0-V1)
        elif metric == 'delta_mean_V':
            FS = (V0 -(V0 * self.N_0/self.N_tot + V1 * self.N_1/self.N_tot) )**2 + (V1 -(V0 * self.N_0/self.N_tot + V1 * self.N_1/self.N_tot) )**2
        return FS
