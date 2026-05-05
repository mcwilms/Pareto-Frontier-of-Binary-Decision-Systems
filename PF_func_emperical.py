# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 11:27:18 2026

@author: wims
"""

#functions for construction of pareto frontier 

import numpy as np
import pandas as pd
from paretoset import paretoset


class pareto_front_binary:
    #general parameter settings
    num_group = 2
    u00, u01, u10, u11 = 1, 0, 0, 1
    v00, v01, v10, v11 = 0, 0, 1, 1
    
    def __init__(self, data, idx_y, idx_a, idx_p, num_thresholds ):
        data_0 = data[data[:,idx_a]==0]
        data_1 = data[data[:,idx_a]==1]
        self.y_vec_0 = data_0[:,idx_y]
        self.y_vec_1 = data_1[:,idx_y]
        self.p_vec_0 = data_0[:,idx_p]
        self.p_vec_1 = data_1[:,idx_p]
        self.thresholds = np.linspace(0, 1, num_thresholds + 1) 
        self.N0 = len(data_0)
        self.N1 = len(data_1)
        self.Ntot = self.N0 + self.N1 

    def utility_dm(self, Y, D):
        U = (1-Y) * ((1-D)*self.u00 + D*self.u10) + Y * ((1-D)*self.u01 + D*self.u11)
        return U
    
    def utility_ds(self, Y, D):
        V = (1-Y) * ((1-D)*self.v00 + D*self.v10) + Y * ((1-D)*self.v01 + D*self.v11)
        return V
    
    def fairness_score(self, V0, V1):
        FS = abs(V0-V1)
        return FS 
    
    def evaluate_threshold_lb(self, t):
        d_vec_0 = (self.p_vec_0 > t).astype(int)
        d_vec_1 = (self.p_vec_1 > t).astype(int)
        U_0 = self.utility_dm(self.y_vec_0, d_vec_0)
        U_1 = self.utility_dm(self.y_vec_1, d_vec_1)
        V_0 = self.utility_ds(self.y_vec_0, d_vec_0)
        V_1 = self.utility_ds(self.y_vec_1, d_vec_1)
        
        U_avg = (sum(U_0)+sum(U_1))/self.Ntot
        U_0_avg = sum(U_0)/self.N0
        U_1_avg = sum(U_1)/self.N1
        V_0_avg = sum(V_0)/self.N0
        V_1_avg = sum(V_1)/self.N1
        return U_0_avg, U_1_avg, V_0_avg, V_1_avg
    
    def evaluate_threshold_ub(self, t):
        d_vec_0 = (self.p_vec_0 <= t).astype(int)
        d_vec_1 = (self.p_vec_1 <= t).astype(int)
        U_0 = self.utility_dm(self.y_vec_0, d_vec_0)
        U_1 = self.utility_dm(self.y_vec_1, d_vec_1)
        V_0 = self.utility_ds(self.y_vec_0, d_vec_0)
        V_1 = self.utility_ds(self.y_vec_1, d_vec_1)
        
        U_avg = (sum(U_0)+sum(U_1))/self.Ntot
        U_0_avg = sum(U_0)/self.N0
        U_1_avg = sum(U_1)/self.N1
        V_0_avg = sum(V_0)/self.N0
        V_1_avg = sum(V_1)/self.N1
        return U_0_avg, U_1_avg, V_0_avg, V_1_avg
    
    def evaluate_thresholds(self):
        # columns = ['t', 'U_0_lb', 'U_0_ub', 'U_1_lb', 'U_1_ub', 'V_0_lb', 'V_0_ub', 'V_1_lb', 'V_1_ub']
        dict_thresholds = {}
        for t in self.thresholds:
            U_0_avg_lb, U_1_avg_lb, V_0_avg_lb, V_1_avg_lb = self.evaluate_threshold_lb(t)
            U_0_avg_ub, U_1_avg_ub, V_0_avg_ub, V_1_avg_ub = self.evaluate_threshold_ub(t)
            row = [t, U_0_avg_lb, U_0_avg_ub, U_1_avg_lb, U_1_avg_ub, V_0_avg_lb, V_0_avg_ub, V_1_avg_lb, V_1_avg_ub]
            dict_thresholds[t] = row
        # df_thresholds = pd.DataFrame(rows, columns=columns)
        return dict_thresholds
    
    def evaluate_threshold_combinations(self):
        dict_trs = self.evaluate_thresholds()
        results_lb_lb = []
        results_ub_ub = []
        results_lb_ub = []
        results_ub_lb = []
        for t0 in self.thresholds:
            for t1 in self.thresholds:
                results_lb_lb.append({
                    't0': t0,
                    't1': t1,
                    'U_avg': (dict_trs[t0][1]*self.N0+dict_trs[t1][3]*self.N1)/self.Ntot,
                    'FS': self.fairness_score(dict_trs[t0][5], dict_trs[t1][7]),
                    'rule': 'lb_lb'
                    })
                results_ub_ub.append({
                    't0': t0,
                    't1': t1,
                    'U_avg': (dict_trs[t0][2]*self.N0+dict_trs[t1][4]*self.N1)/self.Ntot,
                    'FS': self.fairness_score(dict_trs[t0][6], dict_trs[t1][8]),
                    'rule': 'ub_ub'
                    })
                results_lb_ub.append({
                    't0': t0,
                    't1': t1,
                    'U_avg': (dict_trs[t0][1]*self.N0+dict_trs[t1][4]*self.N1)/self.Ntot,
                    'FS': self.fairness_score(dict_trs[t0][5], dict_trs[t1][8]),
                    'rule': 'lb_ub'
                    })
                results_lb_lb.append({
                    't0': t0,
                    't1': t1,
                    'U_avg': (dict_trs[t0][2]*self.N0+dict_trs[t1][3]*self.N1)/self.Ntot,
                    'FS': self.fairness_score(dict_trs[t0][6], dict_trs[t1][7]),
                    'rule': 'ub_lb'
                    })
        return results_lb_lb, results_lb_ub, results_ub_lb, results_ub_ub
    
    def get_pareto_frontiers(self):
        results_lb_lb, results_lb_ub, results_ub_lb, results_ub_ub = self.evaluate_threshold_combinations()
        results_all = results_lb_lb + results_ub_ub + results_lb_ub + results_ub_lb
        df_results = pd.DataFrame(results_all)
        mask_lb_lb = paretoset(df_results[df_results.rule == 'lb_lb'][['U_avg','FS']], sense=['max','min'])
        df_paretoset_lb_lb = df_results[df_results.rule == 'lb_lb'][mask_lb_lb].sort_values(by='U_avg')
        mask_lb_ub = paretoset(df_results[df_results.rule == 'lb_ub'][['U_avg','FS']], sense=['max','min'])
        df_paretoset_lb_ub = df_results[df_results.rule == 'lb_ub'][mask_lb_ub].sort_values(by='U_avg')
        mask_ub_lb = paretoset(df_results[df_results.rule == 'ub_lb'][['U_avg','FS']], sense=['max','min'])
        df_paretoset_ub_lb = df_results[df_results.rule == 'ub_lb'][mask_ub_lb].sort_values(by='U_avg')
        mask_ub_ub = paretoset(df_results[df_results.rule == 'ub_ub'][['U_avg','FS']], sense=['max','min'])
        df_paretoset_ub_ub = df_results[df_results.rule == 'ub_ub'][mask_ub_ub].sort_values(by='U_avg')
        return df_paretoset_lb_lb, df_paretoset_lb_ub, df_paretoset_ub_lb, df_paretoset_ub_ub
    
                