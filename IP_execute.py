
import IP_PFSMG_OG as pfsmg
import IP_functions_OG as func
import pickle
import time 

SEED_arr = [112233455, 666, 889099, 558866, 559966, 99998866, 99888866, 9966668, 99677808, 7542229,  \
            11234455, 6600451, 8852099, 110036, 277066, 277066, 900186, 9820816, 908253, 1089532]
num_data = range(1000, 6001, 500)

data_idx = 8
prob = func.Fairness_LogRe_DI_binary('data/Adult_income_gender_reduced.txt', 'Adult_income_gender', 1, \
                                     SEED_arr[data_idx], num_data[data_idx], 'train')

run = pfsmg.Main_SMG(prob)
run.max_len_pareto_front = 1500
run.max_iter = 1000

## key parameters
run.point_per_iteration = 2 
run.num_steps_per_point = 3 
run.stepsize = 2.1 
run.step_scheme = 3 
run.alpha = 1.0/3 
run.discount_iter_interval = 80 
run.batch1_init = 80 
run.batch1_factor = 1.005 
run.batch1_max = 1 
run.batch2_init = 80  
run.batch2_factor = 1.005 
run.batch2_max = 1.0/2  

## other parameters
run.num_starting_pts = 5
run.percent_explore = 0.4 
run.f1_explore_interval = 15 
run.f2_explore_interval = 1000 

run.f1_explore_pt_per_iter = 2 
run.f2_explore_pt_per_iter = 1 
run.f1_num_steps_per_point = 3*run.num_steps_per_point 
run.f2_num_steps_per_point = 2*run.num_steps_per_point 

run.num_max_hole_points = 5 
run.max_hole_explore_pt_per_iter = 2*run.point_per_iteration 
run.max_hole_num_steps_per_point = run.num_steps_per_point 
run.max_hole_only = False 
run.dense_threshold = 0 # 1.0/(800 + self.num_iter/2.0)

start_time = time.time()
f1_arrays7, f2_arrays7, point_arrays7, total_time = run.main_SMG()
print("--- %s seconds ---" % (time.time() - start_time))


## Dump resulting Pareto front into pickle if you want
logfile = "gender_seed%s_num_data%s.pickle"\
          %(SEED_arr[data_idx], num_data[data_idx])
with open(logfile, 'wb') as handle:
    pickle.dump([prob.data_name, total_time, run.num_grad_eval_f1, \
             run.num_grad_eval_f2, run.num_iter, f1_arrays7, \
             f2_arrays7, point_arrays7], handle)
        