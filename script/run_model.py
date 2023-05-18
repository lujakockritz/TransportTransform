# %%
import os
import sys
import time
import datetime
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import importlib

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
abm_dir = os.path.join( ROOT_DIR, 'ABM' )
sys.path.append(abm_dir)


from TT_Model import TransportModel
from visualization import viz_methods

output_dir = os.path.join('..', 'results', 'output', 'temp')

# %%
####  RUNNING MODEL FUNCTION ####
def modelRuns(runs, visualize,
        include_all,
        number_agents, steps, preference_1_prob, preference_2_prob,
        habit_on, cluster_factor_run,
        crowding_on, occupancy_function, occupancy_parameter,
        norms_on, norm_threshold, norm_prob):
    # get system start time
    t0total = time.time()
    systemTime = datetime.datetime.now()
    # add system time to file names
    current_time = systemTime.strftime("%Y-%m-%d_%H:%M:%S")
    if include_all:
        print(f'starting at {current_time}')

        print("model function started")
    modes = ['d_car','d_bike','d_public','d_carsh']
    multi_columns = pd.MultiIndex.from_product([range(runs), modes])
    df_mode_counts = pd.DataFrame(index = range(steps), columns = multi_columns)
    
    mode_counts_list = [] # Initialize list to accumulate mode counts for each run
    degrees_list = []  # list to store degrees of all nodes for each run

    for j in range(runs):

        ### Run the Model ###
        t0total = time.time()
        
        TT_model = TransportModel(include_all, number_agents, steps, preference_1_prob, preference_2_prob,
                                  habit_on, cluster_factor_run,
                                  crowding_on, occupancy_function, occupancy_parameter,
                                  norms_on, norm_threshold, norm_prob)
        
        #print(f"model initialized for run {j}")

        for i in range(steps):    
            TT_model.step()

        df_mode_counts.loc[:, (j, modes)] = TT_model.mode_counts
        mode_counts_list.append(TT_model.mode_counts)
        degrees = [TT_model.G.degree(n) for n in TT_model.G.nodes()]  # get the degrees for all nodes
        degrees_list.append(degrees)  # store the degrees for this run

        if visualize:
            viz_methods.visualize_graph_pref_1(TT_model)
    
    # Create the mode_counts_list DataFrame using pd.concat
    
    df_mode_counts = pd.concat(mode_counts_list, axis=1, keys=range(runs))
    
    print("model complete")

    systemTime = datetime.datetime.now()
    # add system time to file names
    current_time = systemTime.strftime("%Y-%m-%d_%H:%M:%S")
    print(f'finishing at {current_time}')

    t1total = time.time()
    total=t1total-t0total
    print("total time:", total)

    return df_mode_counts

# %%
########################## RESULTS ##########################
## Results 
if 'TT_Model' in sys.modules:
    del sys.modules['TT_Model']
from TT_Model import TransportModel

# Setting parameters
runs = 5
steps = 100
number_agents = 1000


cluster_factor = 0.2
preference_1_prob = 0.7
preference_2_prob = 0.25
habit_on = True
include_all = False
crowding_on = False
occupancy_function = "threshold"
occupancy_parameter = 0.4#(0.09,0.415)

visualize = False
mode_counts_list_all = []
mode_counts_list = modelRuns(runs, visualize,
                    include_all,
                        number_agents, 
                        steps, preference_1_prob , preference_2_prob,
                        habit_on, cluster_factor,
                        crowding_on, occupancy_function, occupancy_parameter,
                    norms_on = True, norm_threshold = 0.5, norm_prob = 0.8)
    
mode_counts_list_all.append(mode_counts_list)
# %%
if 'visualization' in sys.modules:
    del sys.modules['visualization_v2']
from visualization import viz_methods

title = f"Social Norms results - {runs} runs with {number_agents} agents"

shading = True
text = ['norm_threshold = 0.6','norm_prob = 0.8']
viz_methods.viz_modeCounts(mode_counts_list_all, runs, title, shading, text)