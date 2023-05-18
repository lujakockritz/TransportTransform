# %%

#import config
import sys
import os
import random
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid

# Add the parent directory to the system path
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
viz_dir = os.path.join(ROOT_DIR, 'script')
sys.path.append(viz_dir)

from visualization import viz_methods

TT_Model_Data = pd.read_csv('..\data\edited\TT_Model_Data_cleaned.csv') 

modes = ['d_public','d_carsh','d_car','d_bike']

modes_attitudes = ['Attitude_ICECar',
 'Attitude_PT',
 'Attitude_Bike',
 'Attitude_CarSh']


# %%
###############
#### AGENT ####
###############
              

class UserAgent(Agent):
    def __init__(self, unique_id, model, group, preference_1, preference_2, license, car_owner, attributes):

        super().__init__(unique_id, model)
        self.ID = unique_id
        self.model = model
        self.group = group
        self.car_owner = car_owner
        self.preference_1 = preference_1
        self.preference_2 = preference_2
        self.license = license
        self.pos = unique_id

        self.attitudes = {} # currently not used
        self.characteristics = {} # currently not used
        self.frequencies = {}   # currently the only data used

        self.experiences = []

        self.main_mode = []

        for attr_name, attr_value in attributes.items():
            # Store the attitudes of the agent
            if attr_name.startswith('Attitude_'):
                self.attitudes[attr_name] = attr_value
            elif attr_name.startswith('c_'):
                self.characteristics[attr_name] = attr_value
            elif attr_name.startswith('f_'):
                self.frequencies[attr_name] = attr_value
            else:
                print('unknown attribute found ' + str(attr_name))

        self.available_modes = ['d_public', 'd_bike']
        if self.license:
            self.available_modes.append('d_carsh')            
        if self.license and self.car_owner:
            self.available_modes.append('d_car')

        self.habit = random.randint(5, 10)
        self.habit_next = random.randint(1, self.habit)
        self.time_changed_mode = 0

    def step(self):                
        def preference_mode_choice(mymodes):                   
            probs = [0.05] * len(mymodes)
            if self.preference_1 in mymodes:
                idx = mymodes.index(self.preference_1)
                probs[idx] = self.model.preference_1_prob
                if self.preference_2 in mymodes:
                    probs[mymodes.index(self.preference_2)] = self.model.preference_2_prob
            elif self.preference_2 in mymodes:
                probs[mymodes.index(self.preference_2)] = self.model.preference_1_prob
            
            return random.choices(mymodes, weights=probs, k=1)[0]
        
        def social_pressure(self):
            
            num_neighbors = len(self.model.G.edges(self.ID))
            
            # Does the agent have neighbors?
            if num_neighbors > 0:
                percentages = {}
                
                neighbor_indices = list(self.model.G.neighbors(self.ID))
                for mode in modes:
                    neighbor_modes = [self.model.all_agents[i].main_mode for i in neighbor_indices]
                    count = neighbor_modes.count(mode)
                    percentage = count / num_neighbors 
                    percentages[mode] = percentage
                max_mode, max_percentage = max(percentages.items(), key=lambda x: x[1])
                
                # Do enough of the neighbors share the same mode?
                if max_percentage >= self.model.norm_threshold:
                    self.current_norm = max_mode
                    if self.current_norm != self.main_mode:
                        if random.random() < self.model.norm_prob:
                            self.adopted_norm = self.current_norm

        
        def calculate_average_probability(last_experiences):
            """
            Calculate the average probability from the experiences since the agent last updated their main_mode.

            Parameters:
            - last_experiences (list): List of experiences since the agent last updated their main_mode.

            Returns:
            - average_probability (float): Average probability.
            """
            total_probability = sum(experience[2] for experience in last_experiences)
            num_experiences = len(last_experiences)
            
            if num_experiences > 0:
                average_probability = total_probability / num_experiences
            else:
                average_probability = 0.0
            
            return average_probability
        
        def decision_tree():
            consider_modes = self.available_modes.copy()
            
            # Occupancy module
            if self.model.crowding_on:
                last_experiences = [experience for experience in self.experiences if experience[0] >= self.time_changed_mode]
                sum_last_experiences = sum(experience[2] for experience in last_experiences[1:])
                if self.model.occupancy_function == 'threshold':                              
                    if sum_last_experiences < 0:   
                        if self.main_mode in consider_modes: # to prevent bugs
                            consider_modes.remove(self.main_mode)  
                else:
                    _prob = calculate_average_probability(last_experiences)
                    if random.random() < _prob:
                        if self.main_mode in consider_modes: # to prevent bugs
                            consider_modes.remove(self.main_mode) 
                    
            # Base module
            self.main_mode = preference_mode_choice(consider_modes)

            # Social Norm module
            if self.model.norms_on:
                # Reset norms
                self.current_norm = None
                self.adopted_norm = None

                social_pressure(self)

                # Change mode to norm if agent adopted norm
                if self.adopted_norm is not None: 
                    self.main_mode = self.adopted_norm  

        # evaluate experience from last time step
        def crowding_last_step(num_same_mode, max_agents, occupancy_parameter):   
            
            threshold = max_agents * occupancy_parameter
            if num_same_mode > threshold:
                return - 1 # It was crowded
            else:
                return 1 # It was not crowded

        def linear_probability(num_same_mode, max_agents):
            # Calculate the probability based on a linear relationship
            probability = num_same_mode / max_agents
            return probability  
        
        def exponential_probability(num_same_mode, max_agents, exponent):
            # Calculate the probability based on an exponential relationship
            probability = (num_same_mode / max_agents) ** exponent
            return probability
        
        def log_probability(num_same_mode, max_agents, base):
            # Calculate the probability based on a logarithmic relationship
            probability = np.log(num_same_mode + 1) / np.log(max_agents + 1) / np.log(base)
            return probability
                
        def sigmoid_probability(x, max_x, base, turning_point_share):
            turning_point = max_x * turning_point_share
            exponent = -base * (x - turning_point)
            probability = 1 / (1 + np.exp(exponent))
            return probability
        

        # Agent decision tree

        # First mode choice from avilable modes
        if self.model.currentStep == 0:
            self.main_mode = self.main_mode = preference_mode_choice(self.available_modes)
        else:
            # From first timestep on

            # Check habit of agent
            if self.model.habit_on:
                if self.habit_next != 0:
                    self.habit_next -= 1
                else:
                    # Only consider changing the mode if habit time is up
                    decision_tree()                                  
                    
                    # reset habit times
                    self.habit_next = self.habit # reset habit time
                    self.time_changed_mode = self.model.currentStep # Record time the habit changed 
            else:
                #To evaluate effect of leaving habit out
                decision_tree()            
      
            if self.model.crowding_on: 
                num_same_mode = self.model.mode_counts.loc[self.model.currentStep - 1, self.main_mode]
                max_agents = self.model.num_agents
                if self.model.occupancy_function == 'threshold':
                    exp = crowding_last_step(num_same_mode, max_agents, self.model.occupancy_parameter)                
                elif self.model.occupancy_function == 'linear':
                    exp = linear_probability(num_same_mode, max_agents)
                elif self.model.occupancy_function == 'exponential':
                    exp = exponential_probability(num_same_mode, max_agents, self.model.occupancy_parameter)
                elif self.model.occupancy_function == 'sigmoid':
                    exp = sigmoid_probability(num_same_mode, max_agents, self.model.occupancy_parameter[0],self.model.occupancy_parameter[1])
                elif self.model.occupancy_function == 'logarithm':
                    exp = log_probability(num_same_mode, max_agents, self.model.occupancy_parameter)

                # Record experiencs from previous step
                my_experience = ((self.model.currentStep - 1), self.main_mode, exp)
                self.experiences.append(my_experience)


###############
#### MODEL ####
###############

class TransportModel(Model):
    def __init__(self, include_all, number_agents_included, step_number, preference_1_prob, preference_2_prob, 
                 habit_on, cluster_factor, 
                 crowding_on, occupancy_function, occupancy_parameter,  
                 norms_on, norm_threshold, norm_prob,
                 network_type = 'random'):
        
        self.num_agents = number_agents_included #  unless include_all is True
        if include_all:
            print(self.num_agents)
        self.step_number = step_number
        self.network_type = network_type  
        self.currentStep = 0    

        self.G = nx.Graph()  
        self.G.add_nodes_from(range(0, self.num_agents))
        self.schedule = RandomActivation(self)
        self.grid = NetworkGrid(self.G)

        # Module parameters
        self.preference_1_prob = preference_1_prob
        self.preference_2_prob = preference_2_prob
        self.habit_on = habit_on
        self.cluster_factor = cluster_factor
        self.crowding_on = crowding_on
        self.occupancy_function = occupancy_function
        print(self.occupancy_function)
        self.occupancy_parameter = occupancy_parameter
        self.norms_on = norms_on
        self.norm_threshold = norm_threshold 
        self.norm_prob = norm_prob

        # Create Bags for Network Data
        self.all_agents=[] # Creates empty bag to place agents in 
        
        self.mode_counts = pd.DataFrame(columns=modes)

        pref_index = pd.MultiIndex.from_product([modes, ['pref_1', 'pref_2']])
        self.pref_counts = pd.DataFrame(index= pref_index, columns = range(self.step_number) )

    
        exclude_list = ['ID','group','preference_1','preference_2','license','car_owned','ID.1','Original_ID'] 

        self.agent_id = 0  # Initialize the agent ID variable
        
        for i, row in TT_Model_Data.iterrows() if include_all else TT_Model_Data.head(number_agents_included).iterrows():
            agent_attributes = {k: v for k, v in row.items() if k not in exclude_list } 
            agent = UserAgent(row['ID'], self, row['group'], row['preference_1'],row['preference_2'],row['license'], row['car_owned'], agent_attributes)
            self.grid.place_agent(agent, self.agent_id)  # Place agent in the grid
            self.all_agents.append(agent)
            self.schedule.add(agent)  # Add agent to the schedule
            self.agent_id += 1  # Increment the agent ID variable
        

        def calculate_weight(agent, neighbor):
            diffs = [abs(agent.attitudes[mode] - neighbor.attitudes[mode]) for mode in modes_attitudes]
            weight = np.sqrt(sum([diff ** 2 for diff in diffs]))

            return weight  
        
        def calculate_similarity(agent, neighbor, weights):
            similarity_score = 0
            #print("checking similarity")
            for attribute in agent.characteristics:
                #print(attribute)
                if attribute == 'c_age':
                    # Check if the difference in age is within 10 years
                    #print("checking age")
                    age_diff = abs(agent.characteristics['c_age'] - neighbor.characteristics['c_age'])
                    if age_diff > age_diff_similarity:
                        continue
                    else:
                        similarity_score += weights[attribute] #* (1 - age_diff / 5)
                else:
                    if agent.characteristics[attribute] == neighbor.characteristics[attribute]:
                        similarity_score += weights[attribute]
            return similarity_score   
        
        for agent_id in range(self.num_agents):
            for other_id in range(self.num_agents):
                if (agent_id != other_id) and not (self.G.has_edge(agent_id, other_id)):
                    agent = self.schedule.agents[agent_id]
                    
                    neighbor = self.schedule.agents[other_id]
                    if self.network_type == 'agent_characteristics':
                        sum_diffs = calculate_similarity(agent, neighbor, char_weights)
                        #print(sum_diffs)
                                    
                        if random.random() < sum_diffs:
                            weight = calculate_weight(agent, neighbor)
                            #print(weight)
                            self.G.add_edge(agent_id, other_id, weight=weight)
                    
                    if self.network_type == 'random':
                        if random.random() < self.cluster_factor:
                            self.G.add_edge(agent_id, other_id, weight=1)

        self.pos = nx.spring_layout(self.G)

 

    def step(self):
        def count_modes():
            counts = {}
            for mode in modes:
                counts[mode] = sum(agent.main_mode == mode for agent in self.all_agents)
            #print(counts)
            self.mode_counts.loc[self.currentStep] = counts # - 1] = counts

        def count_pref():
            pref_counts_1 =[]
            pref_counts_2 =[]
            pref_counts_1 = sum(agent.main_mode == agent.preference_1 for agent in self.all_agents)
            pref_counts_2 = sum(agent.main_mode == agent.preference_2 for agent in self.all_agents)
            self.pref_counts.loc[self.currentStep,['pref_1']] = pref_counts_1 
            self.pref_counts.loc[self.currentStep,['pref_2']] = pref_counts_2

        def count_pref():
            # Initialize empty dictionary to store counts
            counts_dict = {col: {mode: 0 for mode in modes} for col in ['pref_1', 'pref_2']}
            
            for agent in self.all_agents:
                # Increment count for preference 1
                if agent.main_mode == agent.preference_1:
                    counts_dict['pref_1'][agent.preference_1] += 1
                elif agent.main_mode == agent.preference_2:
                    # Increment count for preference 2
                    counts_dict['pref_2'][agent.preference_2] += 1
            
            # Convert counts dictionary to a pandas dataframe
            counts_df = pd.DataFrame(counts_dict)
            
            self.pref_counts[self.currentStep] = counts_df.stack().T


        self.schedule.step()
        count_modes()
        count_pref()
        self.currentStep += 1  

        draw_main_mode = False
        
        if draw_main_mode:
            if self.currentStep == 1:
                viz_methods.visualize_graph_attribute_network_static(self, "main_mode", "chosen mode", self.currentStep)

            if self.currentStep == (self.step_number -1):
                viz_methods.visualize_graph_attribute_network_static(self, "main_mode", "chosen mode", self.currentStep)

