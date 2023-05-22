# TransportTransform
The model code for the TransportTransform agent-based model

## Project organization

```
.
├── .gitignore
├── CITATION.md
├── LICENSE.md
├── README.md
├── requirements.txt
├── ABM                <- Source code of the TransportTransform agent-based model
└── script             <- Source code for this project to execude and visualise the TransportTransform model

```

## Python dependencies

Python 3.6+

Other dependencies are specified in the requirements.txt file.

To install them run pip install -r requirements.txt.

It is a good practice to install python dependencies in an isolated virtual environment.

## License

This project is licensed under the terms of the [MIT License](/LICENSE.md)

## Citation

Please [cite this project as described here](/CITATION.md).

## Running the code

1. Import dependencies as described above
2. Generate a dummy dataset using the (/script/random_datat_generator.py)
3. Change the path in the (/script/run_model.py) to the (relative) location of the dummy dataset
4. Run the ABM by importing (/ABM?TT_model.py) it into (/script/run_model.py). This code is avilable in the run_model.py file.
5. Create visulaizations of the model results by importing visualization.py from (/script/run_model.py). This code is avilable in the run_model.py file.

Alternatively, the model can be initialized directly to create a single run with the following code:

```python:
TT_model = TransportModel(include_all, number_agents_included, step_number, preference_1_prob, preference_2_prob, 
                 habit_on, cluster_factor, 
                 occupancy_on, occupancy_function, occupancy_parameter,  
                 norms_on, norm_threshold, norm_prob,)
for i in range(step_number):    
  TT_model.step()
```

The variables are:
```python:
include_all (bool): Indicates whether to include all agents.
number_agents_included (int): Number of agents to include if `include_all` is False.
step_number (int): Total number of steps in the model.
preference_1_prob (float): Probability of preference 1 for agents.
preference_2_prob (float): Probability of preference 2 for agents.
habit_on (bool): Indicates whether habit is enabled for agents.
cluster_factor (float): Cluster factor used for creating agent networks.
    Between 0 and 1; usually around 0.1 and 0.3
occupancy_on (bool): Indicates whether occupancy module is enabled.
occupancy_function (str): Function used to calculate occupancy probability. 
    Options are ; ['threshold', 'linear','exponential','sigmoid','logarithm']
occupancy_parameter (float or tuple): Parameter(s) for the occupancy function.
norms_on (bool): Indicates whether the social norms module is enabled.
norm_threshold (float): Norm threshold value.
    Between 0 and 1; usually around 0.4 and 0.6
norm_prob (float): Probability of norm adherence.
```
