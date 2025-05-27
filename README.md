# InventorySystem
Inventory System with simpy, with Reinforcement Learning Agent (DQN and REINFORCE)

## Problem Description
Warehouse wants to stock and sell 2 different Products.
The simulation's parameters are:
- demand_inter_arrival_mean_time = 0.1,
- order_setup_cost = 10,
- order_incremental_cost = 3,
- holding_cost = 1,
- shortage_cost = 7

First product's parameters are:
- name = "First Product",
- demand_distribution = ([1, 2, 3, 4], [1/6, 1/3, 1/3, 1/6]),
- lead_time_min = 0.5,
- lead_time_max = 1.0

Second product's paramters are:
- name = "Second Product",
- demand_distribution = ([5, 4, 3, 2], [1/8, 1/2, 1/4, 1/8]),
- lead_time_min = 0.2,
- lead_time_max = 0.7

## Choosing actions
Obviously we need to control the simulation, so the warehouse need to order
every day (or not), some quantity for the each Product.

### S-min S-max Policy
Simple policy that states: If the stocked quantity of a product is less than
S-min, then order products up to S-max

### Reinforcement Learning
I have implemented from scratch these two methods:

#### DQN
Classic Deep Q Network, so a Value-based approach with Replay Buffer, Temporal Difference

#### REINFORCE
Another classic algorithm, Policy-based approach, Monte-carlo

## Performance
I have optimized each methods to act, empirically, the best that they could. Then i have tested
these methods, to see which is the best (basically minimizing the total_cost after a year):
1. DQN: 248427
2. REINFORCE: 364943
3. S-min S-max: 703995

## Usage
I used Python 3.13.3, but for sure python 3.11 and 3.12 will still works.
```
$ pip install -r requirements.txt
```

### Train
```
$ python main.py train {dqn,reinforce} --save_path {folder}
```

### Test
In case of smin_smax you don't need to specify the load path
```
$ python main.py test {dqn,reinforce,smin_smax} [--load_path {path_to_pt_file}]  
```

### Changing Hyper Parameters
As of right now you need to modify them inside the code, maybe i can introduce others CLI flags to specify them.