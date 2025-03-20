import random
import statistics
import time
from collections import deque
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import simpy
import tqdm
import matplotlib.pyplot as plt

import simulation_data as sim_data
from src.warehouse import Warehouse
from src.dqn import ReinforcementWarehouse
import src.actor_critic as ac

def set_seed(seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main_rl():
    set_seed()

    # TODO: Numero giorni rimanenti per la consegna degli ordini
    # I1, I2, O1, O2, D1, D2.
    # Stock attuali, Ordini pendenti, Domanda giorno precedente
    in_dim: int = 6
    # POsso normalizzare gli stock tra 0 e 1 
    # posso normalizzare anche il reward con un massimo empirico

    # r1, r2
    # quantita ordinate prodotto 1 e 2
    # vincoli ordine non puo superare capacita massima: I1 + I2 + O1 + O2 + r1 + r2 <= C
    # r1 e r2 appartengono a {0, 1, ..., M} M quantita massima ordinabile
    out_dim: int = 2

    reinforcement_class = ReinforcementWarehouse(in_dim, out_dim)

    steps_done = 0

    returns = []
    epsilons = []
    for num_episode in tqdm.tqdm(range(reinforcement_class.num_of_episodes)):
        env = simpy.Environment()
        warehouse = Warehouse(
            env,
            sim_data.simulation_parameters,
            [sim_data.first_product, sim_data.second_product],
            reinforcement_learning=reinforcement_class
        )
        state = (60, 60, 0, 0, 0, 0)
        total_reward = 0
        day = 1
        
        while True:
            reinforcement_class.steps_done = steps_done
            reinforcement_class.state = state
            print(f"EPSILON: {reinforcement_class.epsilon}")
            # epsilon = reinforcement_class.calculate_epsilon(steps_done)
            # action = reinforcement_class.select_action(state, epsilon, env.action_space)
            env.run(until=24*day + 1)

            action = reinforcement_class.action
            next_state, reward, done, truncated = reinforcement_class.return_value
            total_reward += reward

            reinforcement_class.memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(reinforcement_class.memory) >= reinforcement_class.batch_size:
                batch = random.sample(reinforcement_class.memory, reinforcement_class.batch_size)
                reinforcement_class.optimize_model(batch, 0.99)

            if done or truncated:
                break

            steps_done +=1
            day += 1
        
        if num_episode % reinforcement_class.target_update == 0:
            reinforcement_class.load_new_dict()
        
        returns.append(total_reward)
        epsilons.append(reinforcement_class.epsilon)

    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Episode Return')
    plt.title('DQN on CartPole-v1')
    plt.show()

    # plt.plot(epsilons)
    # plt.xlabel('Episode')
    # plt.ylabel('Episode final epsilon value')
    # plt.title('DQN on CartPole-v1')
    # plt.show()

def main_actor_critic():
    set_seed()
    state_dim: int = 6
    action_dim: int = 2
    max_action: int = 200
    num_of_days_for_episode = 50000

    agent = ac.DDPG(state_dim, action_dim, max_action)
    # Fare un training su numero di timesteps
    num_episodes = 20000
    day = 1
    returns = []
    total_costs = []
    for episode in tqdm.tqdm(range(num_episodes)):
        state = (max_action, max_action, 0, 0, 0, 0)
        agent.state = None
        agent.reward = None
        agent.action = None
        agent.done = False

        env = simpy.Environment()
        warehouse = Warehouse(
            env,
            sim_data.simulation_parameters,
            total_inventory_level_per_product=max_action,
            products=[sim_data.first_product, sim_data.second_product],
            reinforcement_learning=agent
        )
        episode_reward = 0

        for t in range(num_of_days_for_episode):
            action = agent.select_action(state, noise_scale=0.05)
            agent.action = action
            env.run(until=24*day + 1)
            next_state, reward, done = agent.state, agent.reward, agent.done

            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train(batch_size=64)

            state = next_state
            episode_reward += reward

            if done:
                break

            day += 1
        
        # print(f"Episode reward: {episode_reward}")

        returns.append(episode_reward)
        total_costs.append(warehouse.total_cost)
    
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Episode Return')
    plt.title('DQN on CartPole-v1')
    plt.show()

def main():
    #TODO: Separare gli s_min s_max per ogni prodotto
    set_seed()
    params = [
        # (s_min, s_max)
        (20, 40),
        (20, 60),
        (20, 80),
        (20, 100),
        (40, 60),
        (40, 80),
        (40, 100),
        (60, 80),
        (60, 100)
    ]
    params_separate = list(product(params, params))

    results_prod_1 = {}
    results_prod_2 = {}
    for s_min_max in params_separate:
        for _ in range(30):
            env = simpy.Environment()
            warehouse = Warehouse(
                env,
                sim_data.simulation_parameters,
                [sim_data.first_product, sim_data.second_product],
                s_min_max=list(s_min_max)
            )
            env.run(until=24*50)
            # results.setdefault(s_min, {}).setdefault(s_max, []).append(warehouse.total_cost)
            results_prod_1.setdefault(s_min_max[0][0], {}).setdefault(s_min_max[0][1], []).append(warehouse.total_cost)
            results_prod_2.setdefault(s_min_max[1][0], {}).setdefault(s_min_max[1][1], []).append(warehouse.total_cost)
    
    for s_min, d in results_prod_1.items():
        s_maxs = list(d.keys())
        average_total_costs = [statistics.mean(l) for l in d.values()]
        plt.plot(s_maxs, average_total_costs, label=f's_min: {s_min}')
    
    for s_min, d in results_prod_2.items():
        s_maxs = list(d.keys())
        average_total_costs = [statistics.mean(l) for l in d.values()]
        plt.plot(s_maxs, average_total_costs, label=f's_min: {s_min}')

    plt.title('Average Total Cost over s_max')
    plt.xlabel('s_max')
    plt.ylabel('Average Total Cost')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # main()
    # main_rl()
    main_actor_critic()