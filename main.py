import random
import statistics
import pickle
from itertools import product

import numpy as np
import torch
import simpy
import tqdm
import matplotlib.pyplot as plt

import simulation_data as sim_data
from src.warehouse import Warehouse
from src.dqn import DQNAgent

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_list_to_file(list_to_save, filename):
    with open(filename, "wb") as f:
        pickle.dump(list_to_save, f)

def main_rl():
    set_seed()
    
    init_inv: list[int] = [20, 20]

    # TODO: Numero giorni rimanenti per la consegna degli ordini
    
    # TODO: Avere 2 scalari per prodotto uno per quando ho i prodotti, uno per stock-out: Fatto
    
    # TODO: provare REINFORCE policy based
    
    # TODO: Domanda Giorno Precedente e' inutile: Fatto
    
    # I1, I2, S1, S2, O1, O2
    # Stock attuali positivi, Stock attuali negativi, Ordini pendenti (ovviamente stock positivi quando uno e' positivo, l'altro e' 0)
    in_dim: int = 6
    # TODO: Posso normalizzare gli stock tra 0 e 1 
    # TODO: posso normalizzare anche il reward con un massimo empirico

    # Out Dim: max_action_for_single_product -> basically an index in a matrix that represent
    # every couple of (action_1, action_2) from (0, 0) to (25, 25) -> is calculated inside the agent
    
    max_action_for_single_product: int = 25 # include the (0, 0) to (25, 25) inclusive 
    
    num_of_days_for_episode = 2_000
    num_episodes = 1_000
    agent: DQNAgent = DQNAgent(in_dim, max_action_for_single_product, device, hidden_dim=512)
    
    time_steps = num_episodes * num_of_days_for_episode
    
    episode_product = []
    returns = []
    total_costs = []
    for episode in tqdm.tqdm(range(num_episodes)):
        state = np.array((init_inv[0], init_inv[1], 0, 0, 0, 0))
        agent.state = state
        agent.reward = None
        agent.action = None
        
        env = simpy.Environment()
        warehouse = Warehouse(
            env,
            sim_data.simulation_parameters,
            initial_inventory_per_product=init_inv,
            products=[sim_data.first_product, sim_data.second_product],
            reinforcement_learning=agent
        )
        
        episode_reward = 0
        products_per_day = []
        
        for _ in range(num_of_days_for_episode):
            action, action_index = agent.select_action(state)
            agent.action = action
            
            # Salvo inizio giorno inventario
            inventory_1 = max(0, warehouse.current_inventory_level_products[0] - warehouse.last_day_order_request[0])
            inventory_2 = max(0, warehouse.current_inventory_level_products[1] - warehouse.last_day_order_request[1])
            agent.last_day_inventory = inventory_1 + inventory_2
            products_per_day.append(inventory_1 + inventory_2)
            
            env.run(env.now + 24)
            
            next_state, reward = agent.state, agent.reward
            episode_reward = reward
            agent.memory.push(state, action_index, reward, next_state)
            agent.train_step()
            state = next_state
        
        returns.append(episode_reward)
        total_costs.append(warehouse.total_cost)
        episode_product.append(statistics.mean(products_per_day))

        # print(f"LOSS FN: {agent.loss_fn_values[-1]}")

        if (episode % 100 == 0 and episode != 0) or episode == num_episodes - 1:
            agent.save_model(f'model_{episode}.pt')
    
    # TODO: plottare la quantita dei prodotti medi per ogni episodio (positivi + stock-out): Fatto
    # TODO: plottare i costi: Fatto
    # TODO: plottare loss_function: Fatto
    # TODO: plottare i returns: Fatto
    
    # TODO: normalizzare i reward (moltiplicare * 10^x, oppure tra 0 1)
    # TODO: normalizzare gli input 
    
    # Saving to file
    save_list_to_file(episode_product, "episode_product.pkl")
    save_list_to_file(total_costs, "total_costs.pkl")
    save_list_to_file(agent.loss_fn_values, "loss_function_values.pkl")
    save_list_to_file(returns, "returns.pkl")
    
    # Plot
    _, ((ax_returns, ax_loss_fn), (ax_total_costs, ax_episode_products)) = plt.subplots(2, 2)
    
    ax_returns.plot(returns)
    ax_returns.xlabel('Episode')
    ax_returns.ylabel('Episode Return')
    ax_returns.title('DQN on Warehouse')
    
    ax_loss_fn.plot(agent.loss_fn_values)
    ax_loss_fn.xlabel('Episode')
    ax_loss_fn.ylabel('Episode Return')
    ax_loss_fn.title('DQN on Warehouse')
    
    ax_total_costs.plot(total_costs)
    ax_total_costs.xlabel('Episode')
    ax_total_costs.ylabel('Episode Return')
    ax_total_costs.title('DQN on Warehouse')
    
    ax_episode_products.plot(episode_product)
    ax_episode_products.xlabel('Episode')
    ax_episode_products.ylabel('Episode Return')
    ax_episode_products.title('DQN on Warehouse')
    
    plt.show()


def main_rl_test():
    set_seed()
    
    init_inv: list[int] = [20, 20]
    in_dim: int = 6
    max_action_for_single_product: int = 25 # include the (0, 0) to (25, 25) inclusive 
    
    num_of_days_for_episode = 365
    num_episodes = 30
    
    agent: DQNAgent = DQNAgent(in_dim, max_action_for_single_product, device, hidden_dim=512)
    agent.load_model("model_400.pt")
    
    total_costs = []
    for _ in tqdm.tqdm(range(num_episodes)):
        state = np.array((init_inv[0], init_inv[1], 0, 0, 0, 0))
        agent.state = state
        agent.reward = None
        agent.action = None

        env = simpy.Environment()
        warehouse = Warehouse(
            env,
            sim_data.simulation_parameters,
            initial_inventory_per_product=init_inv,
            products=[sim_data.first_product, sim_data.second_product],
            reinforcement_learning=agent,
            eval_mode = True
        )
        
        for t in range(num_of_days_for_episode):
            action, _ = agent.select_action(state, inference=True)
            agent.action = action
            
            env.run(env.now + 24)
            
            next_state = agent.state
            state = next_state
        
        total_costs.append(warehouse.total_cost)

    print(f"mean total cost {statistics.mean(total_costs)}")
    plt.plot(total_costs)
    plt.xlabel('Episode')
    plt.ylabel('Episode cost')
    plt.title('DQN Warehouse')
    plt.show()

def main():
    #TODO: Separare gli s_min s_max per ogni prodotto: Fatto
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
    
    days = 365

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
            env.run(until=24*days)
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
    main()
    # main_rl()
    # main_rl_test()
    