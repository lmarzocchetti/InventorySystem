import random
import statistics
import pickle
from argparse import ArgumentParser
from itertools import product

import numpy as np
import torch
import simpy
import tqdm
import matplotlib.pyplot as plt

import simulation_data as sim_data
from src.warehouse import Warehouse
from src.dqn import DQNAgent
from src.reinforce import ReinforceAgent

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_list_to_file(list_to_save, filename):
    with open(filename, "wb") as f:
        pickle.dump(list_to_save, f)

def main_smin_smax():
    #TODO: Separare gli s_min s_max per ogni prodotto: Fatto
    set_seed(1923674)
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
        
        total_costs = {}
        
        for _ in range(30):
            env = simpy.Environment()
            warehouse = Warehouse(
                env,
                sim_data.simulation_parameters,
                [sim_data.first_product, sim_data.second_product],
                s_min_max=list(s_min_max)
            )
            env.run(until=24*days)
            results_prod_1.setdefault(s_min_max[0][0], {}).setdefault(s_min_max[0][1], []).append(warehouse.total_cost)
            results_prod_2.setdefault(s_min_max[1][0], {}).setdefault(s_min_max[1][1], []).append(warehouse.total_cost)
            if s_min_max in total_costs: 
                total_costs[s_min_max] += warehouse.total_cost
            else:
                total_costs[s_min_max] = warehouse.total_cost
        
        total_costs[s_min_max] = total_costs[s_min_max] / 30
    
    print(f"S_min S_max minimum: {total_costs[min(total_costs, key=lambda x: x[1])]}")
    
    for s_min, d in results_prod_1.items():
        s_maxs = list(d.keys())
        average_total_costs = [statistics.mean(l) for l in d.values()]
        plt.plot(s_maxs, average_total_costs, label=f'prod_1 s_min: {s_min}')
    
    for s_min, d in results_prod_2.items():
        s_maxs = list(d.keys())
        average_total_costs = [statistics.mean(l) for l in d.values()]
        plt.plot(s_maxs, average_total_costs, label=f'prod_2 s_min: {s_min}')
    
    plt.title('Average Total Cost over s_max')
    plt.xlabel('s_max')
    plt.ylabel('Average Total Cost')
    plt.legend()
    plt.show()

def main_rl(save_path: str):
    set_seed()
    
    init_inv: list[int] = [20, 20]

    # TODO: Numero giorni rimanenti per la consegna degli ordini
    
    # TODO: Avere 2 scalari per prodotto uno per quando ho i prodotti, uno per stock-out: Fatto
    
    # TODO: provare REINFORCE policy based: Fatto
    
    # TODO: Domanda Giorno Precedente e' inutile: Fatto
    
    # I1, I2, S1, S2, O1, O2
    # Stock attuali positivi, Stock attuali negativi, Ordini pendenti (ovviamente stock positivi quando uno e' positivo, l'altro e' 0)
    in_dim: int = 6
    # TODO: Posso normalizzare gli stock tra 0 e 1: Provato
    # TODO: posso normalizzare anche il reward con un massimo empirico: Provato a fare

    # Out Dim: max_action_for_single_product -> basically an index in a matrix that represent
    # every couple of (action_1, action_2) from (0, 0) to (25, 25) -> is calculated inside the agent
    
    max_action_for_single_product: int = 25 # include the (0, 0) to (25, 25) inclusive 
    
    num_of_days_for_episode = 2_000
    num_episodes = 1_000
    agent: DQNAgent = DQNAgent(in_dim, max_action_for_single_product, device, hidden_dim=512)
    
    # time_steps = num_episodes * num_of_days_for_episode
    
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
            episode_reward += reward if reward is not None else 0
            agent.memory.push(state, action_index, reward, next_state)
            agent.train_step()
            state = next_state
        
        returns.append(episode_reward)
        total_costs.append(warehouse.total_cost)
        episode_product.append(statistics.mean(products_per_day))

        # print(f"LOSS FN: {agent.loss_fn_values[-1]}")

        if (episode % 100 == 0 and episode != 0) or episode == num_episodes - 1:
            agent.save_model(f'{save_path}/DQN_{episode}.pt')
    
    # TODO: plottare la quantita dei prodotti medi per ogni episodio (positivi + stock-out): Fatto
    # TODO: plottare i costi: Fatto
    # TODO: plottare loss_function: Fatto
    # TODO: plottare i returns: Fatto
    
    # TODO: normalizzare i reward (moltiplicare * 10^x, oppure tra 0 1): Provato a fare
    # TODO: normalizzare gli input: Provato a fare
    
    # Saving to file
    save_list_to_file(episode_product, "DQN_episode_product.pkl")
    save_list_to_file(total_costs, "DQN_total_costs.pkl")
    save_list_to_file(agent.loss_fn_values, "DQN_loss_function_values.pkl")
    save_list_to_file(returns, "DQN_returns.pkl")
    
    # Plot
    _, ((ax_returns, ax_loss_fn), (ax_total_costs, ax_episode_products)) = plt.subplots(2, 2, figsize=(8, 8))
    
    ax_returns.plot(returns)
    ax_returns.set_xlabel('Episode')
    ax_returns.set_ylabel('Episode Reward')
    ax_returns.set_title('DQN Rewards')
    
    ax_loss_fn.plot(agent.loss_fn_values)
    ax_loss_fn.set_xlabel('Episode')
    ax_loss_fn.set_ylabel('Loss value')
    ax_loss_fn.set_title('DQN Loss function values')
    
    ax_total_costs.plot(total_costs)
    ax_total_costs.set_xlabel('Episode')
    ax_total_costs.set_ylabel('total cost')
    ax_total_costs.set_title('DQN total costs')
    
    ax_episode_products.plot(episode_product)
    ax_episode_products.set_xlabel('Episode')
    ax_episode_products.set_ylabel('Total Inventory')
    ax_episode_products.set_title('DQN Total products')
    
    plt.show()


def main_rl_test(model: str):
    set_seed(1923674)
    
    init_inv: list[int] = [20, 20]
    in_dim: int = 6
    max_action_for_single_product: int = 25 # include the (0, 0) to (25, 25) inclusive 
    
    num_of_days_for_episode = 365
    num_episodes = 30
    
    agent: DQNAgent = DQNAgent(in_dim, max_action_for_single_product, device, hidden_dim=512)
    # agent.load_model("old_train/dqn_train/model_400.pt")
    agent.load_model(model)
    
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

    print(f"DQN mean total cost {statistics.mean(total_costs)}")
    plt.plot(total_costs)
    plt.xlabel('Episode')
    plt.ylabel('Episode cost')
    plt.title('DQN Warehouse')
    # plt.show()

def main_reinforce(save_path: str):
    set_seed()
    
    init_inv: list[int] = [20, 20]
    in_dim: int = 6
    max_action_for_single_product: int = 25 # include the (0, 0) to (25, 25) inclusive 
    
    num_of_days_for_episode = 2_000
    num_episodes = 1_000

    agent: ReinforceAgent = ReinforceAgent(in_dim, max_action_for_single_product, device, hidden_dim=1024)

    total_costs = []
    rewards = []
    episode_product = []
    for episode in tqdm.tqdm(range(num_episodes)):
        state = np.array((init_inv[0], init_inv[1], 0, 0, 0, 0))
        agent.state = state
        agent.reward = 0
        agent.action = None
        
        log_probs = []
        rewards = []
        rewards_to_print = []
        entropies = []
        
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
            action, _, log_prob, entropy = agent.select_action(state)
            agent.action = action
                        
            # Salvo inizio giorno inventario
            inventory_1 = max(0, warehouse.current_inventory_level_products[0] - warehouse.last_day_order_request[0])
            inventory_2 = max(0, warehouse.current_inventory_level_products[1] - warehouse.last_day_order_request[1])
            agent.last_day_inventory = inventory_1 + inventory_2
            products_per_day.append(inventory_1 + inventory_2)
            
            env.run(env.now + 24)
            next_state, reward = agent.state, agent.reward

            # if reward is not None:
            episode_reward += reward
            rewards.append(torch.tensor(reward, dtype=torch.float).to(device))
            rewards_to_print.append(reward)
            log_probs.append(log_prob)
            entropies.append(entropy)      
            
            state = next_state

        #print(f"ENTROPY MEAN: {torch.stack(entropies).mean().item()}")
        agent.train_step(rewards, log_probs, entropies)
        
        episode_product.append(statistics.mean(products_per_day))
        total_costs.append(warehouse.total_cost)
        rewards.append(episode_reward)

        # print(f"Policy Loss_fn: {agent.loss_fn_values[-1]}")
        if (episode % 20 == 0 and episode != 0) or episode == num_episodes - 1:
            agent.save_model(f'{save_path}/REINFORCE_{episode}.pt')

    # Saving to file
    save_list_to_file(episode_product, "REINFORCE_episode_product.pkl")
    save_list_to_file(total_costs, "REINFORCE_total_costs.pkl")
    save_list_to_file(agent.loss_fn_values, "REINFORCE_loss_function_values.pkl")
    save_list_to_file(rewards_to_print, "REINFORCE_rewards.pkl")

    # Plot
    _, ((ax_returns, ax_total_costs), (ax_loss_fn_actor, ax_episode_products)) = plt.subplots(2, 2, figsize=(8, 8))
    
    ax_returns.plot(rewards_to_print)
    ax_returns.set_xlabel('Episode')
    ax_returns.set_ylabel('Reward')
    ax_returns.set_title('Actor Critic Reward')
    
    ax_loss_fn_actor.plot(agent.loss_fn_values)
    ax_loss_fn_actor.set_xlabel('Episode')
    ax_loss_fn_actor.set_ylabel('Loss Values Policy')
    ax_loss_fn_actor.set_title('Policy Loss function')
        
    ax_total_costs.plot(total_costs)
    ax_total_costs.set_xlabel('Episode')
    ax_total_costs.set_ylabel('Total Cost')
    ax_total_costs.set_title('Reinforce Total Costs')
    
    ax_episode_products.plot(episode_product)
    ax_episode_products.set_xlabel('Episode')
    ax_episode_products.set_ylabel('Inventory')
    ax_episode_products.set_title('Reinforce total products')
    
    plt.show()

def main_reinforce_test(model: str):
    set_seed(1923674)
    
    init_inv: list[int] = [20, 20]
    in_dim: int = 6
    max_action_for_single_product: int = 25 # include the (0, 0) to (25, 25) inclusive 
    
    num_of_days_for_episode = 365
    num_episodes = 30
    
    agent: ReinforceAgent = ReinforceAgent(in_dim, max_action_for_single_product, device, hidden_dim=1024)
    agent.load_model(model)
    
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
            action, _, _, _ = agent.select_action(state)
            agent.action = action
            
            env.run(env.now + 24)
            
            next_state = agent.state
            state = next_state
        
        total_costs.append(warehouse.total_cost)

    print(f"Reinforce mean total cost {statistics.mean(total_costs)}")
    return statistics.mean(total_costs)
    # plt.plot(total_costs)
    # plt.xlabel('Episode')
    # plt.ylabel('Episode cost')
    # plt.title('DQN Warehouse')
    # plt.show()

def normalize_state(state):
    i1, i2, s1, s2, o1, o2 = state
    return np.array((i1/100, i2/100, s1/100, s2/100, o1/10, o2/10))

def main():
    arg_parser: ArgumentParser = ArgumentParser(description="Warehouse simulation with smin_smax policy or reinforcement learning agent")
    subparsers = arg_parser.add_subparsers(dest="command", required=True, help="Train or Test your model")
    
    parser_train = subparsers.add_parser("train", help="Train your model")
    parser_train.add_argument("--algo", type=str, required=True, choices=["dqn", "reinforce"], help="Model you want to train")
    parser_train.add_argument("--save_path", type=str, required=False, default="", help="Path in which you want to save your model weights")
    
    parser_test = subparsers.add_parser("test", help="Test your model")
    parser_test.add_argument("--algo", type=str, required=True, choices=["dqn", "reinforce", "smin_smax"], help="Model you want to test")
    parser_test.add_argument("--load_path", type=str, required=False, default=".", help="Path to your model weights (the right one or non if choosing smin_smax)")
    
    args = arg_parser.parse_args()
    
    if args.command == "train":
        if args.algo == "dqn":
            main_rl(args.save_path)
        else:
            main_reinforce(args.save_path)
    elif args.command == "test":
        algo = args.algo
        if algo == "smin_smax":
            main_smin_smax()
        elif algo == "dqn":
            main_rl_test(args.load_path)
        else:
            main_reinforce_test(args.load_path)
    else:
        print(f"Error: Command {args.command} not supported. Insert either 'test' or 'train'")
        exit(1)

if __name__ == "__main__":
    main()
    
    # main_smin_smax()
    
    # main_rl()
    # 400
    # main_rl_test("old_train/dqn_train/model_400.pt")
    
    # main_reinforce()
    # 320
    # main_reinforce_test("/home/rhohen/Workspace/InventorySystem/old_train/reinforce_1024_320k_mean_nopolicygammas_0.05/REINFORCE_320.pt")
    # totals_costs = {}
    # for i in range(20, 1000, 20):
    #    totals_cost = main_reinforce_test(f"REINFORCE_{i}.pt")
    #    totals_costs[i] = totals_cost
    #    print(f"Iteration: {i}: total-cost: {totals_costs[i]}")
    # print(min(totals_costs.items(), key= lambda x: x[1]))
    

"""
    Note:
    - Scalare il reward: Ho provato a scalare il reward con vari parametri (1000, 10000, max), tutti hanno funzionato
        peggio del reward intero senza scalare
    - Normalizzare gli stati: C'e' la funziona di normalizzazione degli stati, stando ai miei test non cambia
        praticamente nulla rispetto a lasciarli interi (molto probabilmente non raggiungono mai valori troppo elevati)
"""