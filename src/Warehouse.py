import simpy
import random
import math

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from .utils import SimulationParaters, Product
from .actor_critic import DDPG
from .dqn import DQNAgent

def number():
    num = 1000
    while True:
        yield num
        num = num + 1

num = number()

def action_spc():
    while True:
        prod_0 = random.randint(0, 200)
        prod_1 = random.randint(0, 200)
        yield prod_0, prod_1

act = action_spc()

# TODO: inventory level separated by product, so not assuming that every product occupy 1 slot
# TODO: s_min and s_max customizable for every product
class Warehouse:
    def __init__(
        self,
        env: simpy.Environment,
        parameters: SimulationParaters,
        products: list[Product],
        total_inventory_level_per_product: float = 60,
        inventory_check_interval: float = 24, # FIXME: 24 ore
        s_min_max: list[tuple[int, int]] = [],
        reinforcement_learning: None | DDPG | DQNAgent = None
    ) -> None:
        """Warehouse that stores a fixed amount of products

        Args:
            env (simpy.Environment): Simpy environment
            parameters (SimulationParaters): Simulation Parameters (see the class)
            products (list[Product]): List of Products that are stored in the warehouse
            total_inventory_level (float, optional): Amount of total products that can be stored in the warehouse(assuming all products occupy the same space). Defaults to 60.
            inventory_check_interval (float, optional): Amount of time that we check the inventory and reorder products. Defaults to 1.
            s_max (float, optional): Defaults to 40.
            s_min (float, optional): Defaults to 20.
        """ 
        self.env = env
        self.demand_inter_arrival_mean_time: float = parameters.demand_inter_arrival_mean_time
        self.order_setup_cost: float = parameters.order_setup_cost
        self.order_incremental_cost: float = parameters.order_incremental_cost
        self.holding_cost: float = parameters.holding_cost
        self.shortage_cost: float = parameters.shortage_cost
        self.products: list[Product] = products
        self.total_inventory_level_per_product: float = total_inventory_level_per_product
        self.inventory_check_interval: float = inventory_check_interval
        self.s_min_max = s_min_max
        self.reinforcement_learning = reinforcement_learning
        self.consecutive_stockout_days = 0

        self.pending_orders: list[int, dict[int, int]] = [{}, {}]
        # self.last_day_order_request: list[int] = [0, 0]
        self.last_day_order_request = [0, 0]

        # TODO: Try other alternative to this, or customizable by the user
        # Initialize inventory levels per product at total_inventory_level_per_product
        self.current_inventory_level_products = [int(self.total_inventory_level_per_product) for i in range(len(self.products))]
        assert len(self.current_inventory_level_products) == len(self.products)

        self.total_order_cost = 0
        self.inventory_history: dict[int, dict[float, float]] = {idx:defaultdict(float) for idx in range(len(self.products))}
        
        self.it: dict[int, list[tuple[int, int]]] = {idx:[(0, int(total_inventory_level_per_product))] for idx in range(len(self.products))}
        self.last_inventory_level: dict[int, int] = {idx:int(total_inventory_level_per_product) for idx in range(len(self.products))}
        self.last_inventory_level_timestamp: float = 0.0

        self.env.process(self.inventory_monitor())
        self.env.process(self.demand_generator())

    def inventory_level_setter(self, idx: int, operation, value):
        self.inventory_history[idx][self.last_inventory_level[idx]] += self.env.now - self.last_inventory_level_timestamp

        if operation == "add":
            self.current_inventory_level_products[idx] += value
        elif operation == "sub":
            self.current_inventory_level_products[idx] -= value
        else:
            print(f"ERROR: Operation {operation} on setting the inventory level not supported")
            exit(-1)
        
        self.it[idx].append((self.env.now, self.current_inventory_level_products[idx]))

        self.last_inventory_level_timestamp = self.env.now
        self.last_inventory_level[idx] = self.current_inventory_level_products[idx]

    @property
    def total_holding_cost(self) -> float:
        acc = 0
        for idx_prod in range(len(self.products)):
            acc += sum(
                max(inventory_level, 0) * duration * self.holding_cost
                for inventory_level, duration in self.inventory_history[idx_prod].items())
        return acc

    @property
    def total_shortage_cost(self) -> float:
        acc = 0
        for idx_prod in range(len(self.products)):
            acc += sum(
                max(-inventory_level, 0) * duration * self.shortage_cost
                for inventory_level, duration in self.inventory_history[idx_prod].items())
        return acc

    @property
    def total_cost(self) -> float:
        return self.total_order_cost + self.total_holding_cost + self.total_shortage_cost
    
    def calculate_reward(self) -> float:
        pass

    def can_store_item_quantity(self, quantities: np.ndarray) -> bool:
        for idx, value in enumerate(self.current_inventory_level_products):
            if value + quantities[idx] > self.total_inventory_level_per_product:
                return False
        return True

    def inventory_monitor(self):
        while True:
            yield self.env.timeout(self.inventory_check_interval)
            if self.reinforcement_learning is None:
                # S_min S_max policy 
                for idx, product in enumerate(self.products):
                    if self.current_inventory_level_products[idx] < self.s_min_max[idx][0]:
                        self.env.process(self.order_up_to_s_max(product, idx))
            else:
                action = self.reinforcement_learning.action
                # self.total_order_cost += (self.order_setup_cost + self.order_incremental_cost * action[0])
                # self.total_order_cost += (self.order_setup_cost + self.order_incremental_cost * action[1])
                
                # order cost
                # reward = (- (self.order_setup_cost + self.order_incremental_cost * action[0])) - ((self.order_setup_cost + self.order_incremental_cost * action[1]))
                # # holding cost # TODO: Controllare
                # reward += (action[0] + action[1]) * self.holding_cost * (- self.env.now - self.last_inventory_level_timestamp)
                # print(10 *self.holding_cost * (self.current_inventory_level_products[0] - self.last_day_order_request[0]))
                # reward = (-5 * max(0, self.last_day_order_request[0] - (self.current_inventory_level_products[0] - self.last_day_order_request[0]))) - 10 *self.holding_cost * (self.current_inventory_level_products[0] - self.last_day_order_request[0]) - (self.order_setup_cost + self.order_incremental_cost * action[1])
                # reward = (-5 * max(0, self.last_day_order_request[1] - (self.current_inventory_level_products[1] - self.last_day_order_request[1]))) - 10 *self.holding_cost * (self.current_inventory_level_products[1] - self.last_day_order_request[1]) - (self.order_setup_cost + self.order_incremental_cost * action[1])
                # large_order_penalty = 0.05 * (action[0]**2 + action[1]**2)  # Penalità quadratica sugli ordini
                # reward -= large_order_penalty

                # reward = 0.1 * (self.current_inventory_level_products[0] + self.current_inventory_level_products[1]) + 10 * (self.last_day_order_request[0] + self.last_day_order_request[1])
                # print(self.current_inventory_level_products[0])

                inventory_1 = max(0, self.current_inventory_level_products[0] - self.last_day_order_request[0])
                inventory_2 = max(0, self.current_inventory_level_products[1] - self.last_day_order_request[1])
                unmet_demand_1 = max(0, self.last_day_order_request[0] - self.current_inventory_level_products[0])
                unmet_demand_2 = max(0, self.last_day_order_request[1] - self.current_inventory_level_products[1])
                # holding_cost = 10 * (inventory_1 + inventory_2)
                holding_cost = 0.1 * (((inventory_1 + inventory_2) + (inventory_1 + action[0] + inventory_2 + action[1])) / 2)  # euro / prodotto / giorno
                # Usare la media dei prodotti da inizio giorno e fine giorno
                stockout_cost = 10 * (unmet_demand_1 + unmet_demand_2)  # Penalità alta per stockout
                # TODO: ORdinare solo se action sono MAGGIORE DI 0
                order_cost = 0
                for idx, act in enumerate(action):
                    if act > 0:
                        order_cost += self.order_setup_cost + self.order_incremental_cost * act
                self.total_order_cost += order_cost
                
                # print(f"UNMENR: {unmet_demand_1}---{unmet_demand_2}")
                # print(f"CURRENT INVENTORY: {self.current_inventory_level_products}---ACTION: {action}---ORDER COST: {order_cost}---HOLDING COST: {holding_cost}---STOCKOUT COST: {stockout_cost}")
                # order_cost = (self.order_setup_cost + self.order_incremental_cost * action[0]) + (self.order_setup_cost + self.order_incremental_cost * action[1])
                reward = - (holding_cost + stockout_cost + order_cost)
                
                # print(f"ACTIONS: {action}")
                # print(f"REWARD: {reward}")

                # Dentro YourInventoryEnv.step():
                max_consecutive_stockout_days = 10  # Soglia massima di stockout consecutivi
                # min_safe_stock = 40  # Soglia minima di inventario

                # # Controlla stockout consecutivi
                if unmet_demand_1 > 0 or unmet_demand_2 > 0:
                    self.consecutive_stockout_days += 1
                else:
                    self.consecutive_stockout_days = 0

                # Condizioni per terminazione anticipata
                # TODO: RIATTIVARE NEL CASO
                if (
                    self.consecutive_stockout_days >= max_consecutive_stockout_days  # Troppi stockout di fila
                    # or inventory_1 < min_safe_stock  # Inventario troppo basso
                    # or inventory_2 < min_safe_stock
                ):
                    # print("Troppi stockout di fila o inventario troppo basso!")
                    self.reinforcement_learning.done = True
                    reward -= 999999
                else:
                    self.reinforcement_learning.done = False

                # if self.current_inventory_level_products[0] + action[0] > 200 or self.current_inventory_level_products[1] + action[1] > 200:
                #     # print("Superato limite inventario")
                #     self.reinforcement_learning.done = True
                #     reward -= 20000

                # self.reinforcement_learning.reward = reward
                # state = np.array((self.current_inventory_level_products[0], self.current_inventory_level_products[1], sum(self.pending_orders[0].values()), sum(self.pending_orders[1].values()), self.last_day_order_request[0], self.last_day_order_request[1]))
                # # print(f"DEBUG: {self.current_inventory_level_products[0]}---{self.current_inventory_level_products[1]}")
                # self.reinforcement_learning.state = state
                if not self.can_store_item_quantity(action):
                    self.reinforcement_learning.done = True
                    reward -= 999999
                # Aggiungere un reward molto negativo se esco prima
                
                self.reinforcement_learning.reward = reward
                state = np.array((self.current_inventory_level_products[0], self.current_inventory_level_products[1], sum(self.pending_orders[0].values()), sum(self.pending_orders[1].values()), self.last_day_order_request[0], self.last_day_order_request[1]))
                # print(f"DEBUG: {self.current_inventory_level_products[0]}---{self.current_inventory_level_products[1]}")
                self.reinforcement_learning.state = state

                self.last_day_order_request = [0, 0]

                self.env.process(self.basic_order(0, action[0]))
                self.env.process(self.basic_order(1, action[1]))

    # def inventory_monitor(self):
    #     while True:
    #         yield self.env.timeout(self.inventory_check_interval)
    #         if self.reinforcement_learning is None:
    #             # S_min S_max policy 
    #             for idx, product in enumerate(self.products):
    #                 if self.current_inventory_level_products[idx] < self.s_min:
    #                     self.env.process(self.order_up_to_s_max(product, idx))
    #         else:
    #             # Reinforcement Learning
    #             epsilon = self.reinforcement_learning.calculate_epsilon(self.reinforcement_learning.steps_done)
    #             self.reinforcement_learning.epsilon = epsilon
    #             action = self.reinforcement_learning.select_action(self.reinforcement_learning.state, epsilon, act) # TODO: Action space is missing
    #             if type(action) == np.ndarray:
    #                 action_0 = max(0, int(math.ceil(action[0])))
    #                 action_1 = max(0, int(math.ceil(action[1])))
    #                 print(f"act1: {action_0}--act2: {action_1}")
    #                 action = np.ndarray((2,), dtype=np.int32)
    #                 action[0] = action_0 if action_0 <= self.total_inventory_level_per_product else self.total_inventory_level_per_product
    #                 action[1] = action_0 if action_1 <= self.total_inventory_level_per_product else self.total_inventory_level_per_product
                
                
    #             self.total_order_cost += (self.order_setup_cost + self.order_incremental_cost * action[0])
    #             self.total_order_cost += (self.order_setup_cost + self.order_incremental_cost * action[1])
                
    #             # print(f"Action: {action}")

    #             # order cost
    #             reward = (- (self.order_setup_cost + self.order_incremental_cost * action[0])) - ((self.order_setup_cost + self.order_incremental_cost * action[1]))
    #             # holding cost # TODO: Controllare
    #             reward += (action[0] + action[1]) * self.holding_cost * (- self.env.now - self.last_inventory_level_timestamp)
    #             # stockout cost
    #             print(f"REWARD: {reward}")

    #             assert(len(self.last_day_order_request) == 2)
    #             state = np.array((self.current_inventory_level_products[0], self.current_inventory_level_products[1], sum(self.pending_orders[0].values()), sum(self.pending_orders[1].values()), self.last_day_order_request[0], self.last_day_order_request[1]))
    #             self.reinforcement_learning.action = action

    #             if not self.can_store_item_quantity(action):
    #                 self.reinforcement_learning.return_value = (state, 200, False, True)
    #                 return
                
    #             # Set return value
    #             self.reinforcement_learning.return_value = (state, reward, False, False)

    #             self.env.process(self.basic_order(0, action[0]))
    #             self.env.process(self.basic_order(1, action[1]))

    #             # lead_time = random.uniform(product.lead_time_min, product.lead_time_max)
    #             # yield self.env.timeout(lead_time)
    #             # self.inventory_level_setter(0, "add", action[0])
    #             # self.inventory_level_setter(1, "add", action[1])

    def basic_order(self, idx_prod: int, quantity):
        lead_time = random.uniform(self.products[idx_prod].lead_time_min, self.products[idx_prod].lead_time_max)
        yield self.env.timeout(lead_time)
        self.inventory_level_setter(idx_prod, "add", quantity)

    def order_up_to_s_max(self, product: Product, idx: int):
        z = self.s_min_max[idx][1] - self.current_inventory_level_products[idx]
        self.total_order_cost += (self.order_setup_cost + self.order_incremental_cost * z)
        lead_time = random.uniform(product.lead_time_min, product.lead_time_max)
        yield self.env.timeout(lead_time)
        self.inventory_level_setter(idx, "add", z)

    def demand_generator(self):
        while True:
            demand_inter_arrival_time = random.expovariate(lambd=self.demand_inter_arrival_mean_time)
            # For every product that warehouse handle
            # self.last_day_order_request = [0, 0]
            for idx, product in enumerate(self.products):
                pop, weights = product.demand_distribution
                demand_size = random.choices(pop, weights=weights, k=1)[0]
                id = next(num)
                self.pending_orders[idx][id] = demand_size
                self.last_day_order_request[idx] += demand_size
                # self.last_day_order_request.append(demand_size)

                yield self.env.timeout(demand_inter_arrival_time)
                del self.pending_orders[idx][id]
                self.inventory_level_setter(idx, "sub", demand_size)

    def plot_inventory_level(self):
        plt.title("I(t): Inventory level over time")
        plt.xlabel('Simulation Time')
        plt.ylabel('Inventory Length')
        
        for idx_prod, product in enumerate(self.products):
            color = np.random.rand(3,)
            x, y = zip(*self.it[idx_prod])
            plt.step(x, y, where='post', color=color, label=product.name)
            plt.fill_between(x, y, step='post', alpha=0.2, color=color)

        plt.legend(loc='lower right')
        plt.show()
            