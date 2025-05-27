import simpy
import random

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from .utils import SimulationParaters, Product
from .reinforce import ReinforceAgent
from .dqn import DQNAgent

def number():
    num = 0
    while True:
        yield num
        num = num + 1

num = number()

# TODO: inventory level separated by product, so not assuming that every product occupy 1 slot: Fatto
# TODO: s_min and s_max customizable for every product: Fatto
# TODO: provare a passare a contare 1 come time-step e non 24 (un giorno)
class Warehouse:
    def __init__(
        self,
        env: simpy.Environment,
        parameters: SimulationParaters,
        products: list[Product],
        initial_inventory_per_product: list[int] = [20, 20],
        inventory_check_interval: float = 24,
        s_min_max: list[tuple[int, int]] = [],
        reinforcement_learning: None | ReinforceAgent | DQNAgent = None,
        eval_mode: bool = False
    ) -> None:
        """
        Warehouse that stores a fixed amount of products
        """
        self.env: simpy.Environment = env
        self.demand_inter_arrival_mean_time: float = parameters.demand_inter_arrival_mean_time
        self.order_setup_cost: float = parameters.order_setup_cost
        self.order_incremental_cost: float = parameters.order_incremental_cost
        self.holding_cost: float = parameters.holding_cost
        self.shortage_cost: float = parameters.shortage_cost
        self.products: list[Product] = products
        self.inventory_check_interval: float = inventory_check_interval
        self.s_min_max: list[tuple[int, int]] = s_min_max
        self.reinforcement_learning: ReinforceAgent | DQNAgent | None = reinforcement_learning
        self.eval_mode = eval_mode
        
        self.consecutive_stockout_days: int = 0

        self.pending_orders: list[int, dict[int, int]] = [{}, {}]
        self.last_day_order_request: list[int] = [0, 0]

        self.current_inventory_level_products = [initial_inventory_per_product[i] for i in range(len(self.products))]
        assert len(self.current_inventory_level_products) == len(self.products)

        self.total_order_cost = 0
        self.inventory_history: dict[int, dict[float, float]] = {idx:defaultdict(float) for idx in range(len(self.products))}
        
        self.it: dict[int, list[tuple[int, int]]] = {idx:[(0, initial_inventory_per_product[idx])] for idx in range(len(self.products))}
        self.last_inventory_level: dict[int, int] = {idx:initial_inventory_per_product[idx] for idx in range(len(self.products))}
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

    def inventory_monitor(self):
        while True:
            # Every Day (24 hours)
            yield self.env.timeout(self.inventory_check_interval)
            if self.reinforcement_learning is None:
                # S_min S_max policy 
                for idx, product in enumerate(self.products):
                    if self.current_inventory_level_products[idx] < self.s_min_max[idx][0]:
                        self.env.process(self.order_up_to_s_max(product, idx))
            else:
                action = self.reinforcement_learning.action

                inventory_1 = max(0, self.current_inventory_level_products[0] - self.last_day_order_request[0])
                inventory_2 = max(0, self.current_inventory_level_products[1] - self.last_day_order_request[1])
                unmet_demand_1 = max(0, self.last_day_order_request[0] - self.current_inventory_level_products[0])
                unmet_demand_2 = max(0, self.last_day_order_request[1] - self.current_inventory_level_products[1])
                
                if not self.eval_mode:
                    #TODO: rivedere i coefficienti: Fatto
                    # holding_cost = self.holding_cost * (inventory_1 + inventory_2)
                    holding_cost = self.holding_cost * (((inventory_1 + inventory_2) + self.reinforcement_learning.last_day_inventory) / 2)
                    
                    # TODO: Usare la media dei prodotti da inizio giorno e fine giorno
                    stockout_cost = self.shortage_cost * (unmet_demand_1 + unmet_demand_2)
                    
                    # TODO: ORdinare solo se action sono MAGGIORE DI 0: Fatto
                    order_cost = 0
                    for idx, act in enumerate(action):
                        if act > 0:
                            order_cost += self.order_setup_cost + self.order_incremental_cost * act
                    self.total_order_cost += order_cost

                    reward = - (holding_cost + stockout_cost + order_cost)
                    #TODO: levare parte min_safe_stock: Fatto       
                    self.reinforcement_learning.reward = reward
                
                state = np.array((0 if self.current_inventory_level_products[0] <= 0 else self.current_inventory_level_products[0],
                                  0 if self.current_inventory_level_products[1] <= 0 else self.current_inventory_level_products[1],
                                  0 if self.current_inventory_level_products[0] >= 0 else self.current_inventory_level_products[0],
                                  0 if self.current_inventory_level_products[1] >= 0 else self.current_inventory_level_products[1],
                                  sum(self.pending_orders[0].values()),
                                  sum(self.pending_orders[1].values())
                ))
                
                self.reinforcement_learning.state = state

                self.last_day_order_request = [0, 0]

                self.env.process(self.basic_order(0, action[0]))
                self.env.process(self.basic_order(1, action[1]))

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

    def product_demand_generator(self, idx, product):
        demand_inter_arrival_time = random.expovariate(lambd=self.demand_inter_arrival_mean_time)
        pop, weights = product.demand_distribution
        demand_size = random.choices(pop, weights=weights, k=1)[0]
        id = next(num)
        self.pending_orders[idx][id] = demand_size
        self.last_day_order_request[idx] += demand_size
        
        yield self.env.timeout(demand_inter_arrival_time)
        del self.pending_orders[idx][id]
        self.inventory_level_setter(idx, "sub", demand_size)

    def demand_generator(self):
        while True:
            # TODO: inserted this in for to simulate two different arrival time for each product: Fatto            
            # For every product that warehouse handle
            for idx, product in enumerate(self.products):
                yield from self.product_demand_generator(idx, product)

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
            