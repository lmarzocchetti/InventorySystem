import simpy
import random

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from .utils import SimulationParaters, Product

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
        s_max: float = 40,
        s_min: float = 20
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
        self.s_max: float = s_max
        self.s_min: float = s_min

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
   
    def inventory_monitor(self):
        while True:
            yield self.env.timeout(self.inventory_check_interval)
            for idx, product in enumerate(self.products):
                if self.current_inventory_level_products[idx] < self.s_min:
                    self.env.process(self.order_up_to_s_max(product, idx))

    def order_up_to_s_max(self, product: Product, idx: int):
        z = self.s_max - self.current_inventory_level_products[idx]
        self.total_order_cost += (self.order_setup_cost + self.order_incremental_cost * z)
        lead_time = random.uniform(product.lead_time_min, product.lead_time_max)
        yield self.env.timeout(lead_time)
        self.inventory_level_setter(idx, "add", z)

    def demand_generator(self):
        while True:
            demand_inter_arrival_time = random.expovariate(lambd=self.demand_inter_arrival_mean_time)
            # For every product that warehouse handle
            for idx, product in enumerate(self.products):
                pop, weights = product.demand_distribution
                demand_size = random.choices(pop, weights=weights, k=1)[0]

                yield self.env.timeout(demand_inter_arrival_time)
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
            