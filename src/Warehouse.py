import simpy
import random

from collections import defaultdict

from utils import SimulationParaters, Product

class Warehouse:
    def __init__(
        self,
        env: simpy.Environment,
        parameters: SimulationParaters,
        products: list[Product],
        inventory_level: float = 60,
        inventory_check_interval: float = 1,
        s_max: float = 40,
        s_min: float = 20
    ) -> None:
        self.env = env
        self.demand_inter_arrival_mean_time: float = parameters.demand_inter_arrival_mean_time
        self.order_setup_cost: float = parameters.order_setup_cost
        self.order_incremental_cost: float = parameters.order_incremental_cost
        self.holding_cost: float = parameters.holding_cost
        self.shortage_cost: float = parameters.shortage_cost
        self.products: list[Product] = products
        self.inventory_level: float = 60
        self.inventory_check_interval: float = inventory_check_interval
        self.s_max: float = s_max
        self.s_min: float = s_min
    
        self.total_order_cost = 0
        self.inventory_history: dict[float, float] = defaultdict(float)
        
        self.it: list[tuple[float, float]] = [(0, inventory_level)]
        self.last_inventory_level: float = inventory_level
        self.last_inventory_level_timestamp: float = 0.0

