import simpy
import random

from dataclasses import dataclass

@dataclass
class Product:
    """ Class that specify the features of a Product """
    name: str
    demand_distribution: tuple[list[int], list[float]]
    lead_time_min: float
    lead_time_max: float

class Warehouse:
    def __init__(
        self,
        env: simpy.Environment,
        products: list[Product],
        demand_inter_arrival_mean_time: float = 0.1,
    ) -> None:
        pass