from src.utils import Product, SimulationParaters

simulation_parameters = SimulationParaters(
    demand_inter_arrival_mean_time = 0.1,
    order_setup_cost = 10,
    order_incremental_cost = 3,
    holding_cost = 1,
    shortage_cost = 7
)

first_product = Product(
    name = "First Product",
    demand_distribution = ([1, 2, 3, 4], [1/6, 1/3, 1/3, 1/6]),
    lead_time_min = 0.5,
    lead_time_max = 1.0
)

second_product = Product(
    name = "Second Product",
    demand_distribution = ([5, 4, 3, 2], [1/8, 1/2, 1/4, 1/8]),
    lead_time_min = 0.2,
    lead_time_max = 0.7
)