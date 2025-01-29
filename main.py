import random

import simpy

from src.warehouse import Warehouse
import simulation_data as sim_data

def main():
    random.seed(42)
    env: simpy.Environment = simpy.Environment()
    warehouse: Warehouse = Warehouse(
        env,
        sim_data.simulation_parameters,
        [sim_data.first_product, sim_data.second_product]
    )
    env.run(until=24*50) # 50 giorni

    print(warehouse.total_cost)


if __name__ == "__main__":
    main()