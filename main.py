import random
import statistics
import time

import simpy
import matplotlib.pyplot as plt

from src.warehouse import Warehouse
import simulation_data as sim_data

def main():
    random.seed(int(time.time()))
    params = (
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
    )

    results = {}
    for s_min, s_max in params:
        for _ in range(30):
            env = simpy.Environment()
            warehouse = Warehouse(
                env,
                sim_data.simulation_parameters,
                [sim_data.first_product, sim_data.second_product],
                s_min=s_min,
                s_max=s_max
            )
            env.run(until=24*50)
            results.setdefault(s_min, {}).setdefault(s_max, []).append(warehouse.total_cost)
    
    for s_min, d in results.items():
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