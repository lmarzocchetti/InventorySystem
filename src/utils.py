from dataclasses import dataclass

"""
K è il costo di setup, cioè il costo base che paghi ogni volta che realizzi un numero n di nuovi prodotti da 
mettere in magazzino 

i: incremental cost, cioè il costo aggiuntivo che paghi per ognuno di quegli n prodotti
(Esempio, quando nell'inventory system ordini 20 nuovi prodotti, il costo sarà dell'ordine sarà k + n*20)

h: holding cost, cioè quanto ti costa tenere n prodotti fermi in magazzino (se H= 1000€/giorno e in magazzino
hai 20 prodotti, spendi 20*1000€ al giorno)

π: shortage cost, è il costo che paghi quando non riesci a soddisfare la domanda 
(es. con π=200€, se hai 10 unità in magazzino e il cliente te ne ordina 15, spendi (10-15)*S per ogni giorno
che passa prima che riesci a soddisfare la domanda)

"""

@dataclass
class SimulationParaters:
    """ Class that specify the Simulation's features """
    demand_inter_arrival_mean_time: float
    order_setup_cost: float
    order_incremental_cost: float
    holding_cost: float
    shortage_cost: float

@dataclass
class Product:
    """ Class that specify the features of a Product """
    name: str
    demand_distribution: tuple[list[int], list[float]]
    lead_time_min: float
    lead_time_max: float