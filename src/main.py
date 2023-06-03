from wfsim import WindFarmSimulator
from utils import evaluate_layout, get_velocity_sum, check_layout_validity
from evolutionary_utils import Population
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Domain parameters
L = 5
H = 2.5
resolution = 32
windfarm_subdomain_L = 0.6*L
windfarm_subdomain_H = 0.6*H
wind_farm_shift_x = 0.2*L
wind_farm_shift_y = 0.2*H
n_turbines = 14
n_grid = 9
grid_width = windfarm_subdomain_L / n_grid
grid_height = windfarm_subdomain_H / n_grid
turbine_width = grid_width * 0.15 # TODO compare with grid_width * 0.2
turbine_height = grid_height * 0.8
min_x_spacing = 0.
min_y_spacing = 2 * turbine_height
Kinv = 80000.

print(f'Grid width: {grid_width:.3f}')
print(f'Grid height: {grid_height:.3f}')
print(f'Min x spacing: {min_x_spacing:.3f}')
print(f'Min y spacing: {min_y_spacing:.3f}')
print(f'Turbine width: {turbine_width:.3f}')
print(f'Turbine height: {turbine_height:.3f}')

# Simulation parameters
nu = 1e-4 #1.0e-2
T_min = 2.

# Evolutionary parameters
n_individuals = 15
mutation_rate = 0.2
n_parents = 8
assert n_parents%2 == 0
population = Population(n_individuals, mutation_rate, n_parents,\
                        wind_farm_shift_x, wind_farm_shift_y,\
                        windfarm_subdomain_L, windfarm_subdomain_H,\
                        grid_width, grid_height, n_grid, n_turbines,\
                        min_x_spacing, min_y_spacing)

# Initialize the simulator
windFarmSimulator = WindFarmSimulator(L,H,resolution,nu,Kinv,turbine_width,turbine_height,None,T_min)

# Visualization and result parameters
plot_frequency = 1

history_fitness = []
history_layouts = []
history_u = []
history_p = []
result_path = 'history_test'+str(n_turbines)+'/'

last_global_update = 1
gen_id = 0
print('#'*60)
while last_global_update < 30 or gen_id < 50:
    print('Generation:', gen_id)

    population.create_next_generation()

    population.evaluate(windFarmSimulator)

    history_fitness.append([individual.fitness for individual in population.individuals])
    best_individual_id = np.argmax(history_fitness[-1])
    history_layouts.append(population.individuals[best_individual_id].turbine_centers)
    history_u.append(population.individuals[best_individual_id].u)
    history_p.append(population.individuals[best_individual_id].p)

    if history_u[-1] is not None:
        windFarmSimulator.plot_results(history_u[-1],history_p[-1],gen_id,result_path)
  
    if gen_id >= 2:
        if np.all(history_layouts[-1] == history_layouts[-2]):
            last_global_update += 1
        else:
            last_global_update = 1
    print(f"Current best layout:\n {np.array2string(history_layouts[-1], formatter={'float': lambda x: f'{x:.3f}'}, separator=', ')}")
    print(f'Current best fitness: {history_fitness[-1][best_individual_id]:.4f}')
    print(f'Global solution was updated {last_global_update} iterations ago.')
    print('#'*60)
    gen_id += 1

print('Save results...')
np.save(result_path+'history_fitness.npy', np.array(history_fitness))
np.save(result_path+'history_layouts.npy', np.array(history_layouts))

five_best_individuals_id = np.argsort(history_fitness[-1])[-5:]
for id in five_best_individuals_id:
    windFarmSimulator.plot_results(population.individuals[id].u,population.individuals[id].p,'final_'+str(id),result_path)
windFarmSimulator.set_turbine_centers(population.individuals[five_best_individuals_id[-1]].turbine_centers)
windFarmSimulator.set_up_mesh_with_bc()
u, p = windFarmSimulator.simulate(save_pvd=True)
print('Done.')