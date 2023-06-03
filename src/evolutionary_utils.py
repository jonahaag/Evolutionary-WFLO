import numpy as np
from utils import evaluate_layout, check_layout_validity, check_single_turbine_validity

class Population():
    def __init__(self, n_individuals, mutation_rate, n_parents, wind_farm_shift_x, wind_farm_shift_y, windfarm_subdomain_L, windfarm_subdomain_H, grid_width, grid_height, n_grid, n_turbines, min_x_spacing, min_y_spacing):
        self.n_individuals = n_individuals
        self.individuals = [Individual() for i in range(n_individuals)]
        self.mutation_rate = mutation_rate
        self.n_parents = n_parents
        self.n_offspring = int(n_parents/2)
        self.n_survivors = int(n_individuals - n_parents - self.n_offspring)
        self.wind_farm_shift_x = wind_farm_shift_x
        self.wind_farm_shift_y = wind_farm_shift_y
        self.windfarm_subdomain_L = windfarm_subdomain_L
        self.windfarm_subdomain_H = windfarm_subdomain_H
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.n_grid = n_grid
        self.n_turbines = n_turbines
        self.min_x_spacing = min_x_spacing
        self.min_y_spacing = min_y_spacing
        self.rng = np.random.default_rng()

    def random_turbine_center(self, n=None):
        if n is None:
            n = self.n_turbines
        grid_positions_x = np.random.randint(self.n_grid, size=n)
        grid_positions_y = np.random.randint(self.n_grid, size=n)
        turbine_centers_x = grid_positions_x * self.grid_width + self.wind_farm_shift_x + self.grid_width/2
        turbine_centers_y = grid_positions_y * self.grid_height + self.wind_farm_shift_y + self.grid_height/2
        turbine_centers = np.array([turbine_centers_x, turbine_centers_y]).T
        return turbine_centers

    def random_individual(self):
        return Individual(self.random_turbine_center())
        
    def create_next_generation(self):
        fitness = np.array([individual.fitness for individual in self.individuals])
        if np.all(fitness == None): # For the very first generation
            self.individuals = [self.random_individual() for _ in range(self.n_individuals)]
        elif fitness.max() > 0.: # At least one valid layout
            parents_id = np.argsort(fitness)[-self.n_parents:]
            survivors_id = np.argsort(fitness)[-self.n_parents-self.n_survivors:-self.n_parents]
            assert len(parents_id) == self.n_parents
            assert len(survivors_id) == self.n_survivors

            valid_parents_id = [id for id in parents_id if fitness[id] > 0.]
            if len(valid_parents_id) < self.n_parents:
                if len(valid_parents_id)%2 == 0: # Not enough valid parents, but an even number
                    # Use the valid parents to create offspring
                    parents = [self.individuals[i] for i in valid_parents_id] 
                    offspring = self.crossover(valid_parents_id)
                    offspring = self.mutate(offspring)
                    # Fill the population with random individuals
                    survivors = [self.random_individual() for _ in range(self.n_individuals - len(parents) - len(offspring))]
                else: # Not enough valid parents, but an odd number
                    # Use the valid parents + one random individual to create offspring
                    valid_parents_id.append(parents_id[-(len(valid_parents_id)+1)])
                    parents = [self.individuals[i] for i in valid_parents_id]
                    offspring = self.crossover(valid_parents_id)
                    offspring = self.mutate(offspring)
                    # Fill the population with random individuals
                    survivors = [self.random_individual() for _ in range(self.n_individuals - len(parents) - len(offspring))]
            else: # Enough valid parents, just create offspring and mutate the survivors
                parents = [self.individuals[i] for i in parents_id]
                offspring = self.crossover(parents_id)
                offspring = self.mutate(offspring)
                survivors = [self.individuals[i] for i in survivors_id]
                survivors = self.mutate(survivors)

            self.individuals = parents + survivors + offspring
        else:
            self.individuals = [self.random_individual() for _ in range(self.n_individuals)]
            
        assert len(self.individuals) == self.n_individuals

    def crossover(self, parents_id):
        offspring = []
        self.rng.shuffle(parents_id)
        for i in range(len(parents_id)//2):
            parent1 = self.individuals[parents_id[i]]
            parent2 = self.individuals[parents_id[-i-1]]
            valid_layout = False
            while valid_layout is False:
                turbine_centers = np.zeros((self.n_turbines,2))
                for j in range(self.n_turbines):
                    if np.random.rand() < 0.5:
                        new_turbine_center = parent1.turbine_centers[j,:]
                    else:
                        new_turbine_center = parent2.turbine_centers[j,:]
                    while check_single_turbine_validity(new_turbine_center, turbine_centers[:j,:], self.min_x_spacing, self.min_y_spacing) is False:
                        new_turbine_center = self.random_turbine_center(1).reshape(-1,)
                    turbine_centers[j,:] = new_turbine_center
                valid_layout = check_layout_validity(turbine_centers, self.min_x_spacing, self.min_y_spacing, self.grid_width, self.grid_height)
            offspring.append(Individual(turbine_centers))
        return offspring

    def mutate(self, individuals):
        for i in range(len(individuals)):
            for j in range(self.n_turbines):
                if np.random.rand() < self.mutation_rate:
                    shift = np.random.randint(4)
                    if shift == 0 and individuals[i].turbine_centers[j,0] < self.wind_farm_shift_x + self.windfarm_subdomain_L - self.grid_width:
                        individuals[i].turbine_centers[j,0] += self.grid_width
                    elif shift == 1 and individuals[i].turbine_centers[j,0] > self.wind_farm_shift_x + self.grid_width:
                        individuals[i].turbine_centers[j,0] -= self.grid_width
                    elif shift == 2 and individuals[i].turbine_centers[j,1] < self.wind_farm_shift_y + self.windfarm_subdomain_H - self.grid_height:
                        individuals[i].turbine_centers[j,1] += self.grid_height
                    elif shift == 3 and individuals[i].turbine_centers[j,1] > self.wind_farm_shift_y + self.grid_height:
                        individuals[i].turbine_centers[j,1] -= self.grid_height
                    individuals[i].mutated = True
        return individuals
    
    def centers_to_grid(self, turbine_centers):
        grid_positions_x = np.round((turbine_centers[:,0] - self.wind_farm_shift_x - self.grid_width/2)/self.grid_width).astype(int)
        grid_positions_y = np.round((turbine_centers[:,1] - self.wind_farm_shift_y - self.grid_height/2)/self.grid_height).astype(int)
        grid_positions = np.array([grid_positions_x, grid_positions_y]).T
        return grid_positions

    def evaluate(self, windFarmSimulator):
        for id in range(self.n_individuals):
            print(f'Individual {id}')
            T_end = 0.
            elapsed_time = 0.
            if self.individuals[id].fitness is None or self.individuals[id].mutated is True:
                fitness, u, p = evaluate_layout(windFarmSimulator, self.individuals[id],\
                                                self.min_x_spacing, self.min_y_spacing,\
                                                self.grid_width, self.grid_height)
                self.individuals[id].fitness = fitness
                self.individuals[id].u = u
                self.individuals[id].p = p
                if self.individuals[id].fitness > 0.:
                    T_end = windFarmSimulator.T_end
                    elapsed_time = windFarmSimulator.elapsed_time
                self.individuals[id].mutated = False
            print(f'Fitness: {self.individuals[id].fitness:.4f}, T: {T_end:.3f}, Elapsed Time: {elapsed_time:.1f}')


class Individual():
    def __init__(self, turbine_centers = None):
        self.turbine_centers = turbine_centers
        self.fitness = None
        self.valid_layout = None
        self.u = None
        self.p = None
        self.mutated = False
    
    def set_turbine_centers(self, turbine_centers):
        self.turbine_centers = turbine_centers

