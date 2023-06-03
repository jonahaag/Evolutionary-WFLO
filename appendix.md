# Appendix

## Implementation details

The basic implementation follows the description in the report.
In the following, some additional aspects are highlighted.

### Relevant parameters

Relevant parameters include
- the domain length L and height H
- the choice of the permeability parameter Kinv
- the minimum number of iterations and stopping criterion for the optimization
- the minimum number of iterations and stopping criterion for any individual wind farm simulation
- the number of turbines
- the evolutionary parameters population size, number of parents, and mutation rate
- the minimum spacing between turbines

These parameters remained unchanged during the experiments.

### FEniCS

The basic setup for the Navier-Stokes Brinkman equations follows the template from the course repository.
The permeability is implemented as a `UserExpression` and simply checks if a point (x,y) is within a rectangle around any of the turbine centers.
The mesh is refined around the turbine centers until `hmin` is smaller than the turbine width.
In all experiments, this resulted in a single refinement which proved to be sufficient for the considered use.

### Evolutionary algorithm

For the evolutionary algorithm, many possible realizations of the recombination and mutation mechanisms exist.
Regarding the recombination, the used approach was to simply pick two parents and then iteratively build the child layout by randomly choosing a location of one of the turbines of one of the parents and adding it to the child.
To obtain more feasible layouts, the chosen location must satisfy the spacing constraint, otherwise it is dismissed.
Mutation is performed by simply iterating over each turbine and in each iteration drawing a random number to decide whether it should be moved or not.

The initial layouts are created by drawing random number with NumPy.

## Additional results

Here, some additional results that did not make the cut for the report are shown.

### 10 Turbines

First, consider the 10 turbine case where the following video highlights the optimization process by showing the fitness history as well as the best layout found so far after each iteration.

https://github.com/jonahaag/Evolutionary-WFLO/assets/44704480/587296c2-d957-4ec5-80d6-ab263e353c48

The associated full fitness history is seen below and appears to be similar to the 14 turbine case in the report.

![fitness_history](https://github.com/jonahaag/Evolutionary-WFLO/assets/44704480/d9e00d3c-cfc5-4431-b5ba-3ef9e6c23aab)

### 14 Turbines

The same video as above for the optimization process of the 14 turbine wind farm looks as follows:

https://github.com/jonahaag/Evolutionary-WFLO/assets/44704480/8f2745bf-5533-46ac-98e2-63fa1784727d

### 18 Turbines

Finally, the fitness history and convergence for the 18 turbine wind farm.
Here, the optimization seems to end prematurely, possibly due to a poorly defined termination condition.

![fitness_history](https://github.com/jonahaag/Evolutionary-WFLO/assets/44704480/9237bb69-dda3-40bd-bc13-0cede6cc2e91)

https://github.com/jonahaag/Evolutionary-WFLO/assets/44704480/3fda23fe-683f-4049-9eb0-3efd1fff0d37

Additionally, we can have a look at the resulting fluid flow, a screenshot of which is shown in the report.
This nicely hightlights that the stopping criterion for the simulation works as intended and illustrates the general flow pattern as well as the development of the wake effect.

https://github.com/jonahaag/Evolutionary-WFLO/assets/44704480/9c500a79-89ea-45e3-b204-e6b35ac1b9dd
