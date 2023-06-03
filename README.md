# Evolutionary-WFLO

This repository contains the content and results of the project for the course 'Advanced Computations in Fluid Mechanics' at the Royal Institute of Technology (KTH) in the spring term 2023.
It concerns the Wind Farm Layout Optimization (WFLO) problem, where, for a given number of turbines in a wind farm, the arrangement with the largest associated production is sought.
For this, an evolutionary algorithm is used to generate candidate layouts.
The project is implemented in Python using NumPy, Matplotlib and the finite element library FEniCS.

## Basic structure of the repository and code

The implementation consists of
- `main.py` which initializes the relevant parameters and controls the optimization process,
- `wfsim.py` which contains the basic wind farm simulator using FEniCS,
- `utils.py` which contains additional utilities for the wind farm simulator and the optimization process, and
- `evolutionary_utils.py` which contains the basic utilities for the evolutionary algorithm.

## Report

The report explains the general methodology, describes and analyzes the results and tries to evaluate the suitability of the used approach.

Additional results as well as details regarding the implementation are discussed in [`appendix.md`](appendix.md).