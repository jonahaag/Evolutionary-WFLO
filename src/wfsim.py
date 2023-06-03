import dolfin
from dolfin import *; from mshr import *
import dolfin.common.plotting as fenicsplot
from matplotlib import pyplot as plt
from utils import create_mesh, create_vector_spaces, create_trial_test_functions, create_boundary_conditions, create_variational_problem, WindFarm, Left, Right, Lower, Upper  
from utils import get_velocity_sum
from time import perf_counter
import numpy as np

class WindFarmSimulator():

    def __init__(self,L,H,resolution,nu,Kinv,turbine_width,turbine_height,turbine_centers,T):
        self.L = L
        self.H = H
        self.resolution = resolution
        self.nu = nu
        self.nueff = nu
        self.Kinv = Kinv
        self.turbine_width = turbine_width
        self.turbine_height = turbine_height
        self.turbine_centers = turbine_centers
        self.T = T

        # Set method parameters
        self.num_nnlin_iter = 5 
        self.prec = "amg" if has_krylov_solver_preconditioner("amg") else "default" 

    def simulate(self, save_pvd=False):
        # Create wind farm and permeability tensor Kinv
        windFarm = WindFarm(element=self.K.ufl_element())
        windFarm.set_turbine_centers(centers=self.turbine_centers)
        windFarm.set_turbine_width(width=self.turbine_width)
        windFarm.set_turbine_height(height=self.turbine_height)
        windFarm.set_turbine_porosity(Kinv=self.Kinv)

        Kinv11 = windFarm
        Kinv12 = Expression('0.0', element = self.K.ufl_element())
        Kinv21 = Kinv12
        Kinv22 = Kinv11

        # Define functions
        u0 = Function(self.V)
        u1 = Function(self.V)
        p0 = Function(self.Q)
        p1 = Function(self.Q)

        # Time step length 
        dt = 0.5*self.mesh.hmin()
        t = dt
        filewrite_time = 0.0
        filewrite_freq = 10

        au, Lu, ap, Lp = create_variational_problem(self.mesh, self.u, self.p, self.v, self.q, \
                                                    Kinv11, Kinv12, Kinv21, Kinv22, \
                                                    self.nu, self.nueff, \
                                                    u0, u1, p1, dt)

        # Open files
        if save_pvd:
            file_u = File("results-BNS/u.pvd")
            file_p = File("results-BNS/p.pvd")
        # !rm results-BNS/*

        last_update = 0
        update_tol = 0.001
        conv_tol = 1/dt #we want to stay within update tolerance for at least one second which equals 1/dt iterations
        fitness = 1e-12
        self.fitnesses = []

        tic = perf_counter()
        while t < self.T + DOLFIN_EPS or last_update < conv_tol:
            # Solve non-linear problem 
            k = 0
            while k < self.num_nnlin_iter: 
                
                # Assemble matrix and vector 
                Au = assemble(au)
                bu = assemble(Lu)

                # Compute solution 
                [bc.apply(Au, bu) for bc in self.bcu]
                [bc.apply(u1.vector()) for bc in self.bcu]
                solve(Au, u1.vector(), bu, "bicgstab", "default")

                # Assemble matrix and vector
                Ap = assemble(ap) 
                bp = assemble(Lp)

                # Compute solution 
                [bc.apply(Ap, bp) for bc in self.bcp]
                [bc.apply(p1.vector()) for bc in self.bcp]
                solve(Ap, p1.vector(), bp, "bicgstab", self.prec)

                k += 1
            
            if t > filewrite_time and save_pvd:   
                # print(f'Time t = {float(repr(t)):.2f}')
                # Save solution to file
                file_u << u1
                file_p << p1
                filewrite_time += self.T/filewrite_freq

            # Update time step
            u0.assign(u1)
            t += dt

            fitness_new = get_velocity_sum(u1, self.turbine_centers, self.turbine_width, self.turbine_height)
            if abs(fitness_new - fitness)/fitness < update_tol:
                last_update += 1
            else:
                last_update = 0
                fitness = fitness_new
            # print(f'Time t = {float(repr(t)):.2f}, fitness = {fitness_new:.4f}, last_update = {last_update}')
            self.fitnesses.append(fitness_new)
        self.elapsed_time = perf_counter() - tic
        # fig, ax = plt.subplots()
        # ax.plot(np.linspace(0,t,len(self.fitnesses),endpoint=True),self.fitnesses)
        # ax.set_title("Fitness")
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Fitness')
        # # plt.axis('off')
        # fig.tight_layout()
        # fig.savefig(f'fitness.png',dpi=300, bbox_inches = 'tight')
        # plt.close()

        # x = self.turbine_centers[0,0]
        # y = self.turbine_centers[0,1]
        # print("Turbine center - turbine_height, u=", np.sqrt(u1(x,y-self.turbine_height)[0]**2 + u1(x,y-self.turbine_height)[1]**2))
        # print("Turbine center - turbine_width, u=", np.sqrt(u1(x-self.turbine_width,y)[0]**2 + u1(x-self.turbine_width,y)[1]**2))
        # print("Turbine center - turbine_width/2, u=", np.sqrt(u1(x-self.turbine_width/2,y)[0]**2 + u1(x-self.turbine_width/2,y)[1]**2))
        # print("Turbine center, u=",np.sqrt(u1(x,y)[0]**2 + u1(x,y)[1]**2))
        # print("Turbine center + turbine_width/2, u=",np.sqrt(u1(x+self.turbine_width/2,y)[0]**2 + u1(x+self.turbine_width/2,y)[1]**2))
        # print("Turbine center + turbine_width, u=",np.sqrt(u1(x+self.turbine_width,y)[0]**2 + u1(x+self.turbine_width,y)[1]**2))
        # print("Turbine center + turbine_height, u=",np.sqrt(u1(x+self.turbine_height,y)[0]**2 + u1(x+self.turbine_height,y)[1]**2))
        # print("Turbine center + turbine_height*2, u=",np.sqrt(u1(x+self.turbine_height*2,y)[0]**2 + u1(x+self.turbine_height*2,y)[1]**2))
        # print("Turbine center_y + turbine_height/2, u=",np.sqrt(u1(x,y+self.turbine_height/2+0.01)[0]**2 + u1(x,y+self.turbine_height/2+0.01)[1]**2))
        # print("Turbine center_y - turbine_height/2, u=",np.sqrt(u1(x,y-self.turbine_height/2-0.01)[0]**2 + u1(x,y-self.turbine_height/2-0.01)[1]**2))
        # self.plot_results(u1, p1, tag='test', path='history_10/')
        # print(f'Simulation end time T = {float(repr(t)):.2f}')
        self.T_end = float(repr(t))
        return u1, p1

    def set_turbine_centers(self, turbine_centers):
        self.turbine_centers = turbine_centers

    def plot_results(self, u, p, tag=None, path=None):
        # Plot solution
        plt.figure()
        plot(u, title="Velocity, Generation: " + str(tag))
        # plt.axis('off')
        plt.tight_layout()
        plt.savefig(path + f'u_{tag}.png',dpi=300, bbox_inches = 'tight')
        plt.close()
        plt.figure()
        plot(p) #, title="Pressure, Generation: " + str(tag))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path + f'p_{tag}.png',dpi=300,transparent=True, bbox_inches = 'tight')
        plt.close()
    
    def set_up_mesh_with_bc(self):
        self.mesh = create_mesh(self.L,self.H,self.resolution,self.turbine_centers,self.turbine_width,self.turbine_height)
        self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        self.boundaries.set_all(0)
        self.left = Left()
        self.right = Right()
        self.lower = Lower()
        self.upper = Upper()
        self.right.set_length(self.L)
        self.upper.set_height(self.H)

        self.left.mark(self.boundaries, 1)
        self.right.mark(self.boundaries, 2)
        self.lower.mark(self.boundaries, 3)
        self.upper.mark(self.boundaries, 4)

        # Generate finite element spaces
        self.V, self.Q, self.K = create_vector_spaces(self.mesh)

        # Define trial and test functions
        self.u, self.p, self.v, self.q = create_trial_test_functions(self.V,self.Q)

        # Define boundary conditions
        self.bcu, self.bcp = create_boundary_conditions(self.H,self.L,self.V,self.Q)
