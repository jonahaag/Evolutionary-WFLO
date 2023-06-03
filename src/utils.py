import numpy as np
import dolfin
from dolfin import *; from mshr import *
import dolfin.common.plotting as fenicsplot
import matplotlib.pyplot as plt

def create_mesh(L,H,resolution,centers,width,height):
    mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L, H)), resolution)

    plt.figure()
    plot(mesh)
    plt.savefig('mesh_refined_'+str(0)+'.png',dpi=300)
    plt.close()
    i = 0
    while mesh.hmin() > width: # 2*width
        print('Refine mesh...')
        cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
        for cell in cells(mesh):
            cell_marker[cell] = False
            p = cell.midpoint()
            for center in centers:
                if p.distance(Point(center[0],center[1])) < height:
                    cell_marker[cell] = True
        mesh = refine(mesh, cell_marker)
        i += 1
        plt.figure()
        plot(mesh)
        plt.savefig('mesh_refined_'+str(i)+'.png',dpi=300)
        plt.close()
    # print(f'hmin = {mesh.hmin():.4f}, dt = {0.5*mesh.hmin():.4f}')

    return mesh

def create_vector_spaces(mesh):
    # Generate finite element spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    K = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    return V, Q, K

def create_trial_test_functions(V,Q):
    # Define trial and test functions 
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)
    return u, p, v, q

class DirichletBoundaryLower(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

class DirichletBoundaryUpper(SubDomain):
    def set_height(self, H):
        self.H = H

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], self.H)

class DirichletBoundaryLeft(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0) 

class DirichletBoundaryRight(SubDomain):
    def set_length(self, L):
        self.L = L

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], self.L)

def create_boundary_conditions(H,L,V,Q):
    dbc_lower = DirichletBoundaryLower()
    dbc_upper = DirichletBoundaryUpper()
    dbc_left = DirichletBoundaryLeft()
    dbc_right = DirichletBoundaryRight()

    dbc_upper.set_height(H)
    dbc_right.set_length(L)

    uin = 1.0
    # Inflow on the left
    bcu_in0 = DirichletBC(V.sub(0), uin, dbc_left)
    bcu_in1 = DirichletBC(V.sub(1), 0.0, dbc_left)
    # Slip BC on top and bottom
    bcu_upp1 = DirichletBC(V.sub(1), 0.0, dbc_upper)
    bcu_low1 = DirichletBC(V.sub(1), 0.0, dbc_lower)
    bcu = [bcu_in0, bcu_in1, bcu_upp1, bcu_low1]

    pout = 0.0
    bcp1 = DirichletBC(Q, pout, dbc_right)
    bcp = [bcp1]

    return bcu, bcp

class WindFarm(UserExpression):
    def set_turbine_centers(self, centers):
        self.centers = centers

    def set_turbine_width(self, width):
        self.width = width
    
    def set_turbine_height(self, height):
        self.height = height
    
    def set_turbine_porosity(self, Kinv):
        self.Kinv = Kinv

    def eval(self, value, x):
        value[0] = 0.
        for center in self.centers:
          if x[0] >= center[0]-self.width/2 and x[0] <= center[0]+self.width/2 and x[1] >= center[1]-self.height/2 and x[1] <= center[1]+self.height/2:
            # TODO Optinal: refine the porosity function
            #value[0] = -self.Kinv * ((x[1] - center[1])/(self.height/2))**2 + self.Kinv
            value[0] = self.Kinv

# Define subdomains
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) 

class Right(SubDomain):
    def set_length(self, L):
        self.L = L

    def inside(self, x, on_boundary):
        return near(x[0], self.L)

class Lower(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Upper(SubDomain):
    def set_height(self, H):
        self.H = H

    def inside(self, x, on_boundary):
        return near(x[1],self.H)

def create_variational_problem(mesh, u, p, v, q, Kinv11, Kinv12, Kinv21, Kinv22, nu, nueff, u0, u1, p1, dt):
    # Define variational problem
    h = CellDiameter(mesh)
    u_mag = sqrt(dot(u1,u1))
    d1 = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag/h,2.0)))
    d2 = h*u_mag

    um = 0.5*(u + u0)
    um1 = 0.5*(u1 + u0)
    Fu = inner((u - u0)/dt + grad(um)*um1, v)*dx - p1*div(v)*dx + nueff*inner(grad(um), grad(v))*dx \
        + d1*inner((u - u0)/dt + grad(um)*um1 + grad(p1), grad(v)*um1)*dx + d2*div(um)*div(v)*dx \
        + nu*(Kinv11*inner(um[0],v[0])*dx + Kinv12*inner(um[0],v[1])*dx + Kinv21*inner(um[1],v[0])*dx + Kinv22*inner(um[1],v[1])*dx)
    au = lhs(Fu)
    Lu = rhs(Fu)

    Fp = d1*inner((u1 - u0)/dt + grad(um1)*um1 + grad(p), grad(q))*dx + div(um1)*q*dx 
    ap = lhs(Fp)
    Lp = rhs(Fp)
    return au, Lu, ap, Lp

def get_velocity_sum(u, turbine_centers, turbine_width, turbine_height):
    velocity_sum = 0.
    for center in turbine_centers:
        x = center[0] - turbine_width/2
        ny = 100
        dy = turbine_height/ny
        for y in np.linspace(center[1]-turbine_height/2, center[1]+turbine_height/2, ny, endpoint=True):
            velocity_sum += u(x,y)[0] * dy
    return velocity_sum

def check_layout_validity(turbine_centers, min_x_spacing, min_y_spacing, grid_width, grid_height):
    n_turbines = turbine_centers.shape[0]
    assert turbine_centers.shape == (n_turbines, 2)
    for i in range(n_turbines):
        for j in range(i):
            turbine_centers_xdist = np.abs(turbine_centers[i,0] - turbine_centers[j,0])
            turbine_centers_ydist = np.abs(turbine_centers[i,1] - turbine_centers[j,1])
            if turbine_centers_ydist < 1e-6 and turbine_centers_xdist < min_x_spacing:
                return False
            if turbine_centers_xdist < 1e-6 and turbine_centers_ydist < min_y_spacing:
                return False
    return True

def check_single_turbine_validity(new_turbine_center, turbine_centers, min_x_spacing, min_y_spacing):
    turbine_centers_xdist = np.abs(turbine_centers[:,0] - new_turbine_center[0])
    turbine_centers_ydist = np.abs(turbine_centers[:,1] - new_turbine_center[1])
    if np.any((turbine_centers_ydist < 1e-6).astype(int) + (turbine_centers_xdist < min_x_spacing).astype(int) == 2):
            return False
    if np.any((turbine_centers_xdist < 1e-6).astype(int) + (turbine_centers_ydist < min_y_spacing).astype(int) == 2):
            return False
    return True

def evaluate_layout(windFarmSimulator, individual, min_x_spacing, min_y_spacing, grid_width, grid_height):
    valid_layout = check_layout_validity(individual.turbine_centers, min_x_spacing, min_y_spacing, grid_width, grid_height)
    # print(valid_layout)
    if valid_layout:
        windFarmSimulator.set_turbine_centers(individual.turbine_centers)
        windFarmSimulator.set_up_mesh_with_bc()
        u, p = windFarmSimulator.simulate()
        fitness = get_velocity_sum(u, individual.turbine_centers, windFarmSimulator.turbine_width, windFarmSimulator.turbine_height)
        return fitness, u, p
    else:
        return 0., None, None