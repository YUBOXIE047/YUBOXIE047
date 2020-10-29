import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


v = 5
L = 2.3
u = 2 * np.pi/180


def system_dynamics(_t, z):
    theta = z[2]
    return [v * np.cos(theta),
            v * np.sin(theta),
            v * np.tan(u) / L]


num_points = 100
t_final = 2
z_initial_condition = [0, 30, 5 * np.pi/180]
solution = solve_ivp(system_dynamics,
                     [0, t_final],
                     z_initial_condition,
                     t_eval=np.linspace(0, t_final, num_points))


x_solution = solution.y[0]
y_solution = solution.y[1]
t_points = solution.t

plt.plot( x_solution, y_solution)
plt.show()
