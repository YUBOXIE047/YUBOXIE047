import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Car:

    def __init__(self,
                 length=2.3,
                 velocity=5,
                 x_pos_init=0, y_pos_init=0, pose_init=2 * np.pi / 180):
        self.__length = length
        self.__velocity = velocity
        self.__x = x_pos_init
        self.__y = y_pos_init
        self.__pose = pose_init

    def move(self, steering_angle, dt):
        def bicycle_model(_t, z):
            x = z[0]
            y = z[1]
            theta = z[2]
            return [self.__velocity * np.cos(theta),
                    self.__velocity * np.sin(theta),
                    self.__velocity * np.tan(steering_angle)
                    / self.__length]

        sol = solve_ivp(bicycle_model,
                        [0, dt],
                        [self.__x, self.__y, self.__pose])

        self.__x = sol.y[0, -1]
        self.__y = sol.y[1, -1]
        self.__pose = sol.y[2, -1]

    def y(self):
        return self.__y


class PidController:
    def __init__(self, kp, ki, kd, ts):

        self.__kp = kp
        self.__kd = kd / ts  # discrete-time Kd
        self.__ki = ki * ts
        self.__previous_error = None
        self.__error_sum = 0.

    def control(self, y, set_point=0.):

        error = set_point - y
        steering_action = self.__kp * error

        if self.__previous_error is not None:
            error_diff = error - self.__previous_error
            steering_action += self.__kd * error_diff

        if self.__previous_error is not None:
            self.__error_sum += error
        steering_action += self.__ki * self.__error_sum

        self.__previous_error = error
        return steering_action


t_sampling = 0.025
car = Car(y_pos_init=0.5, velocity=5)
pid = PidController(kp=1, kd=0.145, ki=0.1, ts=t_sampling)
num_points = 1999
u_disturbance = np.pi / 180
y_cache = np.array([car.y()], dtype=float)
for i in range(num_points):
    u = pid.control(car.y())
    car.move(u + u_disturbance, t_sampling)
    y_cache = np.append(y_cache, car.y())

t_span = t_sampling * np.arange(num_points + 1)
plt.plot(t_span, y_cache)
plt.xlabel('Time(t)')
plt.ylabel('Lateral position, y(m)')
plt.grid()
plt.show()
