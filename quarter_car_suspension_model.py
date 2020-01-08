import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from scipy.integrate import odeint

def surf(t): 
    v = 80
    if v*t > 450 and v*t < 600:
        return 1.5
    elif v*t > 750 and v*t < 1000:
        return 1
    elif v*t > 1200 and v*t < 1450:
        return 3.5
    else:
        return 3 #4

def dot_surf(t): #Approximate an impulse
    v = 80
    if v*t > 445  and v*t < 605:
        return 1.8# 4
    elif v*t > 745  and v*t < 1005:
        return 1.5# 4
    elif v*t > 1195  and v*t < 1455:
        return 4
    else:
        return 0


class QuarterSuspensionModel():

    def __init__(self, g = 9.81,  m1 = 400, m2 = 30, b1 = 2000, b2 = 10000, 
            k1 = 50000, k2 = 80000):
        """
        Args:
            g       gravitational constant [m/s^2]
            m1      mass of sprung mass (car) [kg]
            m2      mass of unsprung mass (tire) [kg] 
            k1      stiffness car - tire [N.s/m]
            k2      stiffness tire - ground [N.s/m]
            b1      damping car - tire [(kg s^2)/m]
            b2      damping tire - ground [(kg s^2)/m]
        """

        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.b1 = b1
        self.b2 = b2

    def dynamic_model(self, states, t, m1, m2, k1, k2, b1, b2):
        x1 = states[0] # vertical position of sprung mass
        v1 = states[1] # vertical velocity of sprung mass
        x2 = states[2] # vertical position of unsprung mass
        v2 = states[3] # vertical velocity of unsprung mass
        
        dx1dt = v1
        dv1dt = (- k1 * (x1 - x2) - b1 * (v1 - v2)) / m1  
        dx2dt = v2
        dv2dt = (k1 * (x1 - x2) + b1 * (v1 - v2) - k2 * (x2 - surf(t)) - b2 * (v2 - dot_surf(t)))/m1

        return [dx1dt, dv1dt, dx2dt, dv2dt]


    def run_simulation(self, nr_seconds = 20, file_name="quarter_car_suspension_simulation", tire_width = 2.0, tire_height = 1.0, car_width = 3.0, car_height = 1.5):
        """
        t           time of simulation in seconds
        """
        # states with the following states:
        #   states[0] displacement of sprung mass
        #   states[1] velocity of sprung mass
        #   states[2] displacement of unsprung mass
        #   states[3] velocity of unsprung mass
        states = [0, 0, 0, 0]

        # generate time steps in milliseconds
        t = np.linspace(0, nr_seconds, 250)

        # TODO 
        surface = [surf(ts) for ts in t] 

        # solve differential equaiton / run simulation
        sol = odeint(self.dynamic_model, states, t, args=(self.m1, self.m2, self.k1, self.k2, self.b1, self.b2))


        # static plot
        plt.figure(figsize=(16,10))
        plt.plot(t, sol[:,0], label = "car displacement")
        plt.plot(t, sol[:,2], label = "tire displacement")
        plt.plot(t, surface, label = "road")
        plt.legend()


        # animation
        fig = plt.figure(figsize=(16, 10), facecolor='w')
        ax = fig.add_subplot(1, 1, 1)
        plt.rcParams['font.size'] = 15
        ax.plot(t, surface, color='k', lw=2)

        lns = []
        for i, ts in enumerate(t):
            tm = ax.text(ts, 0, 'time = {:06.3f}ms'.format(ts))
            point = ax.add_artist(plt.Circle((ts, surface[i]), 0.2, color='black'))
            tire = ax.add_patch(patches.Rectangle((ts - tire_width/2, sol[i,2] + 2 - tire_height/2), 
                width = tire_width, height = tire_height, linewidth = 2, edgecolor = 'black', facecolor = 'blue'))
            car = ax.add_patch(patches.Rectangle((ts - car_width/2, sol[i,0] + 2 + tire_height + 2 - car_height/2), 
                width = car_width, height = car_height, linewidth = 2, edgecolor = 'black', facecolor = 'blue'))
            lns.append([tm, point, tire, car])
        ax.set_ylim([0, 10])
        ax.set_xlim([-1, 1+ts])
        ax.set_aspect('equal', 'datalim')
        ax.grid()
        ani = animation.ArtistAnimation(fig, lns, interval=1)
        #ani.save(file_name + '.gif', writer='imagemagick', fps=1000/50)

        plt.show()
if __name__=="__main__":
    qsm = QuarterSuspensionModel()
    qsm.run_simulation()

