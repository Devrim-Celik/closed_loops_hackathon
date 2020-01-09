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

class QuarterSuspensionModel():

    def __init__(self, g = 9.81,  m1 = 1500, m2 = 50, b1 = 3000, b2 = 10000, 
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

        self.image_folder = './img'
        self.file_name = 'quarter_car_suspension_simulation'


    def dynamic_model(self, states, t, m1, m2, k1, k2, b1, b2):
        """
        Function with ODEs, used by odeint.

        Args
            states      4-tuple of all variables
                            x[0] vertical pos. of sprung mass
                            x[1] vertical vel. of sprung mass
                            x[2] vertical pos. of unsprung mass
                            x[3] vertical vel. of unsprung mass
            t           time
            m1          mass of sprung
            m2          mass of unsprung
            k1          spring constant 1
            k2          spring constant 2
            b1          damping constant 1
            b2          damping constant 2
        """
        x1 = states[0] 
        v1 = states[1] 
        x2 = states[2] 
        v2 = states[3] 
        
        # change of vertical position of sprung is its vertical velocity
        dx1dt = v1
        # force of spring and damper
        dv1dt = (- k1 * (x1 - x2) - b1 * (v1 - v2)) / m1  
        # change of vertical position of unsprung is its vertical velocity
        dx2dt = v2
        # force of spring and damper between sprung and unsprung and force of spring and damper of unsprung to ground
        dv2dt = (k1 * (x1 - x2) + b1 * (v1 - v2) - k2 * (x2 - surf(t)) - b2 * (v2 - (surf(t)-surf(t-1))))/m1

        return [dx1dt, dv1dt, dx2dt, dv2dt]


    def run_simulation(self, nr_seconds = 20):
        """
        Args
            t           time of simulation in seconds
            nr_seconds  amount of time to run simulation in seconds

        Returns
            sol         solution of ODE, row is time and col is states
        """
        # states with the following states:
        #   states[0] displacement of sprung mass
        #   states[1] velocity of sprung mass
        #   states[2] displacement of unsprung mass
        #   states[3] velocity of unsprung mass
        states = [0, 0, 0, 0]

        # generate time steps in milliseconds
        self.t = np.linspace(0, nr_seconds, 250)

        # solve differential equaiton / run simulation
        self.sol = odeint(self.dynamic_model, states, self.t, args=(self.m1, self.m2, self.k1, self.k2, self.b1, self.b2))

        return self.sol


    def static_plot(self):
        """
        Constructs a static plot of the displacement of sprung
        and unsprung mass combined with dispalying the profile of the road.
        """
        # TODO 
        surface = [surf(ts) for ts in self.t] 
        
        plt.figure(figsize=(16,10))
        plt.plot(self.t, self.sol[:,0], label = "sprung displacement")
        plt.plot(self.t, self.sol[:,2], label = "unsprung displacement")
        plt.plot(self.t, surface, label = "ground")
        plt.legend()

        plt.savefig(self.image_folder + '/' + self.file_name + '.png')

        plt.show()

    def animation(self, unsprung_width = 2.0, unsprung_height = 1.0, sprung_width = 3.0, sprung_height = 1.5):
        """
        Creates animation of simulation.

        Args
            unsprung_width      width of unsprung in animation
            unsprung_height     height of unsprung in animation
            sprung_width        width of prung in animation
            sprung_height       height of sprung in animation
        """

        # TODO 
        surface = [surf(ts) for ts in self.t] 
        
        fig = plt.figure(figsize=(16, 10), facecolor='w')
        ax = fig.add_subplot(1, 1, 1)
        plt.rcParams['font.size'] = 15
        ax.plot(self.t, surface, color='k', lw=2)

        draw_objects = []
        for i, ts in enumerate(self.t):
            #TODO add springs and dampeners
            # text of time
            tm = ax.text(ts, 0, 'time = {:06.3f}ms'.format(ts))
            # blak point touching the grond
            point = ax.add_artist(plt.Circle((ts, surface[i]), 0.2, color='black'))
            # unsprung
            unsprung = ax.add_patch(patches.Rectangle((ts - unsprung_width/2, self.sol[i,2] + 2 - unsprung_height/2), 
                width = unsprung_width, height = unsprung_height, linewidth = 2, edgecolor = 'black', facecolor = 'blue'))
            unsprung_txt = ax.text(ts, self.sol[i,2] + 2, "$m2$")
            # sprung
            sprung = ax.add_patch(patches.Rectangle((ts - sprung_width/2, self.sol[i,0] + 2 + unsprung_height + 2 - sprung_height/2), 
                width = sprung_width, height = sprung_height, linewidth = 2, edgecolor = 'black', facecolor = 'blue'))
            sprung_txt = ax.text(ts, self.sol[i,0] + 2 + unsprung_height + 2,"$m1$")

            draw_objects.append([tm, point, unsprung, sprung, unsprung_txt, sprung_txt])

        ax.set_ylim([0, 10])
        ax.set_xlim([-1, 1+ts])
        ax.set_aspect('equal', 'datalim')
        ax.grid()

        ani = animation.ArtistAnimation(fig, draw_objects, interval=1)

        #saving animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=1, metadata=dict(artist='Devrim Celik'), bitrate=1800)

        ani.save(self.image_folder + '/' + self.file_name + '.mp4', writer='writer')

        plt.show()
if __name__=="__main__":
    qsm = QuarterSuspensionModel()
    qsm.run_simulation()
    qsm.static_plot()
    qsm.animation()

