import numpy as np
import csv
import os
from scipy import signal

#do not change constants parameter
Mb = 500        #mass quarter body [kg]
Mt = 50        	#mass tire + suspention system [kg]
Sb = 35000      #spring constant tire to body[N/m]
St = 280000     #spring constant tire to road [N/m]
Dt = 10.02      #damper tire constant [Ns/m]
Dbound = 56000.0 #boundary damping constant [Ns/m] Verhältnis Dbound~100*Da
Da = 900        #damper constant active damping [Ns/m]
c = 560         #linear constant of active suspension [N/A]
#do not change constants parameter

def activeSuspension(Zb, Zt, Zb_dt, Zt_dt, Zh, Zh_dt, i, dt):
    '''
    --- Quarter Car Suspension Model vgl. hackathon task ---
    Zb: z-position body [m]
    Zt: z-position tire [m]
    Zb_dt: velocity body in z [m/s]
    Zt_dt: velocity tire in z [m/s]
    Zh: road profie [m]
    Zh_dt: velocity road profile in z [m/s]

    Tuning Parameter
    i: current of active suspension from 0 to 2 [A]
    '''
    F_push = Da*(Zb_dt-Zt_dt) - c*i
    F_bound = Dbound*(Zb_dt-Zt_dt)
    F_pull = Da*(Zb_dt-Zt_dt) + c*i

    F_D = np.max([F_push,np.min([F_bound,F_pull])])

    updated_Zb_dtdt = (-Sb*(Zb-Zt))/Mb - F_D/Mb
    updated_Zb_dt = Zb_dt + updated_Zb_dtdt*dt
    updated_Zb = Zb + updated_Zb_dt*dt

    updated_Zt_dtdt = (-St*(Zt-Zh))/Mt + (Sb*(Zb-Zt))/Mt + (-Dt*(Zt_dt-Zh_dt))/Mt + F_D/Mt
    updated_Zt_dt = Zt_dt + updated_Zt_dtdt*dt
    updated_Zt = Zt + updated_Zt_dt*dt

    return [updated_Zb, updated_Zt, updated_Zb_dt, updated_Zt_dt, updated_Zb_dtdt, updated_Zt_dtdt ]


class CurrentError():

    def __init__(self, file_name, vel, dir = '/net/projects/scratch/winter/valid_until_31_July_2020/hackathon/datasets/'):
        self.roadProfile = file_name
        self.roadProfileLocation = dir
        self.vel = vel

        self.dt = 0.005


        #road specific weight factor
        self.K = float(self.roadProfile.split("_k_")[1][:3])

        self.intialSuspensionState = np.zeros(6) #inital suspension states at t = 0


        self.timeRecording = []
        self.tripRecording = []
        self.profile = []

        with open(self.roadProfileLocation+self.roadProfile) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                self.timeRecording.append(float(row[0]))
                self.tripRecording.append(float(row[1]))
                self.profile.append(float(row[2]))

        #get simulation time by constant speed
        self.T = float(self.tripRecording[-1])/float(self.vel)

        self.N = int(np.round(self.T/self.dt))
        self.t = np.linspace(0, self.T, self.N+1)

        #get driving speed vector e.g for dynamic (non constant) speed
        self.v = np.ones(self.t.size)*self.vel

        #get trip at each dt
        self.trip = []
        for i in range(0,self.t.size):
            self.trip.append(np.trapz(self.v[0:i+1], dx=self.dt))

        #get the road profile by the tripRecording
        self.profile = np.interp(self.trip, self.tripRecording, self.profile)

        self.C_set = False

    def set_current(self, C=None, const=False, const_a=1.0):

        if const:
            self.C = np.ones(self.t.size)*const_a
        elif True:
            self.C = C
        else:
            raise Exception("GIVE ME A C")
        self.C_set = True


    def butter_bandpass(self, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a


    def calculate_average_error(self):
        if not self.C_set:
            raise Exception("Need C before calculating error.")

        N = self.t.size - 1             # No of time intervals
        u = np.zeros((N+1,6))      # array of [Zb, Zt, Zb_dt, Zt_dt, Zb_dtdt, Zt_dtdt] at each time interval
        u[0] = self.intialSuspensionState # [Zb, Zt, Zb_dt, Zt_dt] at t=0
        for n in range(0,N):
            dt = float(self.t[1] - self.t[0]) # dt
            Zh_dt = float(0)
            if n>0:
                dt = self.t[n] - self.t[n-1]
                Zh_dt = (self.profile[n]-self.profile[n-1])/dt
            #******
            # if no vector C with current over from 0 to 2 [A] time interval is provided
            # one may calculate the next i here based on the array u[0:n], H[0:n] and Zh_dt
            #******
            u[n+1]=activeSuspension(u[n][0], u[n][1], u[n][2], u[n][3], self.profile[n], Zh_dt, self.C[n], dt)


            Zb= u[:,0]
            Zt= u[:,1]
            Zb_dtdt= u[:,4]
            Zt_dtdt= u[:,5]


        #compute bandpass 2nd order from 0.4 - 3 Hz
        b, a = self.butter_bandpass(0.4, 3, int(1/dt), 2)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

        #calculate variance alpha_1
        varZb_dtdt=np.var(z)


        #compute bandpass 2nd order from 0.4 - 3 Hz
        b, a = self.butter_bandpass(10, 30, int(1/dt), 1)
        zi = signal.lfilter_zi(b, a)
        z1, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)
        #calculate variance alpha_2
        varZb_dtdt_h=np.var(z1)

        #compute T_target
        target = self.K*varZb_dtdt_h + varZb_dtdt

        #check boudning condition

        #do not change constants parameter
        Mb = 500        #mass quarter body [kg]
        Mt = 50        	#mass tire + suspention system [kg]
        #do not change constants parameter

        #standard deviation of Zt_dtdt
        devZt_dtdt = np.std(Zt_dtdt*Mt)

        #boundary condition
        F_stat_bound=(Mb+Mt)*9.81/3.0
        if devZt_dtdt > (F_stat_bound):
            bc = 'failed'
        else:
            bc = 'passed'

        return bc, target

if __name__=="__main__":
    CE = CurrentError(file_name='ts3_1_k_3.0.csv', vel=27.0)
    CE.set_current(const=True)
    e,b = CE.calculate_average_error()

    print(e,b)
