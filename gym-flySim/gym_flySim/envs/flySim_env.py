import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
DEG2RAD = np.pi/180

class flySimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.wing = {
            # wingangles
            'psi' : np.array([90, 53, 90, -53]) * DEG2RAD, #[psi0_L psim_L psi0_R psim_R].psi0_R = 90, psim_R = -psim_L;
            'theta' : np.array([0, 0, 0, 0]) * DEG2RAD, # [theta0_L thetam_L theta0_R thetam_R].theta0_R = theta0_L, thetam_R =Wing.thetam_L[rad]
            'phi' : np.array([90, 65, -90, -65]) * DEG2RAD, # [phi0_L phim_L phi0_R phim_R].phi0_R = -phi0_L, phim_R = -phim_L[rad]
            # wing angles phase
            'delta_psi' : -90 * DEG2RAD,# [rad]
            'delta_theta' : 90 * DEG2RAD, # [rad]
            # wave wakk (Roni)
            'C' : 2.4,
            'K' : 0.7,
            # wing locations
            'hingeR' : np.array([0.0001, 0, 0.0001]), # right wing hinge location in body axes[m]
            'hingeL' : np.array([0.0001, 0, 0.0001]), # left wing hinge location in body axes[m]
            'ACloc' : np.array([2.5 / 1000 * 0.7, 0, 0]), # used to calculate the torque around body(wing ax) [m]
            'freq' : 220 * pi * 2, # wing frequency[rad / s]
            'span' : 2.5 / 1000, # span[m]
            'cord' : 0.7 / 1000, # cord[m]
            'speedCalc' : np.array([2.5 / 1000, 0, 0]) # location on wing used to calculate the wing's tangential velocity
        }
        self.wing['T'] = 1 / (self.wing.freq / 2 / pi)
        self.body = {
            'BodIniVel' : np.array([0, 0, 0]), # body initial velocity(body ax) [m / s]
            'BodIniang' : np.array([0, -45, 0]) * DEG2RAD, # body initial angle(lab ax) [rad]
            'BodInipqr' : np.array([0, 0, 0]), # body initial angular velocity[m / s]
            'BodIniXYZ' : np.array([0, 0, 0]), # body initial position(lab ax) [m]
        }
        self.gen = {
            'm' : 1e-6, # mass[kg]
            'g' : np.array([0, 0, -9.8]), # gravity[m / s ^ 2]
            'strkplnAng' : np.array([0, 45, 0]) * pi / 180, # stroke plane(compare to body) [rad]
            'rho' : 1.225, # air density
            'TauExt' : np.array([0, 0, 0]) * 1e-7, # external torque body axes[Nm]
            'time4Tau' : np.array([10,15]), # initial and final time of the external torque[ms]
            'I' : 1.0e-12 * np.array([[0.1440, 0, 0],[0,0.5220,0],[0,0,0.5220]]), # tensor of inertia[Kg m ^ 2]
        }
        self.aero = {
            'CLmax' : 1.8,
            'CDmax' : 3.4,
            'CD0' : 0.4,
            'r22' : 0.4,
        }

    def step(self,action):

        return state,reward,done,info

    def reset(self):
        # self.X0 = np
        #     self.body['BodIniVel'].T,self.body['BodInipqr'].T,self.body['BodIniang'].T]
        # obj.U0 = [obj.wing.psi(1:2)';obj.wing.theta(1:2)';
        # obj.wing.phi(1: 2)';obj.wing.psi(3:4)';
        # obj.wing.theta(3: 4)';obj.wing.phi(3:4)'];

        return state

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        return None
