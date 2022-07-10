# GYM imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import gym_flySim

# Numpy/Scipy imports
import numpy as np
from scipy.integrate import solve_ivp
from numpy import sin, cos, tanh, arcsin
from scipy.spatial.transform import Rotation
from numpy.linalg import norm, inv

# Other Imports
import json
import os
import sys

# Constants
DEG2RAD = np.pi / 180

# TODO: integrate this into the configuration file
CONVERSIONS_FACTORS = {
    "gen": {
        "strkplnAng": DEG2RAD,
        "I": 1.0e-12
    },
    "random": {
        "pqr": DEG2RAD,
        "ang": DEG2RAD
    },
    "aero": {
    },
    "wing": {
        "psi": DEG2RAD,
        "theta": DEG2RAD,
        "phi": DEG2RAD,
        "delta_psi": DEG2RAD,
        "delta_theta": DEG2RAD,
    },
    "body": {
        "BodIniang": DEG2RAD,
        "BodInipqr": DEG2RAD,
        "BodRefPitch": DEG2RAD
    },
    "solver": {

    },
    "reward": {

    }

}


def body_ang_vel_pqr(angles, angles_dot, get_pqr):
    """
    Converts change in euler angles to body rates (if get_pqr is True) or body rates to euler rates (if get_pqr is False)
    :param angles: euler angles (np.array[psi,theta,phi])
    :param angles_dot: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
    :param get_pqr: whether to get body rates from euler rates or the other way around
    :return: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
    """
    psi = angles[0]
    theta = angles[1]
    psi_p_dot = angles_dot[0]
    theta_q_dot = angles_dot[1]
    phi_r_dot = angles_dot[2]

    if get_pqr:
        p = psi_p_dot - phi_r_dot * sin(theta)
        q = theta_q_dot * cos(psi) + phi_r_dot * sin(psi) * cos(theta)
        r = -theta_q_dot * sin(psi) + phi_r_dot * cos(psi) * cos(theta)
        return np.stack([p, q, r])
    else:
        psi_dot = psi_p_dot + theta_q_dot * (sin(psi) * sin(theta)) / cos(theta) + phi_r_dot * (
                cos(psi) * sin(theta)) / cos(theta)
        theta_dot = theta_q_dot * cos(psi) + phi_r_dot * -sin(psi)
        phi_dot = theta_q_dot * sin(psi) / cos(theta) + phi_r_dot * cos(psi) / cos(theta)
        return np.stack([psi_dot, theta_dot, phi_dot])


def wing_angles(psi, theta, phi, omega, delta_psi, delta_theta, c, k, t):
    """
    computes the wing angles given a set of variables, described in (see Whithead et al, "Pitch perfect: how fruit flies
     control their body pitch angle." 2015, appendix 1)
    :param psi: [psi0_L psim_L psi0_R psim_R].psi0_R = 90, psim_R = -psim_L;
    :param theta: [theta0_L thetam_L theta0_R thetam_R].theta0_R = theta0_L, thetam_R =Wing.thetam_L[rad]
    :param phi: [phi0_L phim_L phi0_R phim_R].phi0_R = -phi0_L, phim_R = -phim_L[rad]
    :param omega: wing angular velocity [rad / s]
    :param delta_psi: wing angles phase [rad]
    :param delta_theta: wing angles phase [rad]
    :param c:
    :param k:
    :param t: time in cycle
    :return:
    """
    psi_w = psi[0] + psi[1] * tanh(c * np.sin(omega * t + delta_psi)) / tanh(c)
    theta_w = theta[0] + theta[1] * np.cos(2 * omega * t + delta_theta)
    phi_w = phi[0] + phi[1] * arcsin(k * sin(omega * t)) / arcsin(k)

    psi_dot = -(c * omega * psi[1] * np.cos(delta_psi + omega * t) * (
            np.tanh(c * (sin(delta_psi + omega * t))) ** 2 - 1)) / tanh(c)
    theta_dot = -2 * omega * theta[1] * sin(delta_theta + 2 * omega * t)
    phi_dot = k * (omega * phi[1] * cos(omega * t)) / (
            arcsin(k) * (1 - k ** 2 * (sin(omega * t)) ** 2) ** (1 / 2))

    angles = np.array([psi_w, theta_w, phi_w])
    angles_dot = np.array([psi_dot, theta_dot, phi_dot])
    return angles, angles_dot


class AeroModel(object):
    def __init__(self, span, chord, rho, r22, clmax, cdmax, cd0, hinge_loc, ac_loc):
        self.s = span * chord * np.pi / 4
        self.rho = rho
        self.r22 = r22
        self.clmax = clmax
        self.cdmax = cdmax
        self.cd0 = cd0
        self.span_hat = np.array([1, 0, 0])
        self.hinge_location = hinge_loc
        self.ac_loc = ac_loc

    def get_forces(self, aoa, v_wing, rotation_mat_body2lab, rotation_mat_wing2lab, rotation_mat_sp2lab):
        cl = self.clmax * sin(2 * aoa)
        cd = (self.cdmax + self.cd0) / 2 - (self.cdmax - self.cd0) / 2 * cos(2 * aoa)
        u = v_wing[0] ** 2 + v_wing[1] ** 2 + v_wing[2] ** 2
        uhat = v_wing / norm(v_wing)
        span_hat = self.span_hat
        lhat = (np.cross(span_hat, -uhat)).T  # perpendicular to Uhat
        lhat = lhat / norm(lhat)
        q = self.rho * self.s * self.r22 * u
        drag = -0.5 * cd * q * uhat
        lift = 0.5 * cl * q * lhat
        rot_mat_spw2lab = rotation_mat_sp2lab @ rotation_mat_wing2lab
        ac_loc_lab = rot_mat_spw2lab @ self.ac_loc.T + rotation_mat_body2lab @ self.hinge_location.T  # AC location in lab axes
        ac_loc_body = rotation_mat_body2lab.T @ ac_loc_lab  # AC location in body axes

        f_lab_aero = rot_mat_spw2lab @ lift + rot_mat_spw2lab @ drag
        # force in body axes
        f_body = rotation_mat_body2lab.T @ f_lab_aero
        t_lab = np.cross(ac_loc_lab.T,
                         f_lab_aero).T  # + cross(ACLocB_body, Dbod).T # torque on body (in body axes)
        # (from forces, no CM0)
        t_body = np.cross(ac_loc_body.T,
                          f_body).T  # + cross(ACLocB_lab, Dbod_lab) # torque on body( in bodyaxes)
        # (from forces, no CM0)
        return cl, cd, span_hat, lhat, drag, lift, t_body, f_body, f_lab_aero, t_lab


class flySimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_path=None):
        super(flySimEnv, self).__init__()
        self.state = None
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,))
        self.action_space = spaces.Box(-0.1, +0.1, (1,), dtype=np.float32)
        self.tot_rwd = 0
        ## This is the old initialization before config.json, here for reference and will be deleted in future commit
        if False:
            self.wing = {
                # wing angles
                'psi': np.array([90, 53, 90, -53]) * DEG2RAD,
                # [psi0_L psim_L psi0_R psim_R].psi0_R = 90, psim_R = -psim_L;
                'theta': np.array([0, 0, 0, 0]) * DEG2RAD,
                # [theta0_L thetam_L theta0_R thetam_R].theta0_R = theta0_L, thetam_R =Wing.thetam_L[rad]
                'phi': np.array([90, 65, -90, -65]) * DEG2RAD,
                # [phi0_L phim_L phi0_R phim_R].phi0_R = -phi0_L, phim_R = -phim_L[rad]
                # wing angles phase
                'delta_psi': -90 * DEG2RAD,  # [rad]
                'delta_theta': 90 * DEG2RAD,  # [rad]
                # wave wakk (Roni)
                'C': 2.4,
                'K': 0.7,
                # wing locations
                'hingeR': np.array([0.0001, 0, 0.0001]),  # right wing hinge location in body axes[m]
                'hingeL': np.array([0.0001, 0, 0.0001]),  # left wing hinge location in body axes[m]
                'ACloc': np.array([2.5 / 1000 * 0.7, 0, 0]),  # used to calculate the torque around body(wing ax) [m]
                'omega': 220 * np.pi * 2,  # wing angular velocity[rad / s]
                'span': 2.5 / 1000,  # span[m]
                'chord': 0.7 / 1000,  # chord[m]
                'speedCalc': np.array([2.5 / 1000, 0, 0])
                # location on wing used to calculate the wing's tangential velocity
            }
            self.wing['T'] = 1.0 / (self.wing['omega'] / 2.0 / np.pi)  # seconds
            self.body = {
                'BodIniVel': np.array([0, 0, 0]),  # body initial velocity(body ax) [m / s]
                'BodIniang': np.array([0, -45, 0]) * DEG2RAD,  # body initial angle(lab ax) [rad]
                'BodInipqr': np.array([0, 0, 1000]) * DEG2RAD,  # body initial angular velocity[rad / s]
                'BodIniXYZ': np.array([0, 0, 0]),  # body initial position(lab ax) [m]
                'BodRefPitch': -45 * DEG2RAD
            }
            self.gen = {
                'm': 1e-6 * 0.88,  # mass[kg]
                'g': np.array([0, 0, -9.8]),  # gravity[m / s ^ 2]
                'strkplnAng': np.array([0, 45, 0]) * DEG2RAD,  # stroke plane(compare to body) [rad]
                'rho': 1.225,  # air density
                'TauExt': np.array([0, 0, 0]) * 1e-7,  # external torque body axes[Nm]
                'time4Tau': np.array([10, 15]),  # initial and final time of the external torque[ms]
                'I': 1.0e-12 * np.array([[0.1440, 0, 0], [0, 0.5220, 0], [0, 0, 0.5220]]),
                # tensor of inertia[Kg m ^ 2]
                'controlled': True,  # If true, calculate for each step, False for debugging against matlab version
                't': 0,
                'tsim_in': 0.0,  # simulation initial time[s]
                'tsim_fin': .5,
                'MaxStepSize': 1.0 / 16000

            }
            self.aero = {
                'CLmax': 1.8,
                'CDmax': 3.4,
                'CD0': 0.4,
                'r22': 0.4,
            }

            self.random = {
                'randomize': False,
                'seed': 1234,
                'vel': np.array([0.2, 0.2, 0.2]),
                'pqr': np.array([0, 2400.0, 0]) * DEG2RAD,
                'ang': np.array([0, 60.0, 0]) * DEG2RAD
            }

            self.aero_model_R = AeroModel(self.wing['span'], self.wing['chord'], self.gen['rho'], self.aero['r22'],
                                          self.aero['CLmax'], self.aero['CDmax'], self.aero['CD0'], self.wing['hingeR'],
                                          self.wing['ACloc'])
            self.aero_model_L = AeroModel(self.wing['span'], self.wing['chord'], self.gen['rho'], self.aero['r22'],
                                          self.aero['CLmax'], self.aero['CDmax'], self.aero['CD0'], self.wing['hingeL'],
                                          self.wing['ACloc'])

            if self.random['randomize']:
                self._seed = self.random['seed']
                self._rng = np.random.default_rng(self._seed)
            self.random_bound = {'vel': np.array([0.2, 0.2, 0.2]),
                                 'pqr': np.array([0, 2400.0, 0]) * DEG2RAD,
                                 'ang': np.array([0, 60.0, 0]) * DEG2RAD}

        if config_path is not None:
            self.load_config(config_path)
            print(f'config loaded from:{config_path}')
        else:
            function_directory = os.path.dirname(__file__)
            self.load_config(os.path.join(function_directory, 'config.json'))
            print(f'running with environment default configuration')

        if hasattr(self.random, 'seed'):
            self._rng = np.random.default_rng(self.random['seed'])
        else:
            self._rng = np.random.default_rng()

    def pitch_angle_over_88(self, t, y, *args):
        abs_minus_88 = np.abs(y[7] / DEG2RAD) - 88
        return abs_minus_88

    def load_config(self, config_path):

        ### Load the configuration file
        assert os.path.isfile(config_path), "invalid config path"
        with open(config_path, 'r') as f:
            config = json.load(f)

        ### change all the lists to numpy arrays,
        for pk in config.keys():
            for k in config[pk].keys():
                if type(config[pk][k]) is list:
                    config[pk][k] = np.array(config[pk][k])
                if config['gen']['Convert_values'] and k in CONVERSIONS_FACTORS[pk]:
                    config[pk][k] = config[pk][k] * CONVERSIONS_FACTORS[pk][k]

        ### assign values to self
        self.wing = config['wing']
        self.aero = config['aero']
        self.body = config['body']
        self.gen = config['gen']
        self.random = config['random']
        self.solver = config['solver']
        if 'reward' in config:
            self.reward = config['reward']
        else:
            self.reward = None
        self.wing['omega'] = self.wing['bps'] * 2 * np.pi
        self.wing['T'] = 1 / self.wing['bps']

        ### recreate aero model based on new values
        self.aero_model_R = AeroModel(self.wing['span'], self.wing['chord'], self.gen['rho'], self.aero['r22'],
                                      self.aero['CLmax'], self.aero['CDmax'], self.aero['CD0'], self.wing['hingeR'],
                                      self.wing['ACloc'])
        self.aero_model_L = AeroModel(self.wing['span'], self.wing['chord'], self.gen['rho'], self.aero['r22'],
                                      self.aero['CLmax'], self.aero['CDmax'], self.aero['CD0'], self.wing['hingeL'],
                                      self.wing['ACloc'])
        self.state = None

    def step(self, action):
        self.gen['t'] += 1
        tau_ext = self.gen['TauExt'] * (
                self.gen['time4Tau'][0] < self.gen['t'] < self.gen['time4Tau'][1])
        u = self.calc_u(action)
        if not hasattr(self, 'y0'):
            self.y0 = self.state[:9]
        if self.gen['controlled']:  # To calculate one step
            tvec = np.linspace(0, self.wing['T'], 5)

            sol = solve_ivp(self._fly_sim, [0, self.wing['T']], self.y0, method=self.solver['method'], t_eval=tvec,
                            args=[tau_ext, u], atol=self.solver['atol'], rtol=self.solver['rtol'],
                            events=self.pitch_angle_over_88)

        else:  # To calculate all flight
            tvec = np.arange(self.gen['tsim_in'], self.gen['tsim_fin'], self.wing['T'] / 4)  # self.gen['MaxStepSize'])
            sol = solve_ivp(self._fly_sim, [self.gen['tsim_in'], self.gen['tsim_fin']], self.y0,
                            method=self.solver['method'],
                            t_eval=tvec, args=[tau_ext, u], atol=self.solver['atol'], rtol=self.solver['rtol'])
            avg_sol = (sol.y[:, 3::4] + sol.y[:, 1::4]) / 2

            return avg_sol.T, 0, False, 'none'
        self.y0 = sol.y[:, -1]

        self.state = np.mean(sol.y[:, [1, 3]], axis=1)
        done = self.gen['t'] >= self.gen['tsim_fin'] / self.wing['T']
        if self.reward is not None and self.reward['target']:
            if np.abs(self.state[7] / DEG2RAD + 45) <= 5 and np.abs(self.state[4]) / DEG2RAD <= 10:
                reward = self.reward['target']
                self.tot_rwd += reward
                done = True
        elif 'gains' in self.reward:
            obs_temp = self.state.copy()
            obs_temp[3:] = obs_temp[3:] / DEG2RAD  # Convert angles and angular velocities to degrees
            obs_temp = obs_temp - self.reward['set_point']  # from absolute values to errors w.r.t set point
            obs_temp = np.abs(
                obs_temp)  # this makes sure we only give negative reward (remember, this is not a controller gain)
            obs_temp[7:] = obs_temp[7:] % 360  # if we're back where we started we want to stay there
            reward = sum(self.reward['gains'] * obs_temp)
            # reward = self.reward['gains'][7]* (np.abs(self.state[7] / DEG2RAD + 45) % 360)+...
        else:
            reward = -0.1 * (np.abs(self.state[7] / DEG2RAD + 45) % 360) #- 0.0005 * (np.abs(self.state[4] / DEG2RAD))
            self.tot_rwd += reward
        if len(sol.t_events[0]) >= 1:
            reward = -200
            self.tot_rwd += reward
            done = True
        if done:
            self.tot_rwd = 0

        info = {'t': sol.t, 'traj_done': done}

        return self.state, reward, done, info

    def reset(self):
        if self.random['randomize']:
            rnd_vel = self._rng.uniform(-1, 1, size=(3,)) * self.random['vel']
            rnd_pqr = self._rng.uniform(-1, 1, size=(3,)) * self.random['pqr']
            rnd_ang = self._rng.uniform(-1, 1, size=(3,)) * self.random['ang']
        else:
            rnd_vel = np.zeros(3)
            rnd_pqr = np.zeros(3)
            rnd_ang = np.zeros(3)
        x0 = np.concatenate([
            self.body['BodIniVel'] + rnd_vel,
            self.body['BodInipqr'] + rnd_pqr,
            self.body['BodIniang'] + rnd_ang]).T
        self.state = x0
        self.y0 = x0
        self.gen['t'] = 0
        if self.gen['controlled']:
            _ = self.step([0])
        self.tot_rwd = 0
        return self.state

    def render(self, mode='human'):
        return NotImplementedError('No Vis yet')

    def close(self):
        return None

    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def _fly_sim(self, t, y, tau_ext, u0):

        x1 = y[0]  # u
        x2 = y[1]  # v
        x3 = y[2]  # w
        x4 = y[3]  # p
        x5 = y[4]  # q
        x6 = y[5]  # r
        x7 = y[6]  # psi(roll)
        x8 = y[7]  # theta(pitch)
        x9 = y[8]  # phi(roll)

        u1 = u0[0:2]  # psi wing L
        u2 = u0[2:4]  # theta wing L
        u3 = u0[4:6]  # phi wing L
        u4 = u0[6:8]  # psi wing R
        u5 = u0[8:10]  # theta wing R
        u6 = u0[10:12]  # phi wing R
        wingout_r = self.wing_block(x1, x2, x3, x4, x5, x6, x7, x8, x9, u4, u5, u6, 'R', t)
        wingout_l = self.wing_block(x1, x2, x3, x4, x5, x6, x7, x8, x9, u1, u2, u3, 'L', t)
        vb = np.array([x1, x2, x3])
        fb = wingout_r[0] + wingout_l[0] + self.gen['m'] * wingout_r[2].T @ self.gen['g'].T
        tb = wingout_r[1] + wingout_l[1] + tau_ext

        omega_b = np.array([x4, x5, x6]).T
        x1to3dot = (1 / self.gen['m']) * fb - np.cross(omega_b, vb)
        x4to6dot = inv(self.gen['I']) @ (tb - np.cross(omega_b, self.gen['I'] @ omega_b))
        x7to9dot = body_ang_vel_pqr(np.array([x7, x8, x9]), omega_b, False)
        y_dot = np.concatenate([x1to3dot, x4to6dot, x7to9dot])
        return y_dot

    def wing_block(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, u1, u2, u3, wing_rl, t):

        angles, angles_dot = wing_angles(u1, u2, u3, self.wing['omega'], self.wing['delta_psi'],
                                         self.wing['delta_theta'], self.wing['C'], self.wing['K'], t)
        r_wing2lab = Rotation.from_euler('xyz', [angles[0], angles[1], angles[2]]).as_matrix()
        r_sp2lab = Rotation.from_euler('xyz', self.gen['strkplnAng']).as_matrix()
        r_body2lab = Rotation.from_euler('xyz', [x7, x8, x9]).as_matrix()
        r_spwithbod2lab = r_body2lab @ r_sp2lab

        # Wing velocity
        angular_vel = body_ang_vel_pqr(angles, angles_dot, True)
        tang_wing_v = np.cross(angular_vel, self.wing['speedCalc'])

        # Body velocity
        ac_lab = r_spwithbod2lab @ r_wing2lab @ self.wing['speedCalc']
        ac_bod = r_body2lab.T @ ac_lab
        vel_loc_bod = ac_bod + self.wing[f'hinge{wing_rl}'].T
        vb = np.cross(np.array([x4, x5, x6]), vel_loc_bod) + np.array([x1, x2, x3])
        vb = r_body2lab @ vb
        vb_lab = r_spwithbod2lab.T @ vb
        vb_wing = r_wing2lab.T @ vb_lab
        vw = vb_wing + tang_wing_v.T
        alpha = np.arctan2(vw[2], vw[1])
        if wing_rl == 'R':
            cl, cd, span_hat, lhat, drag, lift, t_body, f_body_aero, f_lab_aero, t_lab = self.aero_model_R.get_forces(
                alpha, vw,
                r_body2lab,
                r_wing2lab,
                r_spwithbod2lab)
        if wing_rl == 'L':
            cl, cd, span_hat, lhat, drag, lift, t_body, f_body_aero, f_lab_aero, t_lab = self.aero_model_L.get_forces(
                alpha, vw,
                r_body2lab,
                r_wing2lab,
                r_spwithbod2lab)
        return (f_body_aero, t_body, r_body2lab, r_wing2lab, r_spwithbod2lab, cl, cd, angles.T,
                span_hat, lhat, drag, lift, f_lab_aero, t_lab, ac_lab)

    def calc_u(self, action):
        u0 = np.concatenate([
            self.wing['psi'][0:2],
            self.wing['theta'][0:2],
            self.wing['phi'][0:2],
            self.wing['psi'][2:4],
            self.wing['theta'][2: 4],
            self.wing['phi'][2:4]]).T

        # delta_clip prevents the wings from going over 180 or under 0 degrees
        delta_clip = min(self.wing['phi'][0] - self.wing['phi'][1],  # Calculation of backwards for left wing
                         180 * DEG2RAD - self.wing['phi'][2] + self.wing['phi'][3])  # Same calculation for fwd
        delta_phi = min(action[0], delta_clip)

        u0[4] = self.wing['phi'][0] - delta_phi / 2  # Change in middle point of the stroke (Left)
        u0[5] = self.wing['phi'][1] + delta_phi / 2  # Change in the amplitude of the stroke (Left)
        u0[10] = self.wing['phi'][2] + delta_phi / 2  # Change in middle point of the stroke (Right)
        u0[11] = self.wing['phi'][3] - delta_phi / 2  # Change in the amplitude of the stroke (Right)
        return u0


def test_zero_cont():
    import time
    # env = gym.make('flySim-v0')
    env = flySimEnv()
    env.gen['controlled'] = False
    o = env.reset()
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []
    start = time.time()
    observations, _, _, info = env.step([0])
    end = time.time()
    print(f'{len(observations)} zero control continuous steps completed in {end - start} seconds')
    print(f'total reward for no control is: {np.sum(rewards)}')
    np.savez('test_res_zero_cont', observations=observations)


def test_zero():
    import time
    # env = gym.make('flySim-v0')
    env = flySimEnv()
    env.gen['controlled'] = True
    o = env.reset()
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []
    start = time.time()
    d = False
    while not d:
        observations.append(o)
        a = 0
        o, r, d, info = env.step([a])
        actions.append(a)
        rewards.append(r)
        dones.append(d)
        infos.append(info)
    end = time.time()
    print(f'{len(rewards)} zero control steps completed in {end - start} seconds')
    actions = np.array(actions)
    rewards = np.array(rewards)
    observations = np.array(observations)
    print(f'total reward for no control is: {np.sum(rewards)}')
    np.savez('test_res_zero', actions=actions, rewards=rewards, observations=observations)


def test_random():
    import time
    # env = gym.make('flySim-v0')
    env = flySimEnv()
    env.gen['controlled'] = True
    o = env.reset()
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []
    start = time.time()
    d = False
    while not d:
        observations.append(o)
        a = env.action_space.sample()
        o, r, d, info = env.step(a)
        actions.append(a)
        rewards.append(r)
        dones.append(d)
        infos.append(info)
    end = time.time()
    print(f'{len(rewards)} random steps completed in {end - start} seconds')
    actions = np.array(actions)
    rewards = np.array(rewards)
    observations = np.array(observations)
    print(f'total reward for random behaviour is: {np.sum(rewards)}')
    np.savez('test_res_rand', actions=actions, rewards=rewards, observations=observations)


def test_linear(i=0, seed=1234):
    import time
    Ki = 0.5
    Kp = 8 / 1000
    body_ref_pitch = -45 * DEG2RAD
    # env = gym.make('flySim-v0')
    env = flySimEnv()
    env.seed(seed)
    env.gen['controlled'] = True
    o = env.reset()
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []
    start = time.time()
    d = False
    while not d:
        theta = o[7]
        theta_dot = body_ang_vel_pqr(o[6:9], o[3:6], False)[1]
        delta_phi = theta_dot * Kp + (theta - (body_ref_pitch)) * Ki
        a = [delta_phi]
        observations.append(o)
        o, r, d, info = env.step(a)
        actions.append(a)
        rewards.append(r)
        dones.append(d)
        infos.append(info)
    end = time.time()
    print(f'{len(rewards)} linear control steps completed in {end - start} seconds')
    actions = np.array(actions)
    rewards = np.array(rewards)
    observations = np.array(observations)
    print(f'total reward for linear control is: {np.sum(rewards)}')
    np.savez(f'test_res_linear_{i}', actions=actions, rewards=rewards, observations=observations)


if __name__ == '__main__':
    test_zero_cont()
    test_zero()
    test_random()
    test_linear()
    # ss = np.random.SeedSequence(1234)
    # seeds = ss.spawn(100)
    # # with mp.Pool(6) as p:
    # #     p.starmap(test_linear,zip(range(100),seeds))
    # for i in range(100):
