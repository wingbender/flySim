## GYM imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import gym_flySim

## Numpy/Scipy imports
import numpy as np
from scipy.integrate import solve_ivp
from numpy import sin, cos, tanh, arcsin
from scipy.spatial.transform import Rotation
from numpy.linalg import norm, inv

## Constants
DEG2RAD = np.pi / 180


def body_ang_vel_pqr(angles, angles_dot, get_pqr):
    '''
    Converts change in euler angles to body rates (if get_pqr is True) or body rates to euler rates (if get_pqr is False)
    :param angles: euler angles (np.array[psi,theta,phi])
    :param angles_dot: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
    :param get_pqr: whether to get body rates from euler rates or the other way around
    :return: euler rates or body rates (np.array[d(psi)/dt,d(theta)/dt,d(phi)/dt] or np.array([p,q,r])
    '''
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
    '''
    computes the wing angles given a set of variables, described in (TODO: ask roni for the article)
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
    '''
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

    def __init__(self):
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
            'BodInipqr': np.array([0, 0, 0]),  # body initial angular velocity[m / s]
            'BodIniXYZ': np.array([0, 0, 0]),  # body initial position(lab ax) [m]
            'BodRefPitch': -45 * DEG2RAD
        }
        self.gen = {
            'm': 1e-6,  # mass[kg]
            'g': np.array([0, 0, -9.8]),  # gravity[m / s ^ 2]
            'strkplnAng': np.array([0, 45, 0]) * DEG2RAD,  # stroke plane(compare to body) [rad]
            'rho': 1.225,  # air density
            'TauExt': np.array([0, 0, 0]) * 1e-7,  # external torque body axes[Nm]
            'time4Tau': np.array([10, 15]),  # initial and final time of the external torque[ms]
            'I': 1.0e-12 * np.array([[0.1440, 0, 0], [0, 0.5220, 0], [0, 0, 0.5220]]),  # tensor of inertia[Kg m ^ 2]
            'controlled': True,
            't': 0,
            'tsim_in': 0.0,  # simulation initial time[s]
            'tsim_fin': 0.1,
            'MaxStepSize': 1.0 / 16000

        }
        self.aero = {
            'CLmax': 1.8,
            'CDmax': 3.4,
            'CD0': 0.4,
            'r22': 0.4,
        }
        self.aero_model_R = AeroModel(self.wing['span'], self.wing['chord'], self.gen['rho'], self.aero['r22'],
                                      self.aero['CLmax'], self.aero['CDmax'], self.aero['CD0'], self.wing['hingeR'],
                                      self.wing['ACloc'])
        self.aero_model_L = AeroModel(self.wing['span'], self.wing['chord'], self.gen['rho'], self.aero['r22'],
                                      self.aero['CLmax'], self.aero['CDmax'], self.aero['CD0'], self.wing['hingeL'],
                                      self.wing['ACloc'])
        self.state = None

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,)
                                            )  # psi,theta,phi
        self.action_space = spaces.Box(-0.1, +0.1, (1,), dtype=np.float32)
        self.ranom_start = False
        # self.spaces = {'observation':self.observation_space,'action': self.action_space}

    def step(self, action):
        self.gen['t'] += 1
        tau_ext = self.gen['TauExt'] * (
                self.gen['time4Tau'][0] < self.gen['t'] < self.gen['time4Tau'][1])
        u = self.calc_u(action)
        if self.gen['controlled']:

            tvec = np.linspace(0, self.wing['T'], 5)
            y0 = self.state[:9, -1] if self.state.ndim > 1 else self.state[:9]
            sol = solve_ivp(self._fly_sim, [0, self.wing['T']], y0, method='RK45', t_eval=tvec,
                            args=[tau_ext, u],
                            atol=1e-5, rtol=1e-4)
        else:
            tvec = np.arange(self.gen['tsim_in'], self.gen['tsim_fin'], self.gen['MaxStepSize'])
            sol = solve_ivp(self._fly_sim, [self.gen['tsim_in'], self.gen['tsim_fin']], self.state[:9], method='RK45',
                            t_eval=tvec,
                            args=[tau_ext, u],
                            atol=1e-6, rtol=1e-5)
        self.state = np.mean(sol.y[:, [1, 3]], axis=1)
        self.state[6:9] = self.state[6:9] % np.pi * np.sign(self.state[6:9])
        reward = 360 * DEG2RAD - np.abs(self.state[7])
        done = self.gen['t'] == 100
        info = {'t': sol.t}

        return self.state, reward, done, info

    def reset(self):
        if self.ranom_start:
            rnd_vel = np.random.normal(0, 0.1, size=(3,))
            rnd_pqr = np.random.normal(0, 100, size=(3,))
            rnd_ang = np.random.normal(0, 2 * DEG2RAD, size=(3,))
        else:
            rnd_vel = np.zeros(3)
            rnd_pqr = np.zeros(3)
            rnd_ang = np.zeros(3)
        x0 = np.concatenate([
            self.body['BodIniVel'] + rnd_vel,
            self.body['BodInipqr'] + rnd_pqr,
            self.body['BodIniang'] + rnd_ang]).T
        # u0 = np.concatenate([
        #     self.wing['psi'][0:2],
        #     self.wing['theta'][0:2],
        #     self.wing['phi'][0:2],
        #     self.wing['psi'][2:4],
        #     self.wing['theta'][2: 4],
        #     self.wing['phi'][2:4]]).T
        # self.state = np.concatenate([x0, u0])
        self.state = x0
        self.gen['t'] = 0
        return self.state

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        return None

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
        if not self.gen['controlled']:
            tau_ext = self.gen['TauExt'] * (
                    self.gen['time4Tau'][0] / 1000.0 <= t <= self.gen['time4Tau'][1] / 1000.0)
        wingout_r = self.wing_block(x1, x2, x3, x4, x5, x6, x7, x8, x9, u4, u5, u6, 'R', t)
        wingout_l = self.wing_block(x1, x2, x3, x4, x5, x6, x7, x8, x9, u1, u2, u3, 'L', t)
        vb = np.array([x1, x2, x3])
        fb = wingout_r[0] + wingout_l[0] + self.gen['m'] * wingout_r[2].T @ self.gen['g'].T
        tb = wingout_r[1] + wingout_l[1] + tau_ext
        # body = np.concatenate([fb, tb])

        omega_b = np.array([x4, x5, x6]).T
        x1to3dot = (1 / self.gen['m']) * fb - np.cross(omega_b, vb)
        x4to6dot = inv(self.gen['I']) @ (tb - np.cross(omega_b, self.gen['I'] @ omega_b))
        x7to9dot = body_ang_vel_pqr(np.array([x7, x8, x9]), omega_b, False)
        y_dot = np.concatenate([x1to3dot, x4to6dot, x7to9dot])
        if True:
            mask = [1,0,1,0,]
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
        if self.gen['controlled']:
            # print(action)
            delta_phi = action[0]
            u0 = np.concatenate([
                self.wing['psi'][0:2],
                self.wing['theta'][0:2],
                self.wing['phi'][0:2],
                self.wing['psi'][2:4],
                self.wing['theta'][2: 4],
                self.wing['phi'][2:4]]).T
            u0[4] = self.wing['phi'][0] - delta_phi / 2
            u0[5] = self.wing['phi'][1] + delta_phi / 2
            u0[10] = self.wing['phi'][2] + delta_phi / 2
            u0[11] = self.wing['phi'][3] - delta_phi / 2
            # u0 = np.concatenate(
            #     # psi0_L
            #     # psim_L  psim_R = -psim_L;
            #     # theta0_L theta0_R = theta0_L
            #     # thetam_L thetam_R = thetam_L
            #     # phi0_L phi0_R = -phi0_L
            #     # phim_L phim_R = -phim_L[rad]
            #     # phi0_R phi0_R = -phi0_L
            #     # phim_R phim_R = -phim_L[rad]
            #     # psi0_R  psi0_R = 90
            #     # psim_R  psim_R = -psim_L
            #     # theta0_R theta0_R = theta0_L
            #     # thetam_R thetam_R = thetam_L [rad]
            # )
        else:
            u0 = np.concatenate([
                self.wing['psi'][0:2],
                self.wing['theta'][0:2],
                self.wing['phi'][0:2],
                self.wing['psi'][2:4],
                self.wing['theta'][2: 4],
                self.wing['phi'][2:4]]).T
        return u0


def test_zero_cont():
    import time
    env = gym.make('flySim-v0')
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
    print(f'100 zero control continuous steps completed in {end - start} seconds')
    observations = np.array(observations)
    print(f'total reward for no control is: {np.sum(rewards)}')
    np.savez('test_res_zero_cont', observations=observations)


def test_zero():
    import time
    env = gym.make('flySim-v0')
    o = env.reset()
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []
    start = time.time()
    for ii in range(100):
        observations.append(o)
        a = 0
        o, r, d, info = env.step([a])
        actions.append(a)
        rewards.append(r)
        dones.append(d)
        infos.append(info)
    end = time.time()
    print(f'100 zero control steps completed in {end - start} seconds')
    actions = np.array(actions)
    rewards = np.array(rewards)
    observations = np.array(observations)
    print(f'total reward for no control is: {np.sum(rewards)}')
    np.savez('test_res_zero', actions=actions, rewards=rewards, observations=observations)


def test_random():
    import time
    env = gym.make('flySim-v0')
    o = env.reset()
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []
    start = time.time()
    for ii in range(100):
        observations.append(o)
        a = env.action_space.sample()
        o, r, d, info = env.step(a)
        actions.append(a)
        rewards.append(r)
        dones.append(d)
        infos.append(info)
    end = time.time()
    print(f'100 random steps completed in {end - start} seconds')
    actions = np.array(actions)
    rewards = np.array(rewards)
    observations = np.array(observations)
    print(f'total reward for random behaviour is: {np.sum(rewards)}')
    np.savez('test_res_rand', actions=actions, rewards=rewards, observations=observations)


def test_linear():
    import time
    Ki = 0.5
    Kp = 8 / 1000
    body_ref_pitch = -45 * DEG2RAD

    env = gym.make('flySim-v0')
    o = env.reset()
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []
    start = time.time()
    for ii in range(100):
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
    print(f'100 linear control steps completed in {end - start} seconds')
    actions = np.array(actions)
    rewards = np.array(rewards)
    observations = np.array(observations)
    print(f'total reward for linear control is: {np.sum(rewards)}')
    np.savez('test_res_linear', actions=actions, rewards=rewards, observations=observations)


if __name__ == '__main__':
    print(f'maximal reward possible is: {360 * DEG2RAD * 100}')
    test_zero_cont()
    test_zero()
    test_random()
    test_linear()
