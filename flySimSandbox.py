import numpy as np
from scipy.integrate import solve_ivp
from numpy import sin, cos, tanh, arcsin
from scipy.spatial.transform import Rotation
from numpy.linalg import norm, inv
from scipy.io import loadmat
import matplotlib.pyplot as plt

DEG2RAD = np.pi / 180


def body_ang_vel_pqr(angles, angles_dot, get_pqr):
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


def aero_model(aero, gen, wing_inp, wing_rl, v_wing, aoa, ac_loc_wing, rotation_mat_body2lab,
               rotation_mat_wing2lab, rotation_mat_sp2lab):
    s = wing_inp['span'] * wing_inp['cord'] * np.pi / 4

    cl = aero['CLmax'] * sin(2 * aoa)
    cd = (aero['CDmax'] + aero['CD0']) / 2 - (aero['CDmax'] - aero['CD0']) / 2 * cos(2 * aoa)
    u = v_wing[0] ** 2 + v_wing[1] ** 2 + v_wing[2] ** 2
    uhat = v_wing / norm(v_wing)
    span_hat = np.array([1, 0, 0])
    lhat = (np.cross(span_hat, -uhat)).T  # perpendicular to Uhat
    lhat = lhat / norm(lhat)
    drag = -0.5 * cd * gen['rho'] * s * aero['r22'] * uhat * u
    lift = 0.5 * cl * gen['rho'] * s * aero['r22'] * lhat * u
    ac_loc_lab = rotation_mat_sp2lab @ rotation_mat_wing2lab @ ac_loc_wing.T + rotation_mat_body2lab @ wing_inp[
        f'hinge{wing_rl}'].T  # AC location in lab axes
    ac_loc_body = rotation_mat_body2lab.T @ ac_loc_lab  # AC location in body axes

    f_lab_aero = rotation_mat_sp2lab @ rotation_mat_wing2lab @ lift + rotation_mat_sp2lab @ rotation_mat_wing2lab @ drag
    # force in body axes
    f_body = rotation_mat_body2lab.T @ f_lab_aero
    t_lab = np.cross(ac_loc_lab.T,
                     f_lab_aero).T  # + cross(ACLocB_body, Dbod).T # torque on body (in body axes)
    # (from forces, no CM0)
    t_body = np.cross(ac_loc_body.T,
                      f_body).T  # + cross(ACLocB_lab, Dbod_lab) # torque on body( in bodyaxes)
    # (from forces, no CM0)
    return cl, cd, span_hat, lhat, drag, lift, t_body, f_body, f_lab_aero, t_lab


def rot_mat_body2lab(angles):
    # return Rotation.from_euler('xyz', angles[0:3]).as_matrix()
    psi_w = angles[0]
    theta_w = angles[1]
    phi_w = angles[2]
    Rx = np.array([[1, 0, 0],
                   [0, cos(psi_w), - sin(psi_w)],
                   [0, sin(psi_w), cos(psi_w)]])
    Ry = np.array([[cos(theta_w), 0, sin(theta_w)],
                   [0, 1, 0],
                   [-sin(theta_w), 0, cos(theta_w)]])
    Rz = np.array([[cos(phi_w), -sin(phi_w), 0],
                   [sin(phi_w), cos(phi_w), 0],
                   [0, 0, 1]])
    return np.round(Rz @ Ry @ Rx * 100000000) / 100000000

    return Rz @ Ry @ Rx


class FlySim(object):
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
            'freq': 220 * np.pi * 2,  # wing frequency[rad / s]
            'span': 2.5 / 1000,  # span[m]
            'cord': 0.7 / 1000,  # cord[m]
            'speedCalc': np.array([2.5 / 1000, 0, 0])
            # location on wing used to calculate the wing's tangential velocity
        }
        self.wing['T'] = 1.0 / (self.wing['freq'] / 2.0 / np.pi)  # seconds
        self.body = {
            'BodIniVel': np.array([0, 0, 0]),  # body initial velocity(body ax) [m / s]
            'BodIniang': np.array([0, -45, 0]) * DEG2RAD,  # body initial angle(lab ax) [rad]
            'BodInipqr': np.array([0, 0, 0]),  # body initial angular velocity[m / s]
            'BodIniXYZ': np.array([0, 0, 0]),  # body initial position(lab ax) [m]
        }
        self.gen = {
            'm': 1e-6,  # mass[kg]
            'g': np.array([0, 0, -9.8]),  # gravity[m / s ^ 2]
            'strkplnAng': np.array([0, 45, 0]) * DEG2RAD,  # stroke plane(compare to body) [rad]
            'rho': 1.225,  # air density
            'TauExt': np.array([0, 0, 0]) * 1e-7,  # external torque body axes[Nm]
            'time4Tau': np.array([10, 15]),  # initial and final time of the external torque[ms]
            'I': 1.0e-12 * np.array([[0.1440, 0, 0], [0, 0.5220, 0], [0, 0, 0.5220]]),  # tensor of inertia[Kg m ^ 2]
            'controlled': False,
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
        self.state = None

    def reset(self):
        x0 = np.concatenate(
            [self.body['BodIniVel'],
             self.body['BodInipqr'],
             self.body['BodIniang']]).T
        u0 = np.concatenate([
            self.wing['psi'][0:2],
            self.wing['theta'][0:2],
            self.wing['phi'][0:2],
            self.wing['psi'][2:4],
            self.wing['theta'][2: 4],
            self.wing['phi'][2:4]]).T
        self.state = np.concatenate([x0, u0])
        return self.state

    def calc_u(self, action):
        if self.gen['controlled']:
            print(action)
            raise NotImplementedError
        else:
            u0 = np.concatenate([
                self.wing['psi'][0:2],
                self.wing['theta'][0:2],
                self.wing['phi'][0:2],
                self.wing['psi'][2:4],
                self.wing['theta'][2: 4],
                self.wing['phi'][2:4]]).T
        return u0

    def step(self, action):
        self.gen['t'] += 1
        tau_ext = self.gen['TauExt'] * (
                self.gen['time4Tau'][0] < self.gen['t'] < self.gen['time4Tau'][1])
        if self.gen['controlled']:
            u = self.calc_u(action)
            sol = solve_ivp(self._fly_sim, [0, self.wing['T']], self.state[:9], method='RK45', t_eval=[self.wing['T']],
                            args=[tau_ext, u],
                            atol=1e-5, rtol=1e-4)
        else:
            tvec = np.arange(self.gen['tsim_in'], self.gen['tsim_fin'], self.gen['MaxStepSize'])
            sol = solve_ivp(self._fly_sim, [self.gen['tsim_in'], self.gen['tsim_fin']], self.state[:9], method='RK45',
                            t_eval=tvec,
                            args=[tau_ext, self.state[9:]],
                            atol=1e-6, rtol=1e-5)
        self.state = sol.y
        reward = 0
        done = False
        info = sol.t
        return self.state, reward, done, info

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
        return y_dot

    def wing_block(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, u1, u2, u3, wing_rl, t):

        angles, angles_dot = wing_angles(u1, u2, u3, self.wing['freq'], self.wing['delta_psi'],
                                         self.wing['delta_theta'], self.wing['C'], self.wing['K'], t)
        r_wing2lab = rot_mat_body2lab(angles)
        r_sp2lab = rot_mat_body2lab(self.gen['strkplnAng'])
        r_body2lab = rot_mat_body2lab(np.array([x7, x8, x9]))
        r_spwithbod2lab = r_body2lab @ r_sp2lab
        # r_wing2lab = Rotation.from_euler('xyz', [angles[0], angles[1], angles[2]]).as_matrix()
        # r_sp2lab = Rotation.from_euler('xyz', self.gen['strkplnAng']).as_matrix()
        # r_body2lab = Rotation.from_euler('xyz', [x7, x8, x9]).as_matrix()
        # r_spwithbod2lab = r_body2lab @ r_sp2lab

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
        cl, cd, span_hat, lhat, drag, lift, t_body, f_body_aero, f_lab_aero, t_lab = aero_model(self.aero,
                                                                                                self.gen,
                                                                                                self.wing,
                                                                                                wing_rl, vw, alpha,
                                                                                                self.wing['ACloc'],
                                                                                                r_body2lab,
                                                                                                r_wing2lab,
                                                                                                r_spwithbod2lab)
        return (f_body_aero, t_body, r_body2lab, r_wing2lab, r_spwithbod2lab, cl, cd, angles.T,
                span_hat, lhat, drag, lift, f_lab_aero, t_lab, ac_lab)


def test(vec, recalc):

    if recalc:
        for i in vec.keys():
            fsim = FlySim()

            fsim.wing['psi'] = np.array([90, 53, 90, -53]) * DEG2RAD
            fsim.wing['theta'] = np.array([0, 0, 0, 0]) * DEG2RAD
            fsim.wing['phi'] = np.array([90, 65, -90, -65]) * DEG2RAD
            fsim.body['BodIniVel'] = np.array([0, 0, 0])  # body initial velocity(body ax) [m / s]
            fsim.body['BodIniang'] = np.array([0, -45, 0]) * DEG2RAD  # body initial angle(lab ax) [rad]
            fsim.body['BodInipqr'] = np.array([0, 0, 0])  # body initial angular velocity[m / s]
            fsim.body['BodIniXYZ'] = np.array([0, 0, 0])  # body initial position(lab ax) [m]
            fsim.gen['strkplnAng'] = np.array([0, 45, 0]) * DEG2RAD
            if vec[i] is not None:
                mult = DEG2RAD if vec[i][0] in ['psi', 'theta', 'phi', 'BodIniang', 'strkplnAng'] else 1
                if vec[i][0] in ['psi', 'theta', 'phi']:
                    fsim.wing[vec[i][0]] = np.array(vec[i][1]) * mult
                elif vec[i][0] in ['BodIniVel', 'BodIniang', 'BodInipqr', 'BodIniXYZ']:
                    fsim.body[vec[i][0]] = np.array(vec[i][1]) * mult
                elif vec[i][0] == 'strkplnAng':
                    fsim.gen['strkplnAng'] = np.array(vec[i][1]) * mult
            fsim.reset()
            state, _, _, info = fsim.step([0])
            np.save(f'./ValidationVectors/results{i}', np.vstack((state, info)))
            print(f'finished test case {i}')
    plotVarVec(vec)


def plotVarVec(vec):
    for i in vec.keys():
        res = np.load(f'./ValidationVectors/results{i}.npy')
        t_sag = res[9, :].T
        uvw_sagiv = res[[0, 1, 2], :].T
        pqr_sagiv = res[[3, 4, 5], :].T
        angles_sagiv = res[[6, 7, 8], :].T
        res_roni = loadmat(f'./ValidationVectors/case{i}')
        t_roni = res_roni['time']
        uvw_roni = res_roni['uvw']
        pqr_roni = res_roni['pqr']
        angles_roni = res_roni['RollPitchYaw']
        f, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].plot(t_sag, uvw_sagiv)
        ax[0].plot(t_roni.T, uvw_roni)
        plt.legend(['u_sag', 'v_sag', 'w_sag', 'u_roni', 'v_roni', 'w_roni'])
        ax[1].plot(t_sag, pqr_sagiv)
        ax[1].plot(t_roni.T, pqr_roni)
        plt.legend(['p_sag', 'q_sag', 'r_sag', 'p_roni', 'q_roni', 'r_roni'])
        ax[2].plot(t_sag, angles_sagiv)
        ax[2].plot(t_roni.T, angles_roni)
        plt.legend(['roll_sag', 'pitch_sag', 'yaw_sag', 'roll_roni', 'pitch_roni', 'yaw_roni'])
        plt.savefig(f'./ValidationVectors/graphs/results{i}.png')
        if i == 12:
            a = 1
        plt.close(f)
        print(f'Saved test case {i}')


def get_test_vec():
    vec = {
        1: None,
        2: (
            'phi', [90, 65, -90, -40]
        ),
        3: (
            'phi', [80, 65, -80, -65]
        ),
        4: (
            'phi', [80, 65, -90, -65]
        ),
        5: None,
        6: (
            'psi', [90, 60, 90, -53]
        ),
        7: (
            'psi', [80, 53, 100, -53]
        ),
        8: (
            'psi', [80, 53, 90, -53]
        ),
        9: None,
        10: (
            'theta', [0, 5, 0, 0]
        ),
        11: (
            'theta', [5, 0, 5, 0]
        ),
        12: (
            'theta', [5, 0, 0, 0]
        ),
        13: (
            'BodIniVel', [0.1, 0, 0]
        ),
        14: (
            'BodIniVel', [0, 0.2, 0]
        ),
        15: (
            'BodIniVel', [0.1, 0.2, 0.1]
        ),
        16: (
            'BodIniang', [0, -60, 0]
        ),
        17: (
            'BodIniang', [20, 0, 0]
        ),
        18: (
            'BodIniang', [0, -10, 30]
        ),
        19: (
            'BodInipqr', [0, 1000, 0]
        ),
        20: (
            'BodInipqr', [500, 600, 0]
        ),
        21: (
            'BodInipqr', [0, 0, 1000]
        ),
        22: (
            'strkplnAng', [10, 0, 0]
        ),
        23: (
            'strkplnAng', [0, 30, 10]
        ),
        24: (
            'strkplnAng', [5, 40, 15]
        ),
    }
    return vec


if __name__ == '__main__':
    test(get_test_vec(), True)
