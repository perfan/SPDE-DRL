import gym
import Burgers
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class SpdeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    
    def __init__(self, u_max, f_max, n_x, x_max, nu, eps):
        self.u_max = u_max
        self.f_max = f_max
        self.n_x = n_x
        self.x_max = x_max
        self.nu = nu
        self.eps = eps
        self.viewer = None
    
        f_high = np.zeros(n_x, dtype = np.float32).fill(self.f_max)
        u_high = np.zeros(n_x, dtype = np.float32).fill(self.u_max)
        self.action_space = spaces.Box(
            low=-f_high,
            high=f_high,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-u_high,
            high=u_high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, f, t_start, t_end, n_t):
        u = self.state  # th := theta

        nu = self.nu
        eps = self.eps

        f = np.clip(f, -self.f_max, self.f_max)
        # costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        new_u = Burgers.convection_diffusion(u, x, t, NT, NX, T_START, T_END, XMAX, NU, EPS, prev_condition)
        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)