import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ctypes
import os
import pygame
import math

class CartPoleCustomEnv(gym.Env):
    """Environment for CartPole using a C backend for physics simulation."""

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 1600
        self.screen_height = 900

        # 1. Ładowanie silnika C
        lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build', 'libcartpole.so'))
        self.lib = ctypes.CDLL(lib_path)

        self.lib.cp_is_done.restype = ctypes.c_int
        self.lib.cp_create.restype = ctypes.c_void_p

        self.lib.cp_step.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
        self.lib.cp_get_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.cp_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self.lib.cp_is_done.argtypes = [ctypes.c_void_p]

        self.cp_handle = self.lib.cp_create()
        self.state_array = (ctypes.c_float * 4)()

        # Parameters for the environment
        self.force_mag = 20.0
        self.dt = 0.02
        self.current_step = 0

        if self.render_mode == "human":
            self.max_steps = 999999999
        else:
            self.max_steps = 1000  # Swing-up needs time to pump energy AND then balance

        # 2. Define space
        # Action: 1 continuous value in the range [-1.0, 1.0] (from -100% to +100% engine power)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: [x, x_dot, cos(theta), sin(theta), theta_dot]
        # We feed cos/sin instead of raw theta so the angle is continuous (no jump at +-pi)
        # and bounded to [-1, 1] — critical for the swing-up task where the pole spins fully.
        high = np.array([4.8, np.inf, 1.0, 1.0, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Reset the C environment with a seed for reproducibility
        seed_val = seed if seed is not None else 0
        self.lib.cp_reset(self.cp_handle, ctypes.c_uint(seed_val))

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # 1. Scaling the action from [-1, 1] to the actual force magnitude
        # Ensuring that the action is clipped to the valid range
        act = float(np.clip(action[0], -1.0, 1.0))
        force_to_apply = act * self.force_mag

        # 2. Phisics simulation step in C
        self.lib.cp_step(self.cp_handle, ctypes.c_float(force_to_apply), ctypes.c_float(self.dt))

        # 3. Read the new state from the C environment
        x, x_dot, theta, theta_dot = self._read_state()
        obs = self._get_obs()

        # 4. Check the termination conditions
        terminated = bool(self.lib.cp_is_done(self.cp_handle))
        truncated = bool(self.current_step >= self.max_steps)

        # 5. Swing-up reward:
        #    - upright term: 1.0 when balanced (theta=0), 0.0 when hanging (theta=pi)
        #    - penalties keep the cart centered, discourage frantic spinning and wasted force
        upright = (math.cos(theta) + 1.0) / 2.0
        reward = (
            upright
            - 0.05 * (abs(x) / 2.4)
            - 0.001 * (theta_dot ** 2)   # gently discourage endless spinning
            - 0.01 * (act ** 2)          # small effort penalty (act in [-1, 1])
        )

        if terminated: reward -= 100.0  # Penalty for driving off the track

        return obs, reward, terminated, truncated, {}

    def _read_state(self):
        """Raw physics state from the C engine: [x, x_dot, theta, theta_dot]."""
        self.lib.cp_get_state(self.cp_handle, self.state_array)
        return (self.state_array[0], self.state_array[1],
                self.state_array[2], self.state_array[3])

    def _get_obs(self):
        """Network observation: angle encoded as cos/sin for continuity."""
        x, x_dot, theta, theta_dot = self._read_state()
        return np.array(
            [x, x_dot, math.cos(theta), math.sin(theta), theta_dot],
            dtype=np.float32,
        )

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("CartPole - Wytrenowany Agent (C + Pygame)")
            self.clock = pygame.time.Clock()

        # Load the current state to visualize
        x, x_dot, theta, theta_dot = self._read_state()

        # Clear the screen with white background
        self.screen.fill((255, 255, 255))

        # Scaling factors to convert from physics coordinates to screen pixels
        scale = 100.0
        cart_px = int(self.screen_width / 2 + x * scale)
        cart_py = int(self.screen_height / 2 + 50)

        # Draw the track (gray line)
        pygame.draw.line(self.screen, (200, 200, 200), (0, cart_py), (self.screen_width, cart_py), 2)

        # Draw the cart (blue rectangle)
        cart_width, cart_height = 50, 30
        pygame.draw.rect(self.screen, (0, 0, 255),
                         (cart_px - cart_width//2, cart_py - cart_height//2, cart_width, cart_height))

        # Draw the pole (red line)
        # The pole length in pixels is scaled from the physical length (2.0) by the same scale factor
        pole_len_px = 2.0 * scale
        end_x = cart_px + int(pole_len_px * math.sin(theta))
        end_y = cart_py - int(pole_len_px * math.cos(theta))

        pygame.draw.line(self.screen, (255, 0, 0), (cart_px, cart_py), (end_x, end_y), 8)

        # Refresh the screen and limit to 60 FPS
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        # Cleanup after closing the window
        if self.screen is not None:
            pygame.quit()
            self.screen = None