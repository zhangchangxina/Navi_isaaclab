import gymnasium as gym
import numpy as np
import torch
from typing import Any, Dict, Tuple

# Ensure Isaac tasks are registered to Gym
import isaaclab_tasks  # noqa: F401


class IsaacSingleEnvAdapter(gym.Env):
    """A minimal adapter to use an Isaac Lab ManagerBasedRLEnv with SAC/MBPO code.

    - Converts Isaac's vectorized dict observations to a single flat vector
    - Converts numpy actions to Isaac torch tensors
    - Exposes finite action_space in [-1, 1] for typical continuous control
    """

    metadata = {"render_modes": [None, "rgb_array"]}

    def __init__(self, task_id: str, env_cfg: Any | None = None, device: str = "cuda:0", render: bool = False):
        import gymnasium as gymnasium

        self.device = device
        self._render = render

        # Create Isaac environment (vectorized under the hood). If cfg is provided, pass it through.
        render_mode = "rgb_array" if render else None
        if env_cfg is not None:
            self._env = gymnasium.make(task_id, cfg=env_cfg, render_mode=render_mode)
        else:
            self._env = gymnasium.make(task_id, render_mode=render_mode)

        # Isaac env is vectorized; enforce num_envs == 1 for on-policy SAC loop
        # Note: this adapter expects num_envs == 1; ensure your cfg uses 1 env

        # Determine which observation group is active (policy or policy_uav)
        # Peek a reset to infer shapes
        obs, info = self._env.reset()
        if isinstance(obs, dict):
            if "policy" in obs:
                self._obs_key = "policy"
            elif "policy_uav" in obs:
                self._obs_key = "policy_uav"
            else:
                # pick the first key deterministically
                self._obs_key = sorted(obs.keys())[0]
        else:
            raise RuntimeError("Expected dict observation from Isaac environment")

        # Build observation_space: flat vector
        vec = self._extract_obs_vector(obs)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=vec.shape, dtype=np.float32)

        # Build action_space: use normalized range [-1, 1] per dimension
        # Infer action dimension from env action space
        act_space = getattr(self._env, "action_space", None)
        if act_space is None or getattr(act_space, "shape", None) is None:
            raise RuntimeError("Env action_space is missing or malformed")
        action_dim = int(act_space.shape[-1])
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Cache last obs for render
        self._last_rgb = None

    # -----------------------------
    # Gym API
    # -----------------------------
    def reset(self, seed: int | None = None, options: Dict | None = None):
        obs, info = self._env.reset(seed=seed, options=options)
        return self._extract_obs_vector(obs), {}

    def step(self, action: np.ndarray):
        # Ensure correct shape and device for Isaac
        if action.ndim == 1:
            action_batched = action[None, :].astype(np.float32)
        else:
            action_batched = action.astype(np.float32)

        action_tensor = torch.from_numpy(action_batched).to(self.device)

        obs, rew, terminated, timed_out, _extras = self._env.step(action_tensor)

        obs_vec = self._extract_obs_vector(obs)
        # Rewards and dones are batched tensors
        if isinstance(rew, torch.Tensor):
            reward = float(rew[0].item())
        else:
            reward = float(np.asarray(rew).reshape(-1)[0])

        if isinstance(terminated, torch.Tensor):
            term = bool(terminated[0].item())
        else:
            term = bool(np.asarray(terminated).reshape(-1)[0])

        if isinstance(timed_out, torch.Tensor):
            trunc = bool(timed_out[0].item())
        else:
            trunc = bool(np.asarray(timed_out).reshape(-1)[0])

        done = term or trunc
        info: Dict[str, Any] = {"terminated": term, "truncated": trunc}
        
        # Extract reward components from extras (RSL-RL style)
        if _extras is not None:
            for key, value in _extras.items():
                if key.startswith("Episode_Reward/"):
                    # Convert tensor to float for single environment
                    if isinstance(value, torch.Tensor):
                        info[key] = float(value[0].item())
                    else:
                        info[key] = float(np.asarray(value).reshape(-1)[0])

        if self._render:
            self._last_rgb = self.render()

        return obs_vec, reward, done, info

    def render(self, mode: str = "rgb_array"):
        frame = self._env.render()
        return frame

    def close(self):
        try:
            self._env.close()
        finally:
            super().close()

    # -----------------------------
    # Helpers
    # -----------------------------
    def _extract_obs_vector(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract single-env observation vector from Isaac dict obs."""
        tensor = obs[self._obs_key]
        if isinstance(tensor, torch.Tensor):
            vec = tensor[0].detach().cpu().numpy().astype(np.float32)
        else:
            vec = np.asarray(tensor)[0].astype(np.float32)
        return vec


class IsaacVecEnvAdapter:
    """Adapter for Isaac vectorized env to batched numpy API for MBPO/SAC collectors.

    reset() -> np.ndarray [num_envs, obs_dim]
    step(actions: np.ndarray[num_envs, act_dim]) -> (next_obs [num_envs, obs_dim], rewards [num_envs], dones [num_envs], info)
    """

    metadata = {"render_modes": [None, "rgb_array"]}

    def __init__(self, task_id: str, env_cfg: Any, device: str = "cuda:0", render: bool = False):
        import gymnasium as gymnasium

        self.device = device
        self._render = render
        render_mode = "rgb_array" if render else None
        self._env = gymnasium.make(task_id, cfg=env_cfg, render_mode=render_mode)

        # Discover obs key
        obs, _ = self._env.reset()
        if isinstance(obs, dict):
            if "policy" in obs:
                self._obs_key = "policy"
            elif "policy_uav" in obs:
                self._obs_key = "policy_uav"
            else:
                self._obs_key = sorted(obs.keys())[0]
        else:
            raise RuntimeError("Expected dict observation from Isaac environment")

        # Shapes
        sample_vec = self._extract_obs_matrix(obs)
        self.num_envs = sample_vec.shape[0]
        self.obs_dim = sample_vec.shape[1]

        # Action dim from action space
        act_space = getattr(self._env, "action_space", None)
        if act_space is None or getattr(act_space, "shape", None) is None:
            raise RuntimeError("Env action_space is missing or malformed")
        self.act_dim = int(act_space.shape[-1])

        # Define unified spaces for reference
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

    def reset(self, seed: int | None = None, options: Dict | None = None):
        obs, info = self._env.reset(seed=seed, options=options)
        return self._extract_obs_matrix(obs), {}

    def step(self, actions: np.ndarray):
        if isinstance(actions, list):
            actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        action_tensor = torch.from_numpy(actions.astype(np.float32)).to(self.device)

        obs, rew, terminated, timed_out, _extras = self._env.step(action_tensor)
        obs_mat = self._extract_obs_matrix(obs)

        if isinstance(rew, torch.Tensor):
            rewards = rew.detach().cpu().numpy().astype(np.float32)
        else:
            rewards = np.asarray(rew, dtype=np.float32)

        if isinstance(terminated, torch.Tensor):
            term = terminated.detach().cpu().numpy().astype(bool)
        else:
            term = np.asarray(terminated).astype(bool)

        if isinstance(timed_out, torch.Tensor):
            trunc = timed_out.detach().cpu().numpy().astype(bool)
        else:
            trunc = np.asarray(timed_out).astype(bool)

        dones = np.logical_or(term, trunc)
        
        # Extract reward components from extras (RSL-RL style)
        info = {}
        if _extras is not None:
            for key, value in _extras.items():
                if key.startswith("Episode_Reward/"):
                    # Convert tensor to numpy array for vectorized environment
                    if isinstance(value, torch.Tensor):
                        info[key] = value.detach().cpu().numpy().astype(np.float32)
                    else:
                        info[key] = np.asarray(value, dtype=np.float32)
        
        return obs_mat, rewards.reshape(-1), dones.reshape(-1), info

    def render(self, mode: str = "rgb_array"):
        return self._env.render()

    def close(self):
        self._env.close()

    def _extract_obs_matrix(self, obs: Dict[str, Any]) -> np.ndarray:
        tensor = obs[self._obs_key]
        if isinstance(tensor, torch.Tensor):
            mat = tensor.detach().cpu().numpy().astype(np.float32)
        else:
            mat = np.asarray(tensor, dtype=np.float32)
        return mat


