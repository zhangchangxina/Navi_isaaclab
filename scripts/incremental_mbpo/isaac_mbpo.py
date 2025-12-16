import numpy as np
import torch


class PredictEnvIsaac:
    """Minimal predict env wrapper compatible with MBPO's rollout_model for Isaac.

    Here we assume an already-trained ensemble model implementing:
      - predict(inputs) -> (means, variances) in numpy, like ensemble_model.EnsembleDynamicsModel

    For now, this is a thin wrapper; termination is set to zeros to allow rollouts.
    """

    def __init__(self, model, obs_dim: int, act_dim: int):
        self.model = model
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def step(self, obs: np.ndarray, act: np.ndarray):
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()

        inputs = np.concatenate([obs, act], axis=-1)
        means, variances = self.model.predict(inputs)
        # means: [B, 1+obs_dim] where first column is reward, rest is delta_obs
        # variances: [B, 1+obs_dim]
        stds = np.sqrt(variances)

        # Sample from the predicted distribution
        batch_size = means.shape[0]
        samples = means + stds * np.random.normal(size=means.shape)

        rewards = samples[:, :1]
        delta_obs = samples[:, 1:]
        next_obs = obs + delta_obs
        terminals = np.zeros((batch_size, 1), dtype=bool)
        info = {}
        return next_obs, rewards.squeeze(-1), terminals.squeeze(-1), info


class IsaacEnvSamplerSimple:
    """Simple real-environment sampler for Isaac single-env adapter (no LQR)."""

    def __init__(self, env, max_path_length: int = 1000):
        self.env = env
        self.max_path_length = max_path_length
        self.path_length = 0
        self.current_state = None

    def sample(self, agent, eval_t: bool = False):
        if self.current_state is None:
            obs, _ = self.env.reset()
            self.current_state = obs

        cur_state = self.current_state
        action = agent.select_action(cur_state, eval=eval_t) if hasattr(agent, "select_action") else self.env.action_space.sample()
        next_state, reward, done, info = self.env.step(action)
        self.path_length += 1
        if done or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
        else:
            self.current_state = next_state
        return cur_state, action, next_state, float(reward), bool(done), info


class IsaacVecSampler:
    """Vectorized sampler: step all envs in parallel and return batched transitions.

    Expects env to be an instance of IsaacVecEnvAdapter.
    """

    def __init__(self, vec_env, max_path_length: int = 1000):
        self.env = vec_env
        self.max_path_length = max_path_length
        self.path_lengths = np.zeros(self.env.num_envs, dtype=np.int32)
        self.states, _ = self.env.reset()

    def sample(self, agent, eval_t: bool = False):
        actions = np.stack([agent.select_action(s, eval=eval_t) for s in self.states], axis=0)
        next_states, rewards, dones, info = self.env.step(actions)
        cur_states = self.states
        self.states = next_states
        self.path_lengths += 1
        resets = np.where(np.logical_or(dones, self.path_lengths >= self.max_path_length))[0]
        if resets.size > 0:
            new_states, _ = self.env.reset()
            self.states[resets] = new_states[resets]
            self.path_lengths[resets] = 0
        return cur_states, actions, next_states, rewards, dones, info


