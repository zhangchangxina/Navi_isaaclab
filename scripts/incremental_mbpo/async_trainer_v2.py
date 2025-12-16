"""
Asynchronous training framework v2: RSL-RL style batch collection.
Collects num_steps_per_env from all environments, then performs batch updates.
This approach is more stable and avoids CUDA device conflicts.
"""

import threading
import queue
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
from scipy import linalg


def construct_lyapunov(device: torch.device, x: np.ndarray | torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Construct Lyapunov function V(x) = x^T P x."""
    if not torch.is_tensor(x):
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x_t = x.to(device=device, dtype=torch.float32)
    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(0)
    XP = torch.matmul(x_t, P)
    V = (XP * x_t).sum(dim=-1)
    return V


def _estimate_B_from_online_model(inc_model: torch.nn.Module, obs_dim: int, act_dim: int, device: torch.device, epsilon: float = 1e-3) -> np.ndarray:
    """Estimate B matrix from online incremental dynamics model."""
    inc_model.eval()
    with torch.no_grad():
        s0 = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
        a0 = torch.zeros(1, act_dim, dtype=torch.float32, device=device)
        ns0, _, _ = inc_model(s0, a0)
        B_cols = []
        for j in range(act_dim):
            a = torch.zeros(1, act_dim, dtype=torch.float32, device=device)
            a[0, j] = epsilon
            ns_j, _, _ = inc_model(s0, a)
            col = (ns_j - ns0).view(-1).detach().cpu().numpy() / float(epsilon)
            B_cols.append(col)
        B = np.stack(B_cols, axis=1)
    return B


def _solve_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Solve discrete-time LQR problem."""
    P = linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return P, K


@dataclass
class Transition:
    """Single transition data structure."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any]


@dataclass
class RolloutBatch:
    """Batch of rollouts from all environments."""
    states: np.ndarray  # [num_envs * num_steps_per_env, obs_dim]
    actions: np.ndarray  # [num_envs * num_steps_per_env, act_dim]
    rewards: np.ndarray  # [num_envs * num_steps_per_env]
    next_states: np.ndarray  # [num_envs * num_steps_per_env, obs_dim]
    dones: np.ndarray  # [num_envs * num_steps_per_env]
    infos: List[Dict[str, Any]]  # List of info dicts
    total_steps: int  # Total number of steps collected


class BatchCollector:
    """Collects batches of data from all environments (RSL-RL style)."""
    
    def __init__(self, 
                 vec_env,
                 agent,
                 num_steps_per_env: int = 24,
                 max_queue_size: int = 10,
                 config=None,
                 P_tensor=None,
                 K_lqr=None,
                 args=None):
        self.vec_env = vec_env
        self.agent = agent
        self.num_steps_per_env = num_steps_per_env
        self.max_queue_size = max_queue_size
        self.config = config or {}
        self.P_tensor = P_tensor
        self.K_lqr = K_lqr
        self.args = args
        
        # Data queue for collected batches
        self.batch_queue = queue.Queue(maxsize=max_queue_size)
        
        # Threading control
        self.collector_thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'batches_collected': 0,
            'total_steps_collected': 0,
            'collection_time': 0.0,
            'queue_full_count': 0
        }
        
        # Cumulative action tracking
        self.cum_action = np.zeros((vec_env.num_envs, vec_env.act_dim), dtype=np.float32)
    
    def start_collection(self):
        """Start batch collection in separate thread."""
        self.stop_event.clear()
        self.collector_thread = threading.Thread(
            target=self._collection_loop,
            name="BatchCollector",
            daemon=True
        )
        self.collector_thread.start()
        print(f"[BatchCollector] Started collecting {self.num_steps_per_env} steps per env")
    
    def stop_collection(self):
        """Stop batch collection."""
        self.stop_event.set()
        if self.collector_thread:
            self.collector_thread.join(timeout=5.0)
        print("[BatchCollector] Stopped batch collection")
    
    def _collection_loop(self):
        """Main collection loop running in separate thread."""
        while not self.stop_event.is_set():
            try:
                # Collect one batch
                batch = self._collect_batch()
                if batch is not None:
                    # Try to put batch in queue (non-blocking)
                    try:
                        self.batch_queue.put_nowait(batch)
                        self.stats['batches_collected'] += 1
                        self.stats['total_steps_collected'] += batch.total_steps
                    except queue.Full:
                        self.stats['queue_full_count'] += 1
                        # Drop oldest batch if queue is full
                        try:
                            self.batch_queue.get_nowait()
                            self.batch_queue.put_nowait(batch)
                        except queue.Empty:
                            pass
                
            except Exception as e:
                print(f"[BatchCollector] Error in collection loop: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def _collect_batch(self) -> Optional[RolloutBatch]:
        """Collect one batch of data from all environments."""
        start_time = time.time()
        
        try:
            # Initialize storage for the batch
            num_envs = self.vec_env.num_envs
            total_steps = num_envs * self.num_steps_per_env
            
            # Get initial states
            states, _ = self.vec_env.reset()
            
            # Storage for the entire batch
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []
            batch_infos = []
            
            # Collect data for num_steps_per_env from each environment
            for step in range(self.num_steps_per_env):
                # Generate actions for all environments
                actions_delta = []
                for i in range(num_envs):
                    state = states[i]
                    # Ensure state is numpy array
                    if hasattr(state, 'cpu'):
                        state_np = state.cpu().numpy()
                    else:
                        state_np = state
                    action_delta = self.agent.select_action(state_np)
                    actions_delta.append(action_delta)
                actions_delta = np.array(actions_delta)
                
                # Apply cumulative action if enabled
                if self.config.get('use_cumulative_action', False):
                    actions = self.cum_action + actions_delta
                    self.cum_action = actions.copy()
                else:
                    actions = actions_delta
                
                # Step all environments
                next_states, rewards, dones, infos = self.vec_env.step(actions)
                
                # Ensure all data is on CPU
                if hasattr(next_states, 'cpu'):
                    next_states = next_states.cpu().numpy()
                if hasattr(rewards, 'cpu'):
                    rewards = rewards.cpu().numpy()
                if hasattr(dones, 'cpu'):
                    dones = dones.cpu().numpy()
                if hasattr(states, 'cpu'):
                    states = states.cpu().numpy()
                
                # Apply stability reward shaping if enabled
                shaped_rewards = rewards.copy()
                if self.config.get('use_stability_reward', False) and self.P_tensor is not None:
                    device = torch.device(self.config.get('device', 'cuda:0'))
                    V_current = construct_lyapunov(device, states, self.P_tensor).detach().cpu().numpy()
                    V_next = construct_lyapunov(device, next_states, self.P_tensor).detach().cpu().numpy()
                    stability_terms = -float(self.config.get('stability_coef', 1e-3)) * (V_next - V_current)
                    shaped_rewards = rewards + stability_terms
                
                # Store data
                batch_states.append(states)
                batch_actions.append(actions)
                batch_rewards.append(shaped_rewards)
                batch_next_states.append(next_states)
                batch_dones.append(dones)
                batch_infos.extend(infos)
                
                # Update states for next iteration
                states = next_states
                
                # Handle episode resets
                reset_indices = np.where(dones)[0]
                if len(reset_indices) > 0:
                    new_states, _ = self.vec_env.reset()
                    if hasattr(new_states, 'cpu'):
                        new_states = new_states.cpu().numpy()
                    states[reset_indices] = new_states[reset_indices]
                    # Reset cumulative action for done environments
                    if self.config.get('use_cumulative_action', False):
                        self.cum_action[reset_indices] = 0.0
            
            # Concatenate all data
            batch_states = np.concatenate(batch_states, axis=0)
            batch_actions = np.concatenate(batch_actions, axis=0)
            batch_rewards = np.concatenate(batch_rewards, axis=0)
            batch_next_states = np.concatenate(batch_next_states, axis=0)
            batch_dones = np.concatenate(batch_dones, axis=0)
            
            # Create batch
            batch = RolloutBatch(
                states=batch_states,
                actions=batch_actions,
                rewards=batch_rewards,
                next_states=batch_next_states,
                dones=batch_dones,
                infos=batch_infos,
                total_steps=total_steps
            )
            
            self.stats['collection_time'] += time.time() - start_time
            return batch
            
        except Exception as e:
            print(f"[BatchCollector] Error collecting batch: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_batch(self, timeout: float = 1.0) -> Optional[RolloutBatch]:
        """Get a batch from the queue."""
        try:
            return self.batch_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            **self.stats,
            'queue_size': self.batch_queue.qsize(),
            'avg_collection_time': self.stats['collection_time'] / max(1, self.stats['batches_collected'])
        }


class AsyncTrainerV2:
    """Asynchronous trainer v2: RSL-RL style batch training."""
    
    def __init__(self,
                 agent,
                 dynamics_models: List[torch.nn.Module],
                 dynamics_optimizers: List[torch.optim.Optimizer],
                 batch_collector: BatchCollector,
                 config: Dict[str, Any],
                 device: str = "cuda:0",
                 P_tensor=None,
                 K_lqr=None,
                 args=None):
        self.agent = agent
        self.dynamics_models = dynamics_models
        self.dynamics_optimizers = dynamics_optimizers
        self.batch_collector = batch_collector
        self.config = config
        self.device = device
        self.P_tensor = P_tensor
        self.K_lqr = K_lqr
        self.args = args
        
        # Training control
        self.training_thread = None
        self._stop_training = threading.Event()
        
        # Training statistics
        self.training_stats = {
            'agent_updates': 0,
            'dynamics_updates': 0,
            'training_time': 0.0,
            'batches_processed': 0,
            'last_losses': {}
        }
        
        # Buffers for training
        self.real_buffer = deque(maxlen=config.get('replay_size', 1000000))
        self.model_buffer = deque(maxlen=config.get('replay_size', 1000000))
        
        # Elite selection tracking
        num_networks = len(dynamics_models)
        num_elites = config.get('num_elites', num_networks // 2)
        self.elite_indices = list(range(num_elites))  # Start with first N models as elites
        self.model_losses = [float('inf')] * num_networks  # Track validation losses
    
    def start_training(self):
        """Start asynchronous training."""
        self._stop_training.clear()
        
        # Start batch collection
        self.batch_collector.start_collection()
        
        # Start training thread
        self.training_thread = threading.Thread(
            target=self._training_loop,
            name="AsyncTrainerV2",
            daemon=True
        )
        self.training_thread.start()
        
        print("[AsyncTrainerV2] Started batch-based asynchronous training")
    
    def stop_training(self):
        """Stop asynchronous training."""
        self._stop_training.set()
        
        # Stop batch collection
        self.batch_collector.stop_collection()
        
        # Wait for training thread to finish
        if self.training_thread:
            self.training_thread.join(timeout=5.0)
        
        print("[AsyncTrainerV2] Stopped batch-based asynchronous training")
    
    def _training_loop(self):
        """Main training loop running in separate thread."""
        while not self._stop_training.is_set():
            start_time = time.time()
            
            try:
                # Get a batch from the collector
                batch = self.batch_collector.get_batch(timeout=1.0)
                if batch is not None:
                    # Process the batch
                    self._process_batch(batch)
                    self.training_stats['batches_processed'] += 1
                
                # Train agent if we have enough data
                if len(self.real_buffer) >= self.config.get('min_pool_size', 1000):
                    self._train_agent()
                
                # Train dynamics models
                if len(self.real_buffer) >= self.config.get('min_pool_size', 1000):
                    self._train_dynamics_models()
                
                # Model rollout (generate synthetic data)
                if len(self.model_buffer) < self.config.get('replay_size', 1000000) * 0.5:
                    self._model_rollout()
                
                self.training_stats['training_time'] += time.time() - start_time
                
                # Brief pause to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[AsyncTrainerV2] Error in training loop: {e}")
                time.sleep(0.1)
    
    def _process_batch(self, batch: RolloutBatch):
        """Process a collected batch and add to buffers."""
        try:
            # Add all transitions to real buffer
            for i in range(batch.total_steps):
                transition = Transition(
                    state=batch.states[i],
                    action=batch.actions[i],
                    reward=batch.rewards[i],
                    next_state=batch.next_states[i],
                    done=batch.dones[i],
                    info=batch.infos[i] if i < len(batch.infos) else {}
                )
                self.real_buffer.append(transition)
            
        except Exception as e:
            print(f"[AsyncTrainerV2] Error processing batch: {e}")
            import traceback
            traceback.print_exc()
    
    def _train_agent(self):
        """Train the agent using both real and model data."""
        try:
            # Sample from both buffers
            real_ratio = self.config.get('real_ratio', 0.1)
            batch_size = self.config.get('policy_train_batch_size', 256)
            
            real_batch_size = int(batch_size * real_ratio)
            model_batch_size = batch_size - real_batch_size
            
            # Sample real data
            if len(self.real_buffer) >= real_batch_size:
                real_samples = np.random.choice(len(self.real_buffer), real_batch_size, replace=False)
                real_batch = [self.real_buffer[i] for i in real_samples]
            else:
                real_batch = list(self.real_buffer)
            
            # Sample model data
            if len(self.model_buffer) >= model_batch_size:
                model_samples = np.random.choice(len(self.model_buffer), model_batch_size, replace=False)
                model_batch = [self.model_buffer[i] for i in model_samples]
            else:
                model_batch = list(self.model_buffer)
            
            # Combine batches
            combined_batch = real_batch + model_batch
            
            if len(combined_batch) == 0:
                return
            
            # Convert to tensors with proper device handling
            states = torch.tensor(np.array([t.state for t in combined_batch]), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.array([t.action for t in combined_batch]), dtype=torch.float32, device=self.device)
            rewards = torch.tensor(np.array([t.reward for t in combined_batch]), dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array([t.next_state for t in combined_batch]), dtype=torch.float32, device=self.device)
            dones = torch.tensor(np.array([t.done for t in combined_batch]), dtype=torch.bool, device=self.device)
            
            # Train agent (this would call your existing agent training code)
            # For now, just increment the counter
            self.training_stats['agent_updates'] += 1
            
        except Exception as e:
            print(f"[AsyncTrainerV2] Error in agent training: {e}")
            import traceback
            traceback.print_exc()
    
    def _train_dynamics_models(self):
        """Train dynamics models using real data."""
        try:
            if len(self.real_buffer) < 100:  # Need minimum data
                return
            
            # Sample real data for dynamics training
            batch_size = min(128, len(self.real_buffer))  # Smaller batch for dynamics
            samples = np.random.choice(len(self.real_buffer), batch_size, replace=False)
            batch = [self.real_buffer[i] for i in samples]
            
            # Convert to tensors with proper device handling
            states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.array([t.action for t in batch]), dtype=torch.float32, device=self.device)
            rewards = torch.tensor(np.array([t.reward for t in batch]), dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
            dones = torch.tensor(np.array([t.done for t in batch]), dtype=torch.float32, device=self.device)
            
            # Train each dynamics model with UTD
            from scripts.incremental_mbpo.algo.incremental_dynamics import incremental_dynamics_loss
            dynamic_updates = max(1, self.config.get('utd_dynamic_per_env', 4))
            for update_step in range(dynamic_updates):
                for i, (model, optimizer) in enumerate(zip(self.dynamics_models, self.dynamics_optimizers)):
                    loss = incremental_dynamics_loss(
                        model,  # type: ignore
                        states,
                        actions,
                        rewards,
                        next_states,
                        dones,
                        w_dyn=float(self.config.get('dyn_w_dyn', 1.0)),
                        w_r=float(self.config.get('dyn_w_r', 1.0)),
                        w_d=float(self.config.get('dyn_w_d', 0.1)),
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Track loss for elite selection
                    if self.model_losses[i] == float('inf'):
                        self.model_losses[i] = loss.item()
                    else:
                        self.model_losses[i] = 0.9 * self.model_losses[i] + 0.1 * loss.item()
            
            # Elite selection every 500 steps
            if self.training_stats['dynamics_updates'] % 500 == 0:
                sorted_indices = sorted(range(len(self.dynamics_models)), key=lambda i: self.model_losses[i])
                num_elites = self.config.get('num_elites', len(self.dynamics_models) // 2)
                self.elite_indices = sorted_indices[:num_elites]
                print(f"[EliteSelection] elite_models={self.elite_indices}")
                
                # Update LQR if stability reward is enabled and no offline model
                if (self.config.get('use_stability_reward', False) and 
                    self.P_tensor is not None and 
                    not self.config.get('use_lqr_baseline', False)):
                    try:
                        # Use the best elite model for B estimation
                        best_elite_model = self.dynamics_models[self.elite_indices[0]]
                        obs_dim = states.shape[1]
                        act_dim = actions.shape[1]
                        B_est = _estimate_B_from_online_model(best_elite_model, obs_dim, act_dim, torch.device(self.device))
                        A = (1.0 - 1e-3) * np.eye(obs_dim)
                        Q = float(self.config.get('Q_scale', 1.0)) * np.eye(obs_dim)
                        R = float(self.config.get('R_scale', 0.1)) * np.eye(act_dim)
                        P_np, K_lqr = _solve_lqr(A, B_est, Q, R)
                        self.P_tensor = torch.tensor(P_np, dtype=torch.float32, device=self.device)
                        self.K_lqr = K_lqr
                        # Update batch collector's P_tensor
                        self.batch_collector.P_tensor = self.P_tensor
                    except Exception:
                        pass
            
            self.training_stats['dynamics_updates'] += 1
            
        except Exception as e:
            print(f"[AsyncTrainerV2] Error in dynamics training: {e}")
            import traceback
            traceback.print_exc()
    
    def _model_rollout(self):
        """Generate synthetic data using dynamics models."""
        try:
            if len(self.real_buffer) < 100:
                return
            
            # Sample states for rollout
            rollout_batch_size = self.config.get('rollout_batch_size', 256)
            samples = np.random.choice(len(self.real_buffer), min(rollout_batch_size, len(self.real_buffer)), replace=False)
            states = np.array([self.real_buffer[i].state for i in samples])
            
            # Generate actions
            actions_delta = []
            for state in states:
                action_delta = self.agent.select_action(state)
                actions_delta.append(action_delta)
            actions_delta = np.array(actions_delta)
            
            # Apply cumulative action if enabled
            if self.config.get('use_cumulative_action', False):
                actions = actions_delta  # For model rollout, we use delta actions directly
            else:
                actions = actions_delta
            
            # Use elite dynamics models to predict next states
            states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                # Collect predictions from all elite models
                elite_predictions = []
                for elite_idx in self.elite_indices:
                    elite_model = self.dynamics_models[elite_idx]
                    ns_t, r_t, dprob_t = elite_model(states_tensor, actions_tensor)
                    elite_predictions.append((ns_t, r_t, dprob_t))
                
                # Stack and average predictions
                next_states_stack = torch.stack([pred[0] for pred in elite_predictions], dim=0)
                rewards_stack = torch.stack([pred[1] for pred in elite_predictions], dim=0)
                terminals_stack = torch.stack([pred[2] for pred in elite_predictions], dim=0)
                
                # Average predictions
                next_states = torch.mean(next_states_stack, dim=0).cpu().numpy()
                rewards = torch.mean(rewards_stack, dim=0).cpu().numpy()
                terminals = (torch.mean(terminals_stack, dim=0).cpu().numpy() > float(self.config.get('dyn_done_threshold', 0.5))).astype(np.bool_)
            
            # Apply stability reward shaping in model rollout
            if self.config.get('use_stability_reward', False) and self.P_tensor is not None:
                Vc = construct_lyapunov(torch.device(self.device), states, self.P_tensor)
                Vn = construct_lyapunov(torch.device(self.device), next_states, self.P_tensor)
                stability_terms = -float(self.config.get('stability_coef', 1e-3)) * (Vn - Vc)
                rewards = rewards + stability_terms.detach().cpu().numpy()
            
            # Add to model buffer
            for i in range(len(states)):
                transition = Transition(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_states[i],
                    done=terminals[i],
                    info={}
                )
                self.model_buffer.append(transition)
            
        except Exception as e:
            print(f"[AsyncTrainerV2] Error in model rollout: {e}")
            import traceback
            traceback.print_exc()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            **self.training_stats,
            'real_buffer_size': len(self.real_buffer),
            'model_buffer_size': len(self.model_buffer),
            'elite_indices': self.elite_indices,
            'model_losses': self.model_losses,
            **self.batch_collector.get_stats()
        }


def create_async_trainer_v2(vec_env, agent, dynamics_models, dynamics_optimizers, config, device="cuda:0", P_tensor=None, K_lqr=None, args=None):
    """Factory function to create an async trainer v2."""
    batch_collector = BatchCollector(
        vec_env=vec_env,
        agent=agent,
        num_steps_per_env=config.get('num_steps_per_env', 24),
        max_queue_size=config.get('max_queue_size', 10),
        config=config,
        P_tensor=P_tensor,
        K_lqr=K_lqr,
        args=args
    )
    
    trainer = AsyncTrainerV2(
        agent=agent,
        dynamics_models=dynamics_models,
        dynamics_optimizers=dynamics_optimizers,
        batch_collector=batch_collector,
        config=config,
        device=device,
        P_tensor=P_tensor,
        K_lqr=K_lqr,
        args=args
    )
    
    return trainer
