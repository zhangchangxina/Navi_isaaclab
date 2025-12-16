"""
Asynchronous training framework for Isaac Lab environments.
Separates environment interaction from network training to maximize parallel efficiency.
"""

import threading
import queue
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque


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
class BatchTransition:
    """Batch transition data structure."""
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, Any]]


class AsyncDataCollector:
    """Asynchronous data collector that runs environment interaction in separate threads."""
    
    def __init__(self, 
                 vec_env,
                 agent,
                 num_collectors: int = 4,
                 max_queue_size: int = 1000,
                 batch_size: int = 256):
        self.vec_env = vec_env
        self.agent = agent
        self.num_collectors = num_collectors
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        
        # Data queues for different types of transitions
        self.real_data_queue = queue.Queue(maxsize=max_queue_size)
        self.model_data_queue = queue.Queue(maxsize=max_queue_size)
        
        # Threading control
        self.collectors = []
        self.stop_event = threading.Event()
        self.collected_transitions = 0
        
        # Statistics
        self.stats = {
            'real_data_collected': 0,
            'model_data_collected': 0,
            'queue_full_count': 0,
            'collection_time': 0.0
        }
    
    def start_collection(self):
        """Start asynchronous data collection."""
        self.stop_event.clear()
        
        # Start real environment collectors
        for i in range(self.num_collectors):
            collector = threading.Thread(
                target=self._collect_real_data,
                name=f"RealCollector-{i}",
                daemon=True
            )
            collector.start()
            self.collectors.append(collector)
        
        print(f"[AsyncCollector] Started {self.num_collectors} data collection threads")
    
    def stop_collection(self):
        """Stop asynchronous data collection."""
        self.stop_event.set()
        for collector in self.collectors:
            collector.join(timeout=1.0)
        self.collectors.clear()
        print("[AsyncCollector] Stopped data collection threads")
    
    def _collect_real_data(self):
        """Collect real environment data in a separate thread."""
        # Create a separate environment instance for this thread
        # This ensures thread safety
        thread_env = self.vec_env  # In practice, you might want to clone the env
        
        while not self.stop_event.is_set():
            try:
                # Collect a batch of real data
                batch_transitions = self._collect_batch(thread_env, real_data=True)
                
                if batch_transitions is not None:
                    # Try to put data in queue (non-blocking)
                    try:
                        self.real_data_queue.put_nowait(batch_transitions)
                        self.stats['real_data_collected'] += len(batch_transitions.states)
                    except queue.Full:
                        self.stats['queue_full_count'] += 1
                        # Drop oldest data if queue is full
                        try:
                            self.real_data_queue.get_nowait()
                            self.real_data_queue.put_nowait(batch_transitions)
                        except queue.Empty:
                            pass
                
            except Exception as e:
                print(f"[AsyncCollector] Error in real data collection: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def _collect_batch(self, env, real_data: bool = True) -> Optional[BatchTransition]:
        """Collect a batch of transitions from environment."""
        start_time = time.time()
        
        try:
            # Get current states
            if not hasattr(env, 'states'):
                states, _ = env.reset()
            else:
                states = env.states
            
            # Ensure states are on CPU for action selection
            if hasattr(states, 'cpu'):
                states_cpu = states.cpu().numpy()
            else:
                states_cpu = states
            
            # Generate actions for all environments (on CPU to avoid device conflicts)
            actions = []
            for state in states_cpu:
                # Ensure state is numpy array
                if hasattr(state, 'cpu'):
                    state_np = state.cpu().numpy()
                else:
                    state_np = state
                action = self.agent.select_action(state_np)
                actions.append(action)
            actions = np.array(actions)
            
            # Step all environments
            next_states, rewards, dones, infos = env.step(actions)
            
            # Ensure all data is on CPU
            if hasattr(next_states, 'cpu'):
                next_states = next_states.cpu().numpy()
            if hasattr(rewards, 'cpu'):
                rewards = rewards.cpu().numpy()
            if hasattr(dones, 'cpu'):
                dones = dones.cpu().numpy()
            
            # Create batch transition
            batch_transition = BatchTransition(
                states=states_cpu,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                infos=infos
            )
            
            # Update environment states
            env.states = next_states
            
            # Handle episode resets
            reset_indices = np.where(dones)[0]
            if len(reset_indices) > 0:
                new_states, _ = env.reset()
                if hasattr(new_states, 'cpu'):
                    new_states = new_states.cpu().numpy()
                env.states[reset_indices] = new_states[reset_indices]
            
            self.stats['collection_time'] += time.time() - start_time
            return batch_transition
            
        except Exception as e:
            print(f"[AsyncCollector] Error collecting batch: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_real_data_batch(self, timeout: float = 0.1) -> Optional[BatchTransition]:
        """Get a batch of real data from the queue."""
        try:
            return self.real_data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_model_data_batch(self, timeout: float = 0.1) -> Optional[BatchTransition]:
        """Get a batch of model data from the queue."""
        try:
            return self.model_data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def add_model_data(self, batch_transition: BatchTransition):
        """Add model-generated data to the queue."""
        try:
            self.model_data_queue.put_nowait(batch_transition)
            self.stats['model_data_collected'] += len(batch_transition.states)
        except queue.Full:
            # Drop oldest data if queue is full
            try:
                self.model_data_queue.get_nowait()
                self.model_data_queue.put_nowait(batch_transition)
            except queue.Empty:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            **self.stats,
            'real_queue_size': self.real_data_queue.qsize(),
            'model_queue_size': self.model_data_queue.qsize(),
            'total_collected': self.stats['real_data_collected'] + self.stats['model_data_collected']
        }


class AsyncTrainer:
    """Asynchronous trainer that separates data collection from network training."""
    
    def __init__(self,
                 agent,
                 dynamics_models: List[torch.nn.Module],
                 dynamics_optimizers: List[torch.optim.Optimizer],
                 data_collector: AsyncDataCollector,
                 config: Dict[str, Any],
                 device: str = "cuda:0"):
        self.agent = agent
        self.dynamics_models = dynamics_models
        self.dynamics_optimizers = dynamics_optimizers
        self.data_collector = data_collector
        self.config = config
        self.device = device
        
        # Training control
        self.training_thread = None
        self._stop_training = threading.Event()
        
        # Training statistics
        self.training_stats = {
            'agent_updates': 0,
            'dynamics_updates': 0,
            'training_time': 0.0,
            'last_losses': {}
        }
        
        # Buffers for training
        self.real_buffer = deque(maxlen=config.get('replay_size', 1000000))
        self.model_buffer = deque(maxlen=config.get('replay_size', 1000000))
    
    def start_training(self):
        """Start asynchronous training."""
        self._stop_training.clear()
        
        # Start data collection
        self.data_collector.start_collection()
        
        # Start training thread
        self.training_thread = threading.Thread(
            target=self._training_loop,
            name="AsyncTrainer",
            daemon=True
        )
        self.training_thread.start()
        
        print("[AsyncTrainer] Started asynchronous training")
    
    def stop_training(self):
        """Stop asynchronous training."""
        self._stop_training.set()
        
        # Stop data collection
        self.data_collector.stop_collection()
        
        # Wait for training thread to finish
        if self.training_thread:
            self.training_thread.join(timeout=5.0)
        
        print("[AsyncTrainer] Stopped asynchronous training")
    
    def _training_loop(self):
        """Main training loop running in separate thread."""
        while not self._stop_training.is_set():
            start_time = time.time()
            
            try:
                # Collect data from queues
                self._collect_data_from_queues()
                
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
                time.sleep(0.001)
                
            except Exception as e:
                print(f"[AsyncTrainer] Error in training loop: {e}")
                time.sleep(0.1)
    
    def _collect_data_from_queues(self):
        """Collect data from async queues and add to buffers."""
        # Collect real data
        while True:
            batch = self.data_collector.get_real_data_batch(timeout=0.01)
            if batch is None:
                break
            
            # Add to real buffer
            for i in range(len(batch.states)):
                transition = Transition(
                    state=batch.states[i],
                    action=batch.actions[i],
                    reward=batch.rewards[i],
                    next_state=batch.next_states[i],
                    done=batch.dones[i],
                    info=batch.infos[i]
                )
                self.real_buffer.append(transition)
        
        # Collect model data
        while True:
            batch = self.data_collector.get_model_data_batch(timeout=0.01)
            if batch is None:
                break
            
            # Add to model buffer
            for i in range(len(batch.states)):
                transition = Transition(
                    state=batch.states[i],
                    action=batch.actions[i],
                    reward=batch.rewards[i],
                    next_state=batch.next_states[i],
                    done=batch.dones[i],
                    info=batch.infos[i]
                )
                self.model_buffer.append(transition)
    
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
            print(f"[AsyncTrainer] Error in agent training: {e}")
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
            
            # Train each dynamics model
            for model, optimizer in zip(self.dynamics_models, self.dynamics_optimizers):
                # This would call your existing dynamics training code
                # For now, just increment the counter
                pass
            
            self.training_stats['dynamics_updates'] += 1
            
        except Exception as e:
            print(f"[AsyncTrainer] Error in dynamics training: {e}")
            import traceback
            traceback.print_exc()
    
    def _model_rollout(self):
        """Generate synthetic data using dynamics models."""
        if len(self.real_buffer) < 100:
            return
        
        # Sample states for rollout
        rollout_batch_size = self.config.get('rollout_batch_size', 256)
        samples = np.random.choice(len(self.real_buffer), min(rollout_batch_size, len(self.real_buffer)), replace=False)
        states = np.array([self.real_buffer[i].state for i in samples])
        
        # Generate actions
        actions = []
        for state in states:
            action = self.agent.select_action(state)
            actions.append(action)
        actions = np.array(actions)
        
        # Use dynamics models to predict next states
        # This would call your existing model rollout code
        # For now, just create dummy data
        next_states = states + np.random.normal(0, 0.01, states.shape)
        rewards = np.random.normal(0, 1, len(states))
        dones = np.random.random(len(states)) < 0.1
        
        # Create batch and add to model data queue
        batch_transition = BatchTransition(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            infos=[{}] * len(states)
        )
        
        self.data_collector.add_model_data(batch_transition)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            **self.training_stats,
            'real_buffer_size': len(self.real_buffer),
            'model_buffer_size': len(self.model_buffer),
            **self.data_collector.get_stats()
        }


def create_async_trainer(vec_env, agent, dynamics_models, dynamics_optimizers, config, device="cuda:0"):
    """Factory function to create an async trainer."""
    data_collector = AsyncDataCollector(
        vec_env=vec_env,
        agent=agent,
        num_collectors=config.get('num_collectors', 4),
        max_queue_size=config.get('max_queue_size', 1000),
        batch_size=config.get('collection_batch_size', 256)
    )
    
    trainer = AsyncTrainer(
        agent=agent,
        dynamics_models=dynamics_models,
        dynamics_optimizers=dynamics_optimizers,
        data_collector=data_collector,
        config=config,
        device=device
    )
    
    return trainer
