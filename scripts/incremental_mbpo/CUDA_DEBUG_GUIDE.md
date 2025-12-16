# CUDA Debug Guide for Async Training

## Problem Description
You're encountering CUDA device-side assertion errors:
```
CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call
```

## Root Causes

### 1. Device Mismatch
- Isaac environment uses one device (e.g., cuda:3)
- Training tensors use different device
- Data transfer between devices causes conflicts

### 2. Thread Safety Issues
- Multiple threads accessing CUDA simultaneously
- Race conditions in device memory access
- Improper synchronization between threads

### 3. Memory Access Problems
- Accessing tensor data on wrong device
- Invalid memory addresses
- Buffer overflow in CUDA kernels

## Debugging Steps

### Step 1: Enable CUDA Debugging
```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
```

### Step 2: Run Simple Test
```bash
./run_async_test.sh
```

This will test basic components without full MBPO complexity.

### Step 3: Check Device Consistency
```python
# In your code, ensure all tensors are on the same device
print(f"Environment device: {env.device}")
print(f"Agent device: {agent.device}")
print(f"Tensor device: {tensor.device}")
```

### Step 4: Verify Data Types
```python
# Ensure consistent data types
print(f"State dtype: {state.dtype}")
print(f"Action dtype: {action.dtype}")
print(f"Reward dtype: {reward.dtype}")
```

## Fixes Applied

### 1. Device Safety in Data Collection
```python
# Ensure states are on CPU for action selection
if hasattr(states, 'cpu'):
    states_cpu = states.cpu().numpy()
else:
    states_cpu = states

# Generate actions on CPU to avoid device conflicts
actions = []
for state in states_cpu:
    if hasattr(state, 'cpu'):
        state_np = state.cpu().numpy()
    else:
        state_np = state
    action = self.agent.select_action(state_np)
    actions.append(action)
```

### 2. Proper Error Handling
```python
try:
    # CUDA operations
    pass
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

### 3. Thread Safety
- All data collection happens on CPU
- Only training tensors are moved to GPU
- Proper synchronization between threads

## Testing Strategy

### 1. Simple Test First
Run `test_async_simple.py` to test basic components:
- Environment creation
- Basic interaction
- Agent action selection
- Async data collection

### 2. Gradual Complexity
If simple test passes, gradually add complexity:
- Add dynamics models
- Add model rollout
- Add full MBPO training

### 3. Monitor Resources
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor CUDA errors
export CUDA_LAUNCH_BLOCKING=1
```

## Common Solutions

### Solution 1: Force CPU for Data Collection
```python
# In async_trainer.py
def _collect_batch(self, env, real_data: bool = True):
    # Always work with CPU data during collection
    states_cpu = states.cpu().numpy() if hasattr(states, 'cpu') else states
    # ... rest of collection logic
```

### Solution 2: Device Consistency Check
```python
def ensure_device_consistency(self):
    """Ensure all components use the same device."""
    assert self.vec_env.device == self.device
    assert self.agent.device == self.device
    # Check all models are on correct device
    for model in self.dynamics_models:
        assert next(model.parameters()).device == torch.device(self.device)
```

### Solution 3: Reduce Parallelism
```python
# Start with fewer collectors
config = {
    'num_collectors': 1,  # Start with 1
    'max_queue_size': 100,  # Smaller queue
    'collection_batch_size': 32  # Smaller batch
}
```

## Prevention

### 1. Always Check Device
```python
def safe_tensor_creation(data, device):
    """Safely create tensor on specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return torch.tensor(data, device=device)
```

### 2. Use Context Managers
```python
with torch.cuda.device(device):
    # All CUDA operations in this context
    pass
```

### 3. Proper Cleanup
```python
def cleanup():
    """Clean up CUDA resources."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

## Running Tests

### Test 1: Basic Functionality
```bash
./run_async_test.sh
```

### Test 2: Debug Mode
```bash
./run_async_mbpo_debug.sh
```

### Test 3: Full Training (when ready)
```bash
./run_async_mbpo.sh
```

## Expected Output

### Successful Test
```
[Test] Starting simple async test with device: cuda:3
[Test] Environment created: obs_dim=24, act_dim=4
[Test 1] Basic environment interaction...
  Step 0: obs shape=(4, 24), rewards shape=(4,)
[Test 2] Agent action selection...
  Action 0: shape=(4,), type=<class 'numpy.ndarray'>
[Test 3] Async data collector...
  Starting data collection...
  Collected batch 0: states shape=(4, 24)
[Test] All tests completed successfully!
```

### Error Output
```
[Test 1] Error: CUDA error: device-side assert triggered
Traceback (most recent call last):
  File "test_async_simple.py", line 85, in main
    obs, rewards, dones, infos = vec_env.step(actions)
```

## Next Steps

1. Run the simple test first
2. If it fails, check device configuration
3. If it passes, gradually increase complexity
4. Monitor GPU memory usage
5. Check for memory leaks

## Contact

If issues persist, provide:
- Full error traceback
- GPU information (`nvidia-smi`)
- Environment configuration
- Test results from simple test
