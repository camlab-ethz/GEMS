import jax
import jax.numpy as jnp
from jax import random
import time

# Set up random key
key = random.PRNGKey(0)

# Check JAX backend
backend = jax.lib.xla_bridge.get_backend().platform
print("JAX backend:", backend)

# Initialize array
x = jnp.array([[0., 2., 4.]])
print("Array x:\n", x)
print("Shape of x:", x.shape)
print("Matrix multiplication x @ x.T:\n", x @ x.T)
print("Element-wise multiplication x * x.T:\n", x * x.T)

# Define a function to test
def fn(x):
    return x + x*x + x*x*x

# JIT compile the function
jit_fn = jax.jit(fn)

# Generate a large random array
x_large = jax.random.normal(key, (10000, 10000))

# Function to time execution and print results
def time_execution(function, x, label):
    start_time = time.time()
    result = function(x).block_until_ready()
    elapsed_time = time.time() - start_time
    print(f"{label} execution time: {elapsed_time:.6f} seconds")
    return elapsed_time

# Run and time the non-JIT function
print("Timing non-JIT function:")
non_jit_time = time_execution(fn, x_large, "Non-JIT")

# Run and time the JIT function
print("Timing JIT function:")
jit_time = time_execution(jit_fn, x_large, "JIT")

# Compare execution times
print(f"\nNon-JIT function time: {non_jit_time:.6f} seconds")
print(f"JIT function time: {jit_time:.6f} seconds")

if backend == 'gpu':
    print("Running on GPU. You should see significant speedup with JIT compilation.")
else:
    print("Running on CPU. Expect longer execution times without GPU acceleration.")