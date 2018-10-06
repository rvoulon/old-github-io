# An Even Easier Introduction to CUDA
https://devblogs.nvidia.com/even-easier-introduction-cuda/  
Adapted for PyCUDA by [Roberta Voulon](https://github.com/rvoulon).  
I'm just learning this stuff as I go.


```python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
```


```python
# Just making sure everything is working as expected
print(f"{cuda.Device.count()} device(s) found")
for i in range(cuda.Device.count()):
    dev = cuda.Device(i)
    print(f"Device {i}: {dev.name()}")
    a, b = dev.compute_capability()
    print(f"  Compute capability: {a}.{b}")
    print(f"  Total memory: {dev.total_memory() / 1024} KB")
```

    1 device(s) found
    Device 0: Quadro P4000
      Compute capability: 6.1
      Total memory: 8308736.0 KB


## Starting from a regular loop
https://devblogs.nvidia.com/even-easier-introduction-cuda/  

Here's the starting code in C++, we'll need to port that to Python:

```c++
#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements

  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add(N, x, y);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] x;
  delete [] y;

  return 0;
}
```



### Here's that same code in Python:
(Yes I know it's not super pythonic, don't worry about that)


```python
def add(n, x, y):
    """Function to add the elements of two arrays"""
    for i in range(0, n):
        y[i] = x[i] + y[i]
```


```python
N = 1<<20 # 1M elements
x = np.ones(N, dtype=np.float32)
y = np.full(N, 2)
```


```python
start = time.time()
add(N, x, y)
print(f"---- {time.time() - start} seconds ----")
```

    ---- 3.84502911567688 seconds ----



```python
start = time.time()
max_error = 0.0
for i in range(0, N):
    global max_error
    max_error = max(max_error, (y[i] - 3.0))
print(f"Max error: {max_error}")
print(f"---- {time.time() - start} seconds ----")

x = []
y = []
```

    Max error: 0.0
    ---- 3.584669589996338 seconds ----


## Single thread on the GPU

### Now let's get this running on the GPU
1. set up your data (array/vector, matrix, ...) on the host, setting type to `np.float32`
1. allocate space on the GPU's memory and copy the data to it (to device)
1. write the key computational kernel in c for the GPU
1. get the function and call it, give as parameters the pointer(s) to your data on the GPU and the block size (making sure the data and blocksize have the same number of dimensions, 1D, 2D or 3D)
1. synchronize with the device: wait for GPU to finish before accessing on host
1. create a new variable to contain the data from the GPU and copy it (to host)
1. free up the memory on the device


```python
N = 1<<20 # 1M elements
x = np.ones(N, dtype=np.float32)
y = np.full(N, 2).astype(np.float32)
```


```python
x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)
```


```python
cuda.memcpy_htod(x_gpu, x)
cuda.memcpy_htod(y_gpu, y)
```


```python
# This loop won't be here for long, just stay with me
mod = SourceModule("""
    __global__ void add(float *x, float *y)
    {
        int n = 1<<20;
        for (int i = 0; i < n; i++)
            y[i] = x[i] + y[i];
    }
""")
```


```python
start = time.time()
func = mod.get_function("add")
# We're just using a single thread for now
func(x_gpu, y_gpu, block=(1, 1, 1))
y_added = np.empty_like(y)
cuda.memcpy_dtoh(y_added, y_gpu)
print(f"---- {time.time() - start} seconds ----")
```

    ---- 0.21840953826904297 seconds ----



```python
x_gpu.free()
y_gpu.free()
```


```python
start = time.time()
max_error = 0.0
for i in range(0, N):
    global max_error
    max_error = max(max_error, (y_added[i] - 3.0))
print(f"Max error: {max_error}")
print(f"---- {time.time() - start} seconds ----")
```

    Max error: 0.0
    ---- 3.9738917350769043 seconds ----


### ZOMG! It worked! 🙀🙀🙀
Even a single thread on the GPU is already much faster! Alright, now let's use parallel threading:

## Same example but parallelly on the GPU


```python
N = 1<<20 # 1M elements
x = np.ones(N, dtype=np.float32)
y = np.full(N, 2).astype(np.float32)
```


```python
x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)
```


```python
cuda.memcpy_htod(x_gpu, x)
cuda.memcpy_htod(y_gpu, y)
```


```python
# Let's fix that loop and make it parallellizable
mod = SourceModule("""
    __global__ void add(float *x, float *y)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        y[idx] = x[idx] + y[idx];
    }
""")
```


```python
start = time.time()
func = mod.get_function("add")
## Now using a block of 256 x 1 x 1 threads (1D)
func(x_gpu, y_gpu, block=(256, 1, 1))
y_added = np.empty_like(y)
cuda.memcpy_dtoh(y_added, y_gpu)
print(f"---- {time.time() - start} seconds ----")
```

    ---- 0.0012471675872802734 seconds ----


### 👆👆👆 Holy sh\*t, batman!


```python
x_gpu.free()
y_gpu.free()
```


```python
start = time.time()
max_error = 0.0
for i in range(0, N):
    global max_error
    max_error = max(max_error, (y_added[i] - 3.0))
print(f"Max error: {max_error}")
print(f"---- {time.time() - start} seconds ----")
```

    Max error: 0.0
    ---- 3.7731337547302246 seconds ----


## Next up?
- learn to use the pycuda profiler rather than using `start = time.time()`
- learn to use pycuda's `gpuarray`
