# Source: https://github.com/srush/GPU-Puzzles/blob/b3c4b237d7f0dc6d82055b753c8ea6e0cbb845eb/GPU_puzzlers.py
# Copyright (c) 2022 Sasha Rush, MIT License. Puzzles 1–4, copied verbatim.

# # GPU Puzzles
# - by [Sasha Rush](http://rush-nlp.com) - [srush_nlp](https://twitter.com/srush_nlp)


# ![](https://github.com/srush/GPU-Puzzles/raw/main/cuda.png)

# GPU architectures are critical to machine learning, and seem to be
# becoming even more important every day. However, you can be an expert
# in machine learning without ever touching GPU code. It is hard to gain
# intuition working through abstractions. 

# This notebook is an attempt to teach beginner GPU programming in a
# completely interactive fashion. Instead of providing text with
# concepts, it throws you right into coding and building GPU
# kernels. The exercises use NUMBA which directly maps Python
# code to CUDA kernels. It looks like Python but is basically
# identical to writing low-level CUDA code. 
# In a few hours, I think you can go from basics to
# understanding the real algorithms that power 99% of deep learning
# today. If you do want to read the manual, it is here:

# [NUMBA CUDA Guide](https://numba.readthedocs.io/en/stable/cuda/index.html)

# I recommend doing these in Colab, as it is easy to get started.  Be
# sure to make your own copy, turn on GPU mode in the settings (`Runtime / Change runtime type`, then set `Hardware accelerator` to `GPU`), and
# then get to coding.

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/GPU-Puzzles/blob/main/GPU_puzzlers.ipynb)

# (If you are into this style of puzzle, also check out my [Tensor
# Puzzles](https://github.com/srush/Tensor-Puzzles) for PyTorch.)

!pip install -qqq git+https://github.com/danoneata/chalk@srush-patch-1
!wget -q https://github.com/srush/GPU-Puzzles/raw/main/robot.png https://github.com/srush/GPU-Puzzles/raw/main/lib.py


import numba
import numpy as np
import warnings
from lib import CudaProblem, Coord

warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)


# ## Puzzle 1: Map
#
# Implement a "kernel" (GPU function) that adds 10 to each position of vector `a`
# and stores it in vector `out`.  You have 1 thread per position.


# **Warning** This code looks like Python but it is really CUDA! You cannot use
# standard python tools like list comprehensions or ask for Numpy properties
# like shape or size (if you need the size, it is given as an argument).
# The puzzles only require doing simple operations, basically
# +, *, simple array indexing, for loops, and if statements.
# You are allowed to use local variables. 
# If you get an
# error it is probably because you did something fancy :). 


# *Tip: Think of the function `call` as being run 1 time for each thread.
# The only difference is that `cuda.threadIdx.x` changes each time.*

# +
def map_spec(a):
    return a + 10


def map_test(cuda):
    def call(out, a) -> None:
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 1 lines)

    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "Map", map_test, [a], out, threadsperblock=Coord(SIZE, 1), spec=map_spec
)
problem.show()

# +
problem.check()
# -

# ## Puzzle 2 - Zip
#
# Implement a kernel that adds together each position of `a` and `b` and stores it in `out`.
# You have 1 thread per position.

# +
def zip_spec(a, b):
    return a + b


def zip_test(cuda):
    def call(out, a, b) -> None:
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 1 lines)

    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
b = np.arange(SIZE)
problem = CudaProblem(
    "Zip", zip_test, [a, b], out, threadsperblock=Coord(SIZE, 1), spec=zip_spec
)
problem.show()
# +

# +
problem.check()
# -

# ## Puzzle 3 - Guards
#
# Implement a kernel that adds 10 to each position of `a` and stores it in `out`.
# You have more threads than positions.

# +
def map_guard_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 2 lines)

    return call


SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem(
    "Guard",
    map_guard_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(8, 1),
    spec=map_spec,
)
problem.show()

# +
problem.check()
# -

# ## Puzzle 4 - Map 2D
#
# Implement a kernel that adds 10 to each position of `a` and stores it in `out`.
# Input `a` is 2D and square. You have more threads than positions.

# +
def map_2D_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        # FILL ME IN (roughly 2 lines)

    return call


SIZE = 2
out = np.zeros((SIZE, SIZE))
a = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
problem = CudaProblem(
    "Map 2D", map_2D_test, [a], out, [SIZE], threadsperblock=Coord(3, 3), spec=map_spec
)
problem.show()

# +
problem.check()
# -
