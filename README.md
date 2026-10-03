# SVD Data Compression

Exploring **Singular Value Decomposition (SVD)** as a tool for data compression, reconstruction, and low-rank representation, with a particular focus on **robotics and robotic perception**.

The project investigates how much information can be removed from images, audio, and video while retaining the information that matters.

---

## Overview

Singular Value Decomposition factors a matrix \(A\) into

$$
A = U\Sigma V^T
$$

where:

* \(U\) contains the left singular vectors
* \(\Sigma\) contains the singular values
* \(V^T\) contains the right singular vectors

By retaining only the largest \(k\) singular values, we can construct a low-rank approximation:

$$
A_k = U_k\Sigma_k V_k^T
$$

When \(k\) is much smaller than the original matrix dimensions, this can provide a compact representation of the original data.

This project explores that idea across several types of media.

---

## Goals

* Understand the mathematics behind SVD and low-rank approximations
* Implement and experiment with SVD-based compression
* Compress and reconstruct images
* Investigate SVD-based representations of audio
* Explore spatial and temporal redundancy in video
* Measure compression ratio and reconstruction error
* Compare different values of \(k\)
* Study the trade-off between compression, quality, and computation
* Investigate how these techniques relate to robotic perception and sensor data
* Eventually explore GPU-accelerated implementations

---

## Robotics Focus

Robots constantly generate large amounts of sensory data:

* Cameras produce images and video
* Microphones produce audio
* Sensor systems produce large numerical matrices
* Autonomous systems may need to transmit or store this data under bandwidth and storage constraints

This project asks a practical question:

> **How much can robotic sensor data be compressed while preserving the information needed by downstream systems?**

The goal is not simply to produce the smallest possible file.

A useful compression method must also consider:

$$
\text{Compression}
\leftrightarrow
\text{Quality}
\leftrightarrow
\text{Information}
\leftrightarrow
\text{Computation}
$$

For example, an image may look visually acceptable after compression while losing information important to a computer-vision system.

---

## Media

The repository contains a small collection of real-world media for experimentation.

### Video

A ~30-second 1080p highway video is used to investigate:

* frame compression
* spatial redundancy
* temporal redundancy
* reconstruction quality
* computational cost

### Audio

A 1:06.206 stereo WAV recording is used to investigate:

* audio representations
* time-frequency representations
* low-rank approximations
* reconstruction quality

### Images

Images are used to investigate how different visual structures affect the effectiveness of low-rank approximations.

All media included in the repository are kept deliberately small to maintain a repository size limit of approximately **100 MB**.

---

## Project Structure

```text
SVD_Data_Compression/
│
├── data/
│   ├── images/
│   │   ├── input/
│   │   └── output/
│   │
│   ├── audio/
│   │   ├── input/
│   │   └── output/
│   │
│   └── video/
│       ├── input/
│       └── output/
│
├── src/
│   ├── core/
│   ├── image/
│   ├── audio/
│   └── video/
│
├── tests/
│
├── notebooks/
│
├── benchmarks/
│
├── docs/
│
├── outputs/
│
├── requirements.txt
└── README.md
```

---

## Libraries

The initial implementation uses a deliberately small scientific Python stack.

| Library        | Purpose                                                              |
| -------------- | -------------------------------------------------------------------- |
| **NumPy**      | Numerical arrays and linear algebra                                  |
| **SciPy**      | Scientific computing, signal processing, and advanced linear algebra |
| **OpenCV**     | Image and video processing                                           |
| **Matplotlib** | Visualization and analysis                                           |
| **Pandas**     | Benchmark and experiment data                                        |
| **SoundFile**  | Reading and writing audio                                            |

GPU acceleration will be investigated later using **PyTorch/CUDA**.

---

## Experiments

The project will progressively investigate:

### 1. SVD Fundamentals

* Matrix decomposition
* Singular values
* Singular vectors
* Rank
* Low-rank approximations
* Reconstruction error

### 2. Image Compression

For an image represented by a matrix \(A\):

$$
A \approx U_k\Sigma_kV_k^T
$$

Experiments will examine how reconstruction changes as \(k\) varies.

Example:

```text
k = 5
k = 10
k = 25
k = 50
k = 100
```

Measurements will include:

* compression ratio
* reconstruction error
* PSNR
* SSIM
* computation time

### 3. Audio Compression

Audio is inherently a time-series signal, so the project will investigate appropriate matrix representations rather than simply reshaping the raw samples.

Potential representations include:

* spectrograms
* STFT matrices
* structured signal matrices

SVD can then be applied to these representations to investigate low-rank structure.

### 4. Video Compression

Video introduces both spatial and temporal redundancy.

The project will investigate:

$$
I_t \approx U_k\Sigma_kV_k^T
$$

for individual frames, followed by approaches that consider relationships between frames.

The objective is to understand whether temporal redundancy can be exploited alongside spatial redundancy.

### 5. Robotics Applications

The techniques will eventually be considered in contexts such as:

* robotic cameras
* autonomous vehicles
* drones
* teleoperation
* remote robots
* sensor-data logging
* bandwidth-constrained robotic systems

---

## Evaluation

Compression will not be evaluated using file size alone.

Important measurements include:

### Compression Ratio

$$
CR =
\frac{\text{original representation size}}
{\text{compressed representation size}}
$$

### Reconstruction Error

For an original matrix \(A\) and reconstruction \(\hat A\):

$$
E = \|A-\hat A\|
$$

Different error metrics will be investigated depending on the type of data.

### Computational Cost

We will also measure:

* decomposition time
* reconstruction time
* memory usage
* scaling with matrix size

This is particularly important for robotics, where computation may be constrained by the robot's hardware.

---

## CPU vs GPU

Once the CPU implementation is understood, the project will investigate GPU acceleration.

The intended progression is:

```text
NumPy / SciPy
      ↓
CPU SVD
      ↓
Benchmark
      ↓
GPU implementation
      ↓
CUDA
      ↓
Benchmark
      ↓
Compare
```

The objective is to understand when GPU acceleration actually provides a meaningful advantage rather than assuming that every SVD operation benefits from a GPU.

---

## Repository Constraints

This project intentionally has a **100 MB maximum repository size**.

The media collection is therefore kept small and focused.

Large external datasets are not required. Experiments should preferably use:

* small real-world media
* generated test data
* reproducible experiments
* compact benchmark results

The project should remain easy to clone, inspect, and experiment with.

---

## Status

🚧 **Early development**

Current focus:

* [x] Define project scope
* [x] Select initial media
* [x] Define Python stack
* [ ] Set up project structure
* [ ] Implement basic SVD experiments
* [ ] Implement image compression
* [ ] Implement audio experiments
* [ ] Implement video experiments
* [ ] Add compression metrics
* [ ] Add robotics-focused experiments
* [ ] Benchmark CPU implementations
* [ ] Investigate GPU acceleration

---

## License & Media Attribution

Source media obtained from external websites retains the licensing terms of its respective creators/providers.

See the project's media attribution information for the source and license of each included asset.

---

## Why SVD?

SVD is more than a compression technique.

It provides a way to identify the **dominant structure within data**.

That makes it useful for understanding:

* dimensionality reduction
* signal reconstruction
* noise reduction
* matrix approximation
* numerical linear algebra
* computer vision
* robotics
* scientific computing

This project uses compression as the practical starting point for exploring those ideas.
