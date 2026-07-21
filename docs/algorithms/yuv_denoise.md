# YUV Domain Noise Reduction (YNR & CNR)

This document details the mathematical algorithms and CUDA implementation design for the YUV Domain Denoising block (`YuvDenoise`) in the camera ISP pipeline.

---

## 1. Background & Core Concepts

Camera sensor noise is heavily amplified in the color correction matrix (CCM) and white balance stages, resulting in visually displeasing color splotches (chromatic noise) and grain (luminance noise). 

While raw-domain denoising (BNR) is effective at removing pixel-level sensor noise, filtering in the YUV (YCbCr) space is crucial for final image quality because it matches **Human Visual System (HVS)** characteristics:
* **Luminance (Y)**: Human eyes are highly sensitive to brightness details. The Y channel requires a **conservative, edge-preserving** filter (Luma Noise Reduction, or **YNR**) to maintain image sharpness.
* **Chrominance (U/V)**: Human eyes are insensitive to spatial color changes. The U and V channels can tolerate **aggressive smoothing** (Chroma Noise Reduction, or **CNR**) to eliminate chromatic color splotches without making the image appear blurry.

---

## 2. Mathematical Algorithms

### A. Luma Denoising (YNR) - Bilateral Filter
To filter the $Y$ channel while preserving sharp structural edges, we employ a standard **Bilateral Filter** that computes weights based on spatial distance and luminance similarity:

$$Y'_{c} = \frac{\sum_{p \in \Omega} W_s(p) \cdot W_{r}(Y_p - Y_c) \cdot Y_p}{\sum_{p \in \Omega} W_s(p) \cdot W_{r}(Y_p - Y_c)}$$

Where:
* $c$ is the center pixel, and $p$ is a neighboring pixel within a spatial window $\Omega$ (typically $5\times5$ or $7\times7$).
* $W_s(p) = \exp\left(-\frac{\|p - c\|^2}{2\sigma_s^2}\right)$ is the spatial Gaussian weight.
* $W_{r}(\Delta) = \exp\left(-\frac{\Delta^2}{2\sigma_r^2}\right)$ is the range Gaussian weight, which penalizes pixels that have significantly different brightness levels (protecting sharp boundaries).

### B. Chroma Denoising (CNR) - Joint Bilateral Filter
Filtering $U$ and $V$ independently or using their own differences can cause **color bleeding** across object borders. 

To prevent color leakage, we use a **Joint Bilateral Filter**, where the spatial weights are standard, but the range similarity weights are guided by the **Luma ($Y$) channel differences**:

$$U'_{c} = \frac{\sum_{p \in \Omega} W_s(p) \cdot W_{r\_y}(Y_p - Y_c) \cdot U_p}{\sum_{p \in \Omega} W_s(p) \cdot W_{r\_y}(Y_p - Y_c)}$$

$$V'_{c} = \frac{\sum_{p \in \Omega} W_s(p) \cdot W_{r\_y}(Y_p - Y_c) \cdot V_p}{\sum_{p \in \Omega} W_s(p) \cdot W_{r\_y}(Y_p - Y_c)}$$

* **Rationale**: The $Y$ channel represents the true physical boundaries of the objects. Guiding the chroma filtering using $Y$ differences ensures color smoothing stops exactly at luminance edges, completely eliminating color bleeding.

---

## 3. Color Space Conversions (BT.709)

Since our pipeline processes colors in normalized floats ($[0.0, 1.0]$), we use the HDTV standard **BT.709** conversion with a $+0.5$ offset for $U$ and $V$ to keep all values non-negative and in the $[0.0, 1.0]$ range.

### RGB to YUV ($R,G,B \in [0, 1] \rightarrow Y,U,V \in [0, 1]$)
$$Y = 0.2126 \cdot R + 0.7152 \cdot G + 0.0722 \cdot B$$
$$U = -0.1146 \cdot R - 0.3854 \cdot G + 0.5000 \cdot B + 0.5$$
$$V = 0.5000 \cdot R - 0.4542 \cdot G - 0.0458 \cdot B + 0.5$$

### YUV to RGB ($Y,U,V \in [0, 1] \rightarrow R,G,B \in [0, 1]$)
$$R = Y + 1.5748 \cdot (V - 0.5)$$
$$G = Y - 0.1873 \cdot (U - 0.5) - 0.4681 \cdot (V - 0.5)$$
$$B = Y + 1.8556 \cdot (U - 0.5)$$

*Note: The final computed RGB channels must be clamped to $[0.0, 1.0]$ to prevent float rounding overflows.*

---

## 4. CUDA Implementation Strategy

To achieve high throughput on modern GPUs, we avoid multiple memory-bound passes by employing a **Fused Kernel design**:

```
[Global Read] Read YUV pixels from neighborhood into Shared Memory or Registers
      │
      ▼
[Input]       RGB to YUV conversion is handled by a dedicated upstream block
      │
      ▼
[Filtering]   Run Bilateral YNR and Joint Bilateral CNR using shared coordinates
      │
      ▼
[Global Write] Write the final denoised YUV pixel back to Global Memory
```

### Key Performance Benefits
1. **Pipeline locality**: The denoiser writes YUV directly. Edge enhancement can operate on that YUV result before the single final conversion to RGB.
2. **Shared Memory Tiling**: Storing neighbor pixels in `__shared__` memory allows threads to share overlapping pixel read loads, drastically reducing global memory access overhead.
