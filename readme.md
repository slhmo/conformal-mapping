# Conformal Image Mapper

An asynchronous, mathematical image-warping platform built to apply advanced complex analysis and coordinate transformations to images. By leveraging inverse coordinate grid mapping and bilinear interpolation, this engine allows you to compose, chain, and render intricate conformal maps—including log-polar Droste spirals, Möbius transformations, and custom Cartesian grid distortions—without grid tearing or pixel artifacts.

---

## 🚀 Features

* **Interactive Control Dashboard:** A clean, simple workspace built so that users can dynamically chain transformation operators, adjust focal centers, configure math scale domains, and track effects on their images.
* **Chained Transformation Pipelines:** Compose functions sequentially via function composition ($f \circ g \circ h(z)$) directly from the UI.
* **Dynamic Custom Expression Sandbox:** Write multi-line Python and NumPy code blocks on the fly inside an interactive modal workspace. The engine dynamically compiles and integrates your custom mathematical mappings into the execution pipeline.
* **Asynchronous Execution Architecture:** Heavy mathematical pixel-mapping loops are offloaded to **Celery workers** backed by a **Redis** message broker, ensuring the web interface remains fluid and non-blocking.
* **High-Fidelity Inverse Mapping Engine:** Uses a backward grid rendering approach utilizing NumPy vectorized meshranges. Instead of pushing source pixels forward (which creates holes), it maps destination coordinates back to the source image space using **bilinear interpolation**.

---

## 🎬 Showcase

![Transformation Showcase](images/engine_showcase_optimized.gif)

*A preview of the transformation pipeline executing chained complex transformations and coordinate distortions.*

---

## 📐 Mathematical Transformations Supported

The engine translates pixel dimensions $(w, h)$ into a continuous complex domain box defined by custom interval bounds (e.g., $[- \pi, \pi]$). The following functions map the target grid vector $z = x + iy$ back to the source image bounds:

### 1. Complex & Conformal Mappings
* **Twisted Droste (Escher/3Blue1Brown variant):** Applies a log-polar conformal warp to produce a seamless, infinite regression spiral framework using:
  $$f(z) = \exp\left( \frac{\ln(z)}{c} \right) \quad \text{where} \quad c = \frac{2\pi i}{\ln(\text{scale}) + 2\pi i}$$
* **Straight Droste:** Concentric infinite nesting without the diagonal spiral twist component.
* **Möbius Transformation:** Maps lines and circles to other lines and circles across the Riemann sphere:
  $$f(z) = \frac{z - 1}{z + 1 + \epsilon}$$
* **Complex Exponential & Logarithm:** Mappings utilizing $f(z) = e^z$ and $f(z) = \ln(z)$ to map rectangular structures into polar structures and vice versa.
* **Sinusoidal Distortion:** Adds uniform mathematical wave noise to coordinates via $f(z) = z + 0.1\sin(z)$.

### 2. Cartesian Coordinate Distortions
Adapts pure spatial transformations $(x, y) \mapsto (x', y')$ back into the complex grid layer:
* **Polynomial Warps:** Non-linear structural distortions along targeted axes:
  * **$x^2$ / $x^3$ Grid Warps:** $f(x, y) = (x^n, y)$
  * **$y^2$ / $y^3$ Grid Warps:** $f(x, y) = (x, y^n)$
* **Coordinate Stretching:** Axis-specific matrix multiplier adjustments.

* **Since we are using reverse mapping to map any pixel of target image to their corresponding source pixel, We need to apply reversed functions. It's not handled in the code so, e.g. you want to apply $f(z) = e^z$ use $f(z) = \ln(z).$**

---

## 🛠️ Tech Stack

* **Backend Framework:** Django 5.2 (Python 3.12-slim base)
* **Task Management:** Celery 5.6 & Redis 7 (Alpine-based)
* **Math & Image Processing:** NumPy 2.4, Pillow 12.2, Matplotlib 3.10
* **Frontend Interface:** HTML5, Tailwind CSS 4, Vanilla JS Async Fetch API

---

## 📦 Infrastructure & Setup

The easiest way to stand up the full stack (Web App, Celery Distributed Worker, and Redis Key-Value Store) is using **Docker Compose**.

### Method 1: Quickstart with Docker (Recommended)

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/slhmo/conformal-mapping.git
   cd conformal-image-mapper
   
2. **Launch the Containerized Environment:**
   ```bash
   docker-compose up --build
   
3. **Access the Dashboard:**
    Open your browser and navigate to
    http://localhost:8000.