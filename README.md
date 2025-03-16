# Failure Probability Estimation for Robotic Grasping (FPERG)

<div align="center">
  <img src="https://github.com/user-attachments/assets/5678c711-f6c3-441d-98b3-a55d7759f65c" width="30%" alt="Success Grasp"/>
  <img src="https://github.com/user-attachments/assets/02fd7a67-472f-4f99-9793-88ce8548956a" width="30%" alt="Failure Mode 1"/>
  <img src="https://github.com/user-attachments/assets/25e37b2c-4e9e-46db-b85a-00461ed0088c" width="30%" alt="Failure Mode 2"/>
</div>
<div align="center">
  <em>Left: Successful grasp. Middle and Right: Common failure modes.</em>
</div>

<br>

This repository contains the implementation of three different failure probability estimation methods for robotic grasping tasks: Direct Estimation (DE), Importance Sampling (IS), and Adaptive Importance Sampling (AIS). Using a simulated robotic arm with a parallel-jaw gripper in PyBullet, we compare these methods in terms of efficiency, accuracy, and computational cost.

## Project Overview

Reliable grasping is fundamental for robotic manipulation tasks, and understanding failure modes is crucial for deployment in real-world applications. This project aims to:

1. Implement and compare three different approaches for estimating the probability of failures in robotic grasping
2. Evaluate these methods in terms of sample efficiency, convergence rate, and failure case coverage
3. Analyze the sources of discrepancies and propose validation techniques to ensure accurate estimation

## Installation

1. Clone this repository:
```bash
git clone https://github.com/jfrausto7/FPERG.git
cd FPERG
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Grasping Policy

To train a hill climbing grasping policy:

```bash
python train.py --episodes 2000 --gui  # Use GUI for visualization
```

Options:
- `--episodes`: Number of training episodes (default: 2000)
- `--gui`: Use GUI mode instead of DIRECT mode (default: False)
- `--load`: Load existing policy (default: False)
- `--eval`: Evaluate only (no training) (default: False)

### Running Failure Probability Estimation

To run failure probability estimation experiments:

```bash
python main.py [OPTIONS]
```

Options:
- `--gui`: Use GUI mode instead of DIRECT mode
- `--multiple N`: Run N grasp attempts
- `--seed N`: Set random seed
- `--hill`: Use Hill Climbing policy instead of default policy
- `--policy-file PATH`: Path to policy file (default: src/best_hill_climbing_policy.pkl)

#### Direct Estimation

```bash
python main.py --estimate --trials 1000 --hill
```

#### Importance Sampling

```bash
python main.py --importance --trials 1000 --depth 1000 --hill
```

#### Adaptive Importance Sampling

```bash
python main.py --adaptive_importance --trials 1000 --depth 1000 --hill
```

### Example: Comparing All Methods

To run a comparison of all three methods with 1000 trials each:

```bash
python main.py --estimate --trials 1000 --hill
python main.py --importance --trials 1000 --depth 1000 --hill
python main.py --adaptive_importance --trials 1000 --depth 1000 --hill
```

The results will be saved in the `results/` directory and printed to the console.

## Visualization

To visualize and compare results from different methods, use the analysis notebook.

## Contributors

- Yasmina Abukhadra (yasabukh@stanford.edu)
- Jacob Frausto (jfrausto@stanford.edu)
- Ian Lasic-Ellis (ianlasic@stanford.edu)
