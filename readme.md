# Uncertainty-Calibrated Planning for Reliable Long-Horizon LLM Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Playwright](https://img.shields.io/badge/playwright-async-orange.svg)](https://playwright.dev/python/)

Official implementation and experimental testbed for **"Uncertainty-Calibrated Planning for Reliable Long-Horizon LLM Agents"**. This framework addresses the critical bottleneck of compounding errors in multi-step LLM agents by introducing history-conditioned trajectory risk estimation ($R_t$) paired with an active intervention gating policy.


## ⚙️ Installation & Environment Setup

```bash 
git clone https://github.com/MiSu2002/webvoyager_uncertainty_agent.git
cd webvoyager_uncertainty_agent
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

## 🚀 Running Tests & Ablation Studies

```bash
python -m tests.run_ablations
python -m evaluation.run_benchmark
```

## 📊 Evaluation Metrics

```text
**Task Success Rate (SR):** Percentage of WebVoyager evaluation tasks successfully reaching the target state.
**Trajectory-Checkpoint Expected Calibration Error (TC-ECE):** Quantifies alignment between estimated failure risk Rt and empirical failure rates across execution checkpoints.
```

## 📂 Project Repository Structure

```text
webvoyager_uncertainty_agent/
│
├── webvoyager_env/            # Asynchronous Playwright browser automation wrappers
│   └── browser_env.py
│
├── agents/                    # Agent control loop implementations
│   ├── base_react.py          # Standard greedy ReAct baseline
│   ├── static_plan.py          # Static Plan-and-Act baseline (fixed-interval replanning)
│   └── uncertainty_agent.py   # Proposed Uncertainty-Calibrated Agent
│
├── core/                      # Mathematical risk engines and gating logic
│   ├── uncertainty_scorer.py  # Computes history-conditioned trajectory risk R_t
│   └── gating_policy.py       # Implements operational thresholds Φ(R_t)
│
├── evaluation/                # Benchmarking and metric tracking harnesses
│   ├── metrics.py             # Computes Task Success Rate & Trajectory-Checkpoint ECE
│   └── run_benchmark.py       # Main evaluation harness across WebVoyager tasks
│
├── tests/                     # Verification and ablation test suite
│   └── run_ablations.py       # Isolates components for multi-component ablation analysis
│
├── requirements.txt           # Project dependencies
└── README.md                  # Reproduction and execution instructions
