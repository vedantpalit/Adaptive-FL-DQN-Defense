# Adaptive Federated Learning Defences via Trust-Aware Deep Q-Networks

Official implementation accompanying our paper [**"Adaptive Federated Learning Defences via Trust-Aware Deep Q-Networks"**](https://arxiv.org/abs/2510.01261) (Under Review at ICLR 2026)  

This repository provides a modular and reproducible framework for studying *robust federated learning defenses* under *partial observability*, implemented with a **POMDP-based server** and multiple **reinforcement learning (RL) controllers** — including DQN, Policy Gradient, Linear Q-learning, and Random baselines.

---

## Overview

Federated learning (FL) is vulnerable to *poisoning and backdoor attacks*.  
We model defense as a **partially observable sequential decision problem**, where the central server must infer client reliability and take corrective actions over time.

Our framework introduces:
- **Trust-aware POMDP server** for belief-based aggregation.
- **Multi-agent controller benchmarking** (DQN, LinearQ, PG, Random).
- **Comprehensive metrics**: accuracy, ASR, ROC-AUC, calibration (ECE), and reward.
- **Non-IID simulation** using Dirichlet-distributed client data.

---

## Installation

### Requirements
- Python 3.10+
- PyTorch ≥ 2.0
- torchvision, numpy, pandas, matplotlib, seaborn, tqdm, scipy

```bash
git clone https://github.com/vedantpalit/TrustAwareFL-RL.git
cd TrustAwareFL-RL
pip install -r flrl/requirements.txt
```



## Reproducing Core Experiments

### **Baseline Backdoor Defense**
```bash
python -m flrl.main baseline --rounds 50 --alpha 0.5 --seed 42 #64 128 200 256
```

### **Dirichlet Non-IID Sweep**
```bash
python -m flrl.main dirichlet --alphas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 5.0 --rounds 50 --runs 5
```

### **Signal-Budget Study**
```bash
python -m flrl.main signal --rounds 50 --seed 42 #64 128 200 256
```

---

## RL Agent Benchmarking

Run all RL agents (DQN, Random, LinearQ, PolicyGradient) and save per-agent results.

```bash
python run_all_agents.py --rounds 50 --runs 3 --alpha 0.5 --out agent_results
```

---

## Metrics

Each run logs:
| Metric | Description |
|--------|--------------|
| **Accuracy** | Global model test accuracy |
| **ASR** | Attack Success Rate (backdoor strength) |
| **ROC-AUC** | Detection of malicious clients |
| **ECE** | Expected Calibration Error |
| **Reward** | Server's POMDP-based reinforcement signal |

---

## Experimental Notes

- Default setting: 10 clients, 20% malicious, Dirichlet α=0.5, Rounds=50 .
- GPU optional; single-GPU (T4) recommended for full benchmark.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{palit2025adaptivefederatedlearningdefences,
      title={Adaptive Federated Learning Defences via Trust-Aware Deep Q-Networks}, 
      author={Vedant Palit},
      year={2025},
      eprint={2510.01261},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.01261}, 
}
```


