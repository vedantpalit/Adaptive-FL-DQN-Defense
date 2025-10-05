import random
import numpy as np
import torch
import pandas as pd
from tqdm import trange
from copy import deepcopy

from .fl_models import BackdoorTrigger
from .fl_data import get_cifar10_loaders, create_non_iid_splits, make_client_subsets
from .fl_server import Client, DQNPOMDPServer
from .fl_metrics import compute_roc_metrics, compute_calibration_error

def get_default_config():
    return {
        "gamma": 0.9,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 5000,
        "batch_size": 64,
        "dqn_lr": 0.001,
        "buffer_capacity": 10000,
        "hidden_dim": 64,
        "update_target_freq": 100,
        "lambda_param": 0.3,
        "eta_param": 0.2,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma_cost": 0.5,
        "delta": 0.5,
        "use_dqn": True,
        "use_belief_update": True,
        "use_trust_scores": True,
        "use_anomaly_detection": True,
        "signal_budget": "full",
    }

def run_enhanced_federated_learning(
    config,
    num_clients=10,
    num_rounds=30,
    clients_per_round=5,
    malicious_ratio=0.2,
    attack_strength=0.5,
    attack_type="backdoor",
    alpha_dirichlet=0.5,
    seed=256,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    train, test_loader = get_cifar10_loaders()
    client_indices = create_non_iid_splits(train, num_clients, alpha=alpha_dirichlet, seed=seed)
    subsets = make_client_subsets(train, client_indices)
    clients = []
    for i in range(num_clients):
        vuln = random.uniform(0.1, 0.8)
        clients.append(Client(i, subsets[i], vulnerability_score=vuln))
    server = DQNPOMDPServer(clients, test_loader, config)

    metrics = {"accuracy": [], "asr": [], "rewards": [], "beliefs": [],
               "trusts": [], "roc_auc": [], "pr_auc": [], "ece": []}

    for round_num in trange(num_rounds, desc="FL Rounds"):
        selected = server.select_clients(clients_per_round)
        selected_vuln = sorted([(i, clients[i].vulnerability_score) for i in selected],
                               key=lambda x: x[1], reverse=True)
        num_mal = max(1, int(len(selected)*malicious_ratio))
        malicious = [cid for cid, _ in selected_vuln[:num_mal]]

        for i in range(num_clients):
            clients[i].is_malicious = (i in malicious)

        if attack_type == "collusion" and server.collusion_vector is None:
            server.collusion_vector = [torch.randn_like(p) for p in server.global_model.parameters()]
            for c in clients:
                if c.is_malicious: c.collusion_vector = server.collusion_vector

        state = server.get_state_tensor()
        old_acc = server.test_global_model()
        asr = None
        if attack_type == "backdoor":
            asr = server.test_backdoor_asr(BackdoorTrigger())

        actions = [server.select_action(state, cid) for cid in selected]
        server.apply_actions(selected, actions)

        updates = [clients[cid].local_train(server.global_model,
                                            attack_type=attack_type,
                                            attack_strength=attack_strength)
                   for cid in selected]

        client_w = server.aggregate_models(updates, selected)
        new_acc = server.test_global_model()
        reward = server.calculate_reward(old_acc, new_acc, selected, actions, malicious, asr)
        server.update_beliefs_and_trust(selected, client_w, updates, old_acc, new_acc)

        next_state = server.get_state_tensor()
        for i, _cid in enumerate(selected):
            server.replay.push(state.squeeze(0).numpy(), actions[i],
                               reward, next_state.squeeze(0).numpy(), False)
        server.update_dqn()

        metrics["accuracy"].append(new_acc)
        if asr is not None: metrics["asr"].append(asr)
        metrics["rewards"].append(reward)
        metrics["beliefs"].append(server.belief.copy())
        metrics["trusts"].append([c.trust_score for c in clients])

        is_mal = [c.is_malicious for c in clients]
        roc_auc, pr_auc, _, _ = compute_roc_metrics(server.belief, is_mal)
        ece = compute_calibration_error(server.belief, np.array(is_mal))
        metrics["roc_auc"].append(roc_auc); metrics["pr_auc"].append(pr_auc); metrics["ece"].append(ece)

        if (round_num + 1) % 5 == 0:
            msg = f"Round {round_num+1}: Acc={new_acc:.2f}%"
            if asr is not None: msg += f", ASR={asr:.2f}%"
            msg += f", ROC-AUC={roc_auc:.3f}, ECE={ece:.3f}"
            print(msg)

    return metrics, server, clients

def run_dirichlet_sweep(alphas=None, num_rounds=50, runs_per_alpha=2):
    if alphas is None: alphas = [0.1, 0.3, 0.5, 1.0, 5.0]
    results = []
    base = get_default_config()
    for a in alphas:
        print(f"\n=== Dirichlet α={a} ===")
        for r in range(runs_per_alpha):
            print(f"  Run {r+1}/{runs_per_alpha}")
            m, _, _ = run_enhanced_federated_learning(
                config=deepcopy(base),
                num_rounds=num_rounds,
                alpha_dirichlet=a,
                attack_type="backdoor",
                seed=42 + r,
            )
            results.append({
                "alpha": a, "run": r,
                "final_accuracy": m["accuracy"][-1],
                "final_asr": m["asr"][-1] if "asr" in m and m["asr"] else 0,
                "avg_roc_auc": float(np.mean(m["roc_auc"])),
                "final_ece": m["ece"][-1],
            })
    return pd.DataFrame(results)

def run_signal_budget_study(num_rounds=50,seed=256):
    budgets = ["full", "no_validation", "directional"]
    results = []
    for b in budgets:
        print(f"\nSignal budget: {b}")
        cfg = get_default_config(); cfg["signal_budget"] = b
        m, _, _ = run_enhanced_federated_learning(
            config=cfg, num_rounds=num_rounds, attack_type="backdoor", seed=seed
        )
        results.append({
            "signal_budget": b,
            "final_accuracy": m["accuracy"][-1],
            "final_asr": m["asr"][-1] if "asr" in m and m["asr"] else 0,
            "avg_roc_auc": float(np.mean(m["roc_auc"])),
            "final_ece": m["ece"][-1],
        })
    return pd.DataFrame(results)

