# run_all_agents.py
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# --- bring in your earlier modules ---
from flrl.fl_experiments import (
    get_default_config,
    run_enhanced_federated_learning,
)
from flrl.fl_metrics import compute_roc_metrics, compute_calibration_error

# Server-side pieces used in the custom-agent runner
from flrl.fl_server import Client, DQNPOMDPServer
from flrl.fl_models import BackdoorTrigger
from flrl.fl_data import get_cifar10_loaders, create_non_iid_splits, make_client_subsets

# ----------------------
# Minimal agent classes
# ----------------------
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RandomAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = "Random"
    def select_action(self, state):
        return random.randint(0, self.action_dim - 1)
    def update(self, *args, **kwargs):
        pass

class LinearQAgent:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.9, epsilon=0.1):
        import numpy as np
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.name = "LinearQ"
        self.weights = np.random.randn(state_dim, action_dim) * 0.01
    def get_q_values(self, state):
        import numpy as np
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if len(state.shape) > 1:
            state = state.flatten()
        return np.dot(state, self.weights)
    def select_action(self, state):
        import numpy as np
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        q = self.get_q_values(state)
        return int(np.argmax(q))
    def update(self, state, action, reward, next_state, done):
        import numpy as np
        if isinstance(state, torch.Tensor): state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor): next_state = next_state.cpu().numpy()
        if len(state.shape) > 1: state = state.flatten()
        if len(next_state.shape) > 1: next_state = next_state.flatten()
        current_q = np.dot(state, self.weights[:, action])
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.get_q_values(next_state))
        td_error = target_q - current_q
        self.weights[:, action] += self.lr * td_error * state
        self.epsilon = max(0.01, self.epsilon * 0.995)

class PolicyGradientAgent:
    """Simple REINFORCE."""
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.name = "PolicyGradient"
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim), nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards = []
    def select_action(self, state):
        import numpy as np
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if len(state.shape) == 1: state = state.unsqueeze(0)
        probs = self.policy_net(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return int(action.item())
    def update(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        if len(self.rewards) >= 100:
            self._update_policy()
    def _update_policy(self):
        if len(self.rewards) == 0 or len(self.saved_log_probs) == 0:
            return
        L = min(len(self.saved_log_probs), len(self.rewards))
        saved = self.saved_log_probs[:L]
        rews = self.rewards[:L]
        returns = []
        R = 0
        for r in reversed(rews):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = []
        for logp, R in zip(saved, returns):
            if logp.requires_grad:
                policy_loss.append(-logp * R.detach())
        if policy_loss:
            self.optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum()
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []

# -----------------------------------------
# Minimal custom-agent FL runner (per agent)
# -----------------------------------------
def run_enhanced_fl_with_custom_agent(
    config,
    agent_type: str,
    num_rounds: int = 50,
    clients_per_round: int = 5,
    malicious_ratio: float = 0.2,
    attack_type: str = "backdoor",
    alpha_dirichlet: float = 0.5,
    seed: int = 42,
):
    import random as pyrandom
    import numpy as onp
    import torch
    from tqdm import tqdm

    pyrandom.seed(seed); onp.random.seed(seed); torch.manual_seed(seed)

    train, test_loader = get_cifar10_loaders()
    num_clients = 10
    client_indices = create_non_iid_splits(train, num_clients, alpha=alpha_dirichlet, seed=seed)
    subsets = make_client_subsets(train, client_indices)

    clients = []
    for i in range(num_clients):
        vuln = pyrandom.uniform(0.1, 0.8)
        clients.append(Client(i, subsets[i], vulnerability_score=vuln))

    server = DQNPOMDPServer(clients, test_loader, config)

    state_dim, action_dim = server.state_dim, server.action_dim
    if agent_type == "Random":
        agent = RandomAgent(state_dim, action_dim)
    elif agent_type == "LinearQ":
        agent = LinearQAgent(state_dim, action_dim, lr=0.01, gamma=config["gamma"])
    elif agent_type == "PolicyGradient":
        agent = PolicyGradientAgent(state_dim, action_dim, lr=0.001, gamma=config["gamma"])
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    metrics = {"accuracy": [], "asr": [], "rewards": [], "beliefs": [],
               "trusts": [], "roc_auc": [], "pr_auc": [], "ece": []}

    for _round in tqdm(range(num_rounds), desc=f"{agent_type} FL"):
        selected = server.select_clients(clients_per_round)

        sel_vuln = sorted([(i, clients[i].vulnerability_score) for i in selected],
                          key=lambda x: x[1], reverse=True)
        num_mal = max(1, int(len(selected) * malicious_ratio))
        malicious = [cid for cid, _ in sel_vuln[:num_mal]]

        for i in range(num_clients):
            clients[i].is_malicious = (i in malicious)

        state = server.get_state_tensor().squeeze(0).numpy()
        old_acc = server.test_global_model()

        asr = None
        if attack_type == "backdoor":
            asr = server.test_backdoor_asr(BackdoorTrigger())

        actions = [agent.select_action(state) for _ in selected]
        server.apply_actions(selected, actions)

        updates = [clients[cid].local_train(server.global_model,
                                            attack_type=attack_type,
                                            attack_strength=0.5)
                   for cid in selected]

        client_w = server.aggregate_models(updates, selected)
        new_acc = server.test_global_model()
        reward = server.calculate_reward(old_acc, new_acc, selected, actions, malicious, asr)
        server.update_beliefs_and_trust(selected, client_w, updates, old_acc, new_acc)

        next_state = server.get_state_tensor().squeeze(0).numpy()
        for i, _cid in enumerate(selected):
            agent.update(state, actions[i], reward, next_state, False)

        metrics["accuracy"].append(new_acc)
        if asr is not None: metrics["asr"].append(asr)
        metrics["rewards"].append(reward)
        metrics["beliefs"].append(server.belief.copy())
        metrics["trusts"].append([c.trust_score for c in clients])

        is_mal = [c.is_malicious for c in clients]
        roc_auc, pr_auc, _, _ = compute_roc_metrics(server.belief, is_mal)
        ece = compute_calibration_error(server.belief, np.array(is_mal))
        metrics["roc_auc"].append(roc_auc); metrics["pr_auc"].append(pr_auc); metrics["ece"].append(ece)

    return metrics

# -----------------------------------------
# Batch runner to just SAVE results per agent
# -----------------------------------------
def run_and_save_all_agents(
    num_rounds: int = 50,
    num_runs: int = 3,
    alpha_dirichlet: float = 0.5,
    attack_type: str = "backdoor",
    out_root: str = "agent_results",
    agents=("DQN", "Random", "LinearQ", "PolicyGradient"),
):
    os.makedirs(out_root, exist_ok=True)
    cfg = get_default_config()

    for agent in agents:
        save_dir = os.path.join(out_root, agent)
        os.makedirs(save_dir, exist_ok=True)

        all_run_rows = []         # summary row per run
        detailed_metrics = []     # full metrics per run

        for run_idx in range(num_runs):
            seed = 42 + run_idx
            print(f"\n[{agent}] Run {run_idx+1}/{num_runs} (seed={seed})")

            if agent == "DQN":
                # Use server's built-in DQN
                metrics, _, _ = run_enhanced_federated_learning(
                    config=cfg,
                    num_rounds=num_rounds,
                    attack_type=attack_type,
                    alpha_dirichlet=alpha_dirichlet,
                    seed=seed
                )
            else:
                # Use custom agent runner
                cfg_local = cfg.copy()
                cfg_local["use_dqn"] = False  # server shouldn't make decisions
                metrics = run_enhanced_fl_with_custom_agent(
                    config=cfg_local,
                    agent_type=agent,
                    num_rounds=num_rounds,
                    attack_type=attack_type,
                    alpha_dirichlet=alpha_dirichlet,
                    seed=seed
                )

            # Summaries for this run
            final_acc = metrics["accuracy"][-1]
            final_asr = metrics["asr"][-1] if metrics["asr"] else 0
            avg_roc  = float(np.mean(metrics["roc_auc"]))
            final_ece = metrics["ece"][-1]
            avg_reward = float(np.mean(metrics["rewards"]))

            acc_arr = np.array(metrics["accuracy"])
            conv_round = int(np.argmax(acc_arr > 50)) if np.any(acc_arr > 50) else num_rounds

            all_run_rows.append({
                "agent": agent,
                "run": run_idx,
                "final_accuracy": final_acc,
                "final_asr": final_asr,
                "avg_roc_auc": avg_roc,
                "final_ece": final_ece,
                "avg_reward": avg_reward,
                "convergence_round": conv_round,
                "max_accuracy": float(np.max(acc_arr)),
                "min_asr": float(np.min(metrics["asr"])) if metrics["asr"] else 0.0,
            })
            detailed_metrics.append(metrics)

        # Save files for this agent
        df_runs = pd.DataFrame(all_run_rows)
        df_runs.to_csv(os.path.join(save_dir, "raw_results.csv"), index=False)

        with open(os.path.join(save_dir, "detailed_metrics.pkl"), "wb") as f:
            pickle.dump(detailed_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[{agent}] Saved: {save_dir}/raw_results.csv and detailed_metrics.pkl")

    print(f"\nAll done. Root folder: {out_root}")

# -----------------------------------------
# CLI
# -----------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run and save per-agent results (no plots, no LaTeX).")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--attack", type=str, default="backdoor", choices=["backdoor", "collusion", "sign_flip", "scaling", "label_flip"])
    parser.add_argument("--out", type=str, default="agent_results")
    parser.add_argument("--agents", nargs="+", default=["DQN", "Random", "LinearQ", "PolicyGradient"])
    args = parser.parse_args()

    run_and_save_all_agents(
        num_rounds=args.rounds,
        num_runs=args.runs,
        alpha_dirichlet=args.alpha,
        attack_type=args.attack,
        out_root=args.out,
        agents=tuple(args.agents),
    )
