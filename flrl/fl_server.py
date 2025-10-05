import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader

from .fl_models import SimpleCNN, BackdoorTrigger
from .fl_metrics import compute_roc_metrics, compute_calibration_error

# ---------- Client ----------
class Client:
    def __init__(self, client_id, train_data, vulnerability_score=0.0):
        self.id = client_id
        self.train_data = train_data
        self.vulnerability_score = vulnerability_score
        self.is_malicious = False
        self.trust_score = 1.0
        self.aggregation_weight = 1.0
        self.collusion_vector = None
        self.trigger = BackdoorTrigger()

    def local_train(self, global_model, epochs=1, batch_size=32, lr=0.01,
                    attack_strength=0.5, attack_type="backdoor",
                    backdoor_fraction=0.1, target_class=7, source_class=3):
        model = SimpleCNN()
        model.load_state_dict(global_model.state_dict())
        loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        opt = optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for x, y in loader:
                opt.zero_grad()
                if self.is_malicious and attack_type == "backdoor":
                    num_poison = int(batch_size * backdoor_fraction)
                    if num_poison > 0:
                        for i in range(min(num_poison, x.size(0))):
                            x[i] = self.trigger.apply_trigger(x[i])
                            y[i] = self.trigger.target_label
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()

                if self.is_malicious:
                    with torch.no_grad():
                        if attack_type == "sign_flip":
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad = -attack_strength * p.grad
                        elif attack_type == "scaling":
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad *= (1 + attack_strength * 10)
                        elif attack_type == "label_flip":
                            import numpy as np
                            num_flip = int(len(y) * attack_strength)
                            idx = torch.randperm(len(y))[:num_flip]
                            for j in idx:
                                y[j] = (y[j] + np.random.randint(1, 10)) % 10
                            out = model(x)
                            loss = loss_fn(out, y)
                            loss.backward()
                        elif attack_type == "collusion" and self.collusion_vector is not None:
                            for p, d in zip(model.parameters(), self.collusion_vector):
                                if p.grad is not None:
                                    p.grad = 0.7 * p.grad + 0.3 * d

                opt.step()
        return model.state_dict()

# ---------- DQN ----------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d): self.buffer.append((s, a, r, ns, d))
    def sample(self, batch):
        import random, torch
        s,a,r,ns,d = zip(*random.sample(self.buffer, batch))
        return (torch.FloatTensor(s), torch.LongTensor(a),
                torch.FloatTensor(r), torch.FloatTensor(ns),
                torch.FloatTensor(d))
    def __len__(self): return len(self.buffer)

# ---------- Server ----------
class DQNPOMDPServer:
    def __init__(self, clients, test_loader, config):
        import math
        self.clients = clients
        self.global_model = SimpleCNN()
        self.test_loader = test_loader
        self.config = config

        self.gamma = config["gamma"]
        self.eps_start = config["epsilon_start"]
        self.eps_end = config["epsilon_end"]
        self.eps_decay = config["epsilon_decay"]
        self.batch_size = config["batch_size"]
        self.dqn_lr = config["dqn_lr"]
        self.update_target_freq = config["update_target_freq"]
        self.train_steps = 0

        self.belief = np.array([c.vulnerability_score for c in clients])
        self.belief = self.belief / (np.max(self.belief) + 0.01)

        self.client_update_history = [[] for _ in clients]
        self.performance_history, self.asr_history = [], []
        self.last_seen = [-1 for _ in clients]

        self.num_clients = len(clients)
        self.state_dim = self.num_clients * 4
        self.action_dim = 3
        self.policy_net = DQN(self.state_dim, self.action_dim, config["hidden_dim"])
        self.target_net = DQN(self.state_dim, self.action_dim, config["hidden_dim"])
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.replay = ReplayBuffer(config["buffer_capacity"])
        self.opt = optim.Adam(self.policy_net.parameters(), lr=self.dqn_lr)
        self.epsilon = self.eps_start

        self.use_dqn = config.get("use_dqn", True)
        self.use_belief_update = config.get("use_belief_update", True)
        self.use_trust_scores = config.get("use_trust_scores", True)
        self.use_anomaly_detection = config.get("use_anomaly_detection", True)
        self.signal_budget = config.get("signal_budget", "full")

        self.collusion_vector = None
        self.rewards_history = []

    # --- Selection / state ---
    def select_clients(self, k):
        probs = np.array([c.vulnerability_score for c in self.clients]) + 0.1
        probs /= probs.sum()
        return np.random.choice(len(self.clients), k, replace=False, p=probs).tolist()

    def get_state_tensor(self):
        import torch
        st = []
        cur = len(self.performance_history)
        for i in range(self.num_clients):
            st += [self.belief[i], self.clients[i].trust_score,
                   self.clients[i].aggregation_weight,
                   (cur - self.last_seen[i]) / max(1, cur)]
        return torch.FloatTensor(st).unsqueeze(0)

    # --- Eval ---
    def test_global_model(self):
        import torch
        self.global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                _, pred = torch.max(self.global_model(x).data, 1)
                total += y.size(0); correct += (pred == y).sum().item()
        acc = 100.0 * correct / total
        self.performance_history.append(acc)
        return acc

    def test_backdoor_asr(self, trigger: BackdoorTrigger):
        import torch
        self.global_model.eval()
        total = success = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                tx = torch.stack([trigger.apply_trigger(xi) for xi in x])
                _, pred = torch.max(self.global_model(tx).data, 1)
                success += (pred == trigger.target_label).sum().item()
                total += y.size(0)
        asr = 100.0 * success / total
        self.asr_history.append(asr)
        return asr

    # --- Anomaly / trust / beliefs ---
    def _cosine_component(self, update, all_updates):
        import torch, torch.nn.functional as F
        global_direction = {k: torch.zeros_like(v) for k, v in update.items()}
        for other in all_updates:
            for k in update: global_direction[k] += other[k]
        cos_sim, cnt = 0.0, 0
        for k in update:
            if update[k].dim() > 1:
                u = update[k].flatten(); g = global_direction[k].flatten()
                if torch.norm(g) > 1e-10:
                    cos_sim += F.cosine_similarity(u, g, dim=0).item(); cnt += 1
        return 1.0 - (cos_sim / max(1, cnt))

    def _magnitude_component(self, cid, update):
        import torch, numpy as np
        mag = torch.norm(torch.cat([update[k].flatten() for k in update])).item()
        hist = self.client_update_history[cid]
        if len(hist) > 0:
            avg, std = np.mean(hist), np.std(hist) + 1e-6
            z = abs(mag - avg) / std
            comp = min(1.0, z / 3.0)
        else:
            comp = 0.0
        if len(hist) >= 5: hist.pop(0)
        hist.append(mag)
        return comp

    def _impact_component(self, update):
        import torch
        impact = 0.0
        state = self.global_model.state_dict()
        for k in update:
            if k in state:
                rel = torch.norm(update[k] - state[k]) / (torch.norm(state[k]) + 1e-6)
                impact += min(1.0, rel.item())
        return impact / len(update)

    def compute_anomaly_score(self, cid, update, all_updates):
        if not self.use_anomaly_detection:
            return 0.0
        comps, used = 0.0, 0
        if self.signal_budget in ["full", "no_validation", "directional"]:
            comps += self._cosine_component(update, all_updates); used += 1
        if self.signal_budget in ["full", "no_validation"]:
            comps += self._magnitude_component(cid, update); used += 1
        if self.signal_budget == "full":
            comps += self._impact_component(update); used += 1
        return min(1.0, comps / max(1, used))

    def bayesian_update(self, prior, likelihood, evidence=None):
        if not self.use_belief_update: return prior
        num = likelihood * prior
        if evidence is None:
            evidence = likelihood * prior + (1 - likelihood) * (1 - prior)
        return num / max(evidence, 1e-6)

    def compute_contribution_score(self, cid, acc_before, acc_after, client_weight):
        base = max(0.0, acc_after - acc_before)
        weighted = base / max(0.01, client_weight)
        return min(1.0, weighted)

    def update_beliefs_and_trust(self, selected, client_weights, updates, old_acc, new_acc):
        cur = len(self.performance_history)
        for cid in selected: self.last_seen[cid] = cur

        anomaly = [self.compute_anomaly_score(cid, updates[i], updates) for i, cid in enumerate(selected)]
        contrib = [self.compute_contribution_score(cid, old_acc, new_acc, client_weights[i]) for i, cid in enumerate(selected)]
        lam, eta = self.config.get("lambda_param", 0.3), self.config.get("eta_param", 0.2)

        for i, cid in enumerate(selected):
            t_old = self.clients[cid].trust_score
            t_new = t_old * (1 - lam * anomaly[i] + eta * contrib[i])
            self.clients[cid].trust_score = float(np.clip(t_new, 0.01, 1.0))

            comp = 0.7 * anomaly[i] + 0.3 * (1 - contrib[i])
            b_old = self.belief[cid]
            post = self.bayesian_update(b_old, comp)
            self.belief[cid] = float(np.clip(0.8 * post + 0.2 * b_old, 0.01, 0.99))

    # --- Actions / DQN ---
    def apply_actions(self, selected, actions):
        for i, cid in enumerate(selected):
            if actions[i] == 1:
                red = 0.5 * self.belief[cid]
                self.clients[cid].aggregation_weight *= (1.0 - red)
            elif actions[i] == 2:
                inc = 0.2 * (1.0 - self.belief[cid])
                self.clients[cid].aggregation_weight *= (1.0 + inc)
            self.clients[cid].aggregation_weight = float(np.clip(self.clients[cid].aggregation_weight, 0.01, 2.0))

    def select_action(self, state, client_id):
        import torch, random, math
        if not self.use_dqn: return random.randint(0, self.action_dim - 1)
        if random.random() < self.epsilon: return random.randint(0, self.action_dim - 1)
        with torch.no_grad(): return int(self.policy_net(state).max(1)[1].item())

    def update_dqn(self):
        import math, torch, torch.nn.functional as F, torch.nn as nn
        if len(self.replay) < self.batch_size: return
        s,a,r,ns,d = self.replay.sample(self.batch_size)
        q = self.policy_net(s).gather(1, a.unsqueeze(1))
        nq = self.target_net(ns).max(1)[0].detach()
        target = r + (1 - d) * self.gamma * nq
        loss = F.smooth_l1_loss(q, target.unsqueeze(1))
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.opt.step()
        self.train_steps += 1
        if self.train_steps % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1.0 * self.train_steps / self.eps_decay)

    # --- Aggregation / reward ---
    def aggregate_models(self, updates, client_ids):
        import torch, numpy as np
        if self.use_trust_scores:
            total = sum(self.clients[c].aggregation_weight * self.clients[c].trust_score for c in client_ids)
            weights = [(self.clients[c].aggregation_weight * self.clients[c].trust_score)/total for c in client_ids]
        else:
            total = sum(self.clients[c].aggregation_weight for c in client_ids)
            weights = [self.clients[c].aggregation_weight/total for c in client_ids]

        new_state = {}
        for k in updates[0]:
            stacked = torch.stack([u[k] for u in updates], dim=0)
            wt = torch.tensor(weights).float().view(-1, *([1] * (stacked.dim() - 1)))
            new_state[k] = torch.sum(stacked * wt, dim=0)
        self.global_model.load_state_dict(new_state)
        return weights

    def calculate_reward(self, old_acc, new_acc, selected_clients, actions, malicious_clients, asr=None):
        model_perf = new_acc - old_acc
        trust_acc = 0
        for i, cid in enumerate(selected_clients):
            mal = cid in malicious_clients
            if (actions[i] == 1 and mal) or (actions[i] == 2 and not mal) or (actions[i] == 0 and not mal):
                trust_acc += 1
            else:
                trust_acc -= 1
        trust_acc /= len(selected_clients)
        defense_cost = sum(1 for a in actions if a != 0) / len(actions)
        asr_penalty = 0.0
        if asr is not None and len(self.asr_history) > 1:
            asr_penalty = (asr - self.asr_history[-2]) / 100.0
        alpha = self.config["alpha"]; beta = self.config["beta"]
        gamma_cost = self.config["gamma_cost"]; delta = self.config.get("delta", 0.5)
        reward = alpha*model_perf + beta*trust_acc - gamma_cost*defense_cost - delta*asr_penalty
        self.rewards_history.append(reward)
        return reward
