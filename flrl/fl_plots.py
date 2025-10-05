import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_enhanced_results(metrics, save_path=None):
    rounds = list(range(1, len(metrics["accuracy"]) + 1))
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    axes[0,0].plot(rounds, metrics["accuracy"], marker="o")
    axes[0,0].set_title("Global Model Accuracy"); axes[0,0].set_xlabel("Rounds"); axes[0,0].set_ylabel("Accuracy (%)"); axes[0,0].grid(True, alpha=0.3)

    if "asr" in metrics and len(metrics["asr"]) > 0:
        axes[0,1].plot(rounds[:len(metrics["asr"])], metrics["asr"], marker="s")
        axes[0,1].set_title("Attack Success Rate"); axes[0,1].set_xlabel("Rounds"); axes[0,1].set_ylabel("ASR (%)"); axes[0,1].grid(True, alpha=0.3)
    else:
        axes[0,1].axis("off")

    axes[0,2].plot(rounds, metrics["rewards"], marker="^")
    axes[0,2].set_title("DQN Reward History"); axes[0,2].set_xlabel("Rounds"); axes[0,2].set_ylabel("Reward"); axes[0,2].grid(True, alpha=0.3)

    belief_matrix = np.array(metrics["beliefs"])
    for i in range(min(10, belief_matrix.shape[1])):
        axes[1,0].plot(rounds, belief_matrix[:, i], alpha=0.7, label=f"C{i}")
    axes[1,0].set_title("Belief State Evolution"); axes[1,0].legend(fontsize="small", ncol=2); axes[1,0].grid(True, alpha=0.3)

    trust_matrix = np.array(metrics["trusts"])
    for i in range(min(10, trust_matrix.shape[1])):
        axes[1,1].plot(rounds, trust_matrix[:, i], alpha=0.7, label=f"C{i}")
    axes[1,1].set_title("Trust Score Evolution"); axes[1,1].legend(fontsize="small", ncol=2); axes[1,1].grid(True, alpha=0.3)

    axes[1,2].plot(rounds, metrics["roc_auc"], marker="o", label="ROC-AUC")
    axes[1,2].plot(rounds, metrics["pr_auc"], marker="s", label="PR-AUC")
    axes[1,2].set_title("Detection Performance"); axes[1,2].legend(); axes[1,2].grid(True, alpha=0.3)

    axes[2,0].plot(rounds, metrics["ece"], marker="d")
    axes[2,0].set_title("Expected Calibration Error"); axes[2,0].grid(True, alpha=0.3)

    if "asr" in metrics and len(metrics["asr"]) > 0:
        axes[2,1].scatter(metrics["accuracy"][:len(metrics["asr"])], metrics["asr"], alpha=0.6)
        axes[2,1].set_title("Accuracy vs ASR"); axes[2,1].grid(True, alpha=0.3)
    else:
        axes[2,1].axis("off")

    if len(metrics["beliefs"]) > 0:
        final_beliefs = metrics["beliefs"][-1]
        axes[2,2].hist(final_beliefs, bins=20, edgecolor="black", alpha=0.7)
        axes[2,2].set_title("Final Belief Distribution"); axes[2,2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()

def plot_dirichlet_sweep(df, save_path="dirichlet_sweep_results.png"):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    sns.boxplot(data=df, x="alpha", y="final_accuracy", ax=axes[0]); axes[0].set_title("Accuracy vs Non-IID Severity")
    sns.boxplot(data=df, x="alpha", y="final_asr", ax=axes[1]); axes[1].set_title("ASR vs Non-IID Severity")
    sns.boxplot(data=df, x="alpha", y="avg_roc_auc", ax=axes[2]); axes[2].set_title("Detection vs Non-IID Severity")
    sns.boxplot(data=df, x="alpha", y="final_ece", ax=axes[3]); axes[3].set_title("Calibration vs Non-IID Severity")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.show()

def plot_signal_budget(df, save_path="signal_budget_study.png"):
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df["signal_budget"]))
    width = 0.2
    ax.bar(x - 1.5*width, df["final_accuracy"], width, label="Accuracy (%)")
    ax.bar(x - 0.5*width, df["final_asr"], width, label="ASR (%)")
    ax.bar(x + 0.5*width, df["avg_roc_auc"]*100, width, label="ROC-AUC (×100)")
    ax.bar(x + 1.5*width, df["final_ece"]*100, width, label="ECE (×100)")
    ax.set_xticks(x); ax.set_xticklabels(df["signal_budget"])
    ax.set_title("Impact of Limited Observability"); ax.grid(True, alpha=0.3, axis="y")
    ax.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.show()
