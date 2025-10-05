import argparse
import pickle

from flrl.fl_experiments import (
    get_default_config,
    run_enhanced_federated_learning,
    run_dirichlet_sweep,
    run_signal_budget_study,
)
from flrl.fl_plots import plot_enhanced_results, plot_dirichlet_sweep, plot_signal_budget

def main():
    p = argparse.ArgumentParser(description="DQN-POMDP Federated Learning Experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Baseline
    b = sub.add_parser("baseline", help="Run baseline backdoor experiment")
    b.add_argument("--rounds", type=int, default=50)
    b.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha")
    b.add_argument("--seed", type=int, default=256)
    b.add_argument("--out", type=str, default="baseline_metrics.pkl")

    # Dirichlet sweep
    d = sub.add_parser("dirichlet", help="Run Dirichlet non-IID sweep")
    d.add_argument("--alphas", type=float, nargs="+", default=[0.1,0.3,0.5,1.0,5.0])
    d.add_argument("--rounds", type=int, default=50)
    d.add_argument("--runs", type=int, default=2)
    d.add_argument("--out", type=str, default="dirichlet_sweep.pkl")

    # Signal budget
    s = sub.add_parser("signal", help="Run signal-budget study")
    s.add_argument("--rounds", type=int, default=50)
    s.add_argument("--seed",type=int,default=42)
    s.add_argument("--out", type=str, default="signal_budget_metrics.pkl")

    args = p.parse_args()

    if args.cmd == "baseline":
        cfg = get_default_config()
        metrics, server, clients = run_enhanced_federated_learning(
            config=cfg, num_rounds=args.rounds, attack_type="backdoor",
            alpha_dirichlet=args.alpha, seed=args.seed
        )
        plot_enhanced_results(metrics, save_path="baseline_backdoor_results.png")
        with open(args.out, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Saved baseline metrics to {args.out}")

    elif args.cmd == "dirichlet":
        df = run_dirichlet_sweep(alphas=args.alphas, num_rounds=args.rounds, runs_per_alpha=args.runs)
        plot_dirichlet_sweep(df)
        with open(args.out, "wb") as f:
            pickle.dump(df, f)
        print(f"Saved dirichlet sweep results to {args.out}")

    elif args.cmd == "signal":
        df = run_signal_budget_study(num_rounds=args.rounds,seed=args.seed)
        plot_signal_budget(df)
        with open(args.out, "wb") as f:
            pickle.dump(df, f)
        print(f"Saved signal-budget results to {args.out}")

if __name__ == "__main__":
    main()

