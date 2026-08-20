"""Run the reproducible BONSAI benchmark/ablation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.benchmarking import ToyBenchmarkConfig, run_toy_suite
from src.utils.real_benchmark import RealBenchmarkConfig, run_real_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("synthetic", "cifar100", "tinyimagenet"), default="synthetic")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("BONSAI", "EWC", "SI", "PackNet", "PNN"),
        default=("BONSAI", "EWC", "SI", "PackNet", "PNN"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/toy_benchmark"))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or dml/directml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--classes-per-task", type=int, default=2)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.35)
    parser.add_argument("--shared-encoder-updates", action="store_true")
    parser.add_argument("--adapter-rank", type=int, default=1)
    parser.add_argument("--encoder-learning-rate-scale", type=float, default=0.1)
    parser.add_argument("--rewire-strength", type=float, default=0.15)
    parser.add_argument("--max-frozen-fraction", type=float, default=0.65)
    parser.add_argument(
        "--route-strategy", choices=("entropy", "prototype", "hybrid"), default="prototype"
    )
    parser.add_argument(
        "--real-route-strategy",
        choices=("entropy", "prototype", "hybrid", "learned", "scaffold", "compatibility", "fused", "global_argmax", "local_energy", "global_direct", "cosine", "discovery", "evidence"),
        default="compatibility",
    )
    parser.add_argument("--task-adapter-rank", type=int, default=8)
    parser.add_argument("--adapter-residual-init-std", type=float, default=0.0)
    parser.add_argument("--replay-per-task", type=int, default=16)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--prototype-weight", type=float, default=1.0)
    parser.add_argument("--route-head-weight", type=float, default=1.0)
    parser.add_argument("--route-calibration-samples", type=int, default=256)
    parser.add_argument("--route-memory-samples", type=int, default=64)
    parser.add_argument("--route-head-epochs", type=int, default=3)
    parser.add_argument("--route-compatibility-epochs", type=int, default=3)
    parser.add_argument("--route-hidden-dim", type=int, default=64)
    parser.add_argument("--route-discovery-hidden-dim", type=int, default=32)
    parser.add_argument("--route-discovery-epochs", type=int, default=10)
    parser.add_argument("--route-evidence-epochs", type=int, default=3)
    parser.add_argument("--global-loss-weight", type=float, default=0.5)
    parser.add_argument("--global-replay-per-task", type=int, default=64)
    parser.add_argument("--global-replay-weight", type=float, default=0.5)
    parser.add_argument("--global-head-epochs", type=int, default=0)
    parser.add_argument("--shared-learning-rate-scale", type=float, default=0.1)
    parser.add_argument("--classifier-learning-rate-scale", type=float, default=1.0)
    parser.add_argument("--freeze-backbone-bn", action="store_true")
    parser.add_argument("--global-route-weight", type=float, default=1.0)
    parser.add_argument("--route-training-weight", type=float, default=0.0)
    parser.add_argument("--route-replay-per-task", type=int, default=16)
    parser.add_argument("--feature-replay-weight", type=float, default=0.0)
    parser.add_argument("--feature-replay-per-task", type=int, default=16)
    parser.add_argument("--local-replay-weight", type=float, default=0.0)
    parser.add_argument("--local-replay-per-task", type=int, default=16)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.dataset == "synthetic":
        records, summaries = run_toy_suite(
            ToyBenchmarkConfig(
                seeds=tuple(args.seeds),
                epochs_per_task=args.epochs,
                samples_per_class=args.samples_per_class,
                num_tasks=args.num_tasks,
                classes_per_task=args.classes_per_task,
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                noise=args.noise,
                shared_encoder_updates=args.shared_encoder_updates,
                adapter_rank=args.adapter_rank,
                encoder_learning_rate_scale=args.encoder_learning_rate_scale,
                rewire_strength=args.rewire_strength,
                max_frozen_fraction=args.max_frozen_fraction,
                route_strategy=args.route_strategy,
                replay_per_task=args.replay_per_task,
                replay_weight=args.replay_weight,
                output_dir=args.output_dir,
                use_wandb=args.wandb,
            )
        )
    else:
        records, summaries = run_real_suite(
            RealBenchmarkConfig(
                dataset=args.dataset,
                data_root=args.data_root,
                methods=tuple(args.methods),
                seeds=tuple(args.seeds),
                device=args.device,
                epochs_per_task=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                output_dir=args.output_dir,
                task_adapter_rank=args.task_adapter_rank,
                adapter_residual_init_std=args.adapter_residual_init_std,
                rewire_strength=args.rewire_strength,
                max_frozen_fraction=args.max_frozen_fraction,
                validation_fraction=args.validation_fraction,
                route_strategy=args.real_route_strategy,
                prototype_weight=args.prototype_weight,
                route_head_weight=args.route_head_weight,
                route_calibration_samples=args.route_calibration_samples,
                route_memory_samples=args.route_memory_samples,
                route_head_epochs=args.route_head_epochs,
                route_compatibility_epochs=args.route_compatibility_epochs,
                route_hidden_dim=args.route_hidden_dim,
                route_discovery_hidden_dim=args.route_discovery_hidden_dim,
                route_discovery_epochs=args.route_discovery_epochs,
                route_evidence_epochs=args.route_evidence_epochs,
                global_loss_weight=args.global_loss_weight,
                global_replay_per_task=args.global_replay_per_task,
                global_replay_weight=args.global_replay_weight,
                global_head_epochs=args.global_head_epochs,
                shared_learning_rate_scale=args.shared_learning_rate_scale,
                classifier_learning_rate_scale=args.classifier_learning_rate_scale,
                freeze_backbone_bn=args.freeze_backbone_bn,
                global_route_weight=args.global_route_weight,
                route_training_weight=args.route_training_weight,
                route_replay_per_task=args.route_replay_per_task,
                feature_replay_weight=args.feature_replay_weight,
                feature_replay_per_task=args.feature_replay_per_task,
                local_replay_weight=args.local_replay_weight,
                local_replay_per_task=args.local_replay_per_task,
                max_samples_per_class=args.max_samples_per_class,
                download=args.download,
            )
        )
    print(f"wrote {len(records)} runs and {len(summaries)} summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
