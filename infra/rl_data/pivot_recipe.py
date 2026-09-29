"""Generate separate pinned student/domain recipes for profiling, PivotRL, and same-data SFT."""

import argparse
from pathlib import Path

import yaml

STUDENTS = {
    "snowball": ("open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92", "fdc3c3c5fff489d3fb04dcbc2387c179b1bf3fab"),
    "grug": ("open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21", "b8c07f7df1df65525abbfdbcd1572318ba11c42f"),
}
TEMPLATE = Path(__file__).resolve().parents[2] / "cloud/iris/configs/snowball_pivotrl_split64.yaml"


def recipe(student: str, mode: str, train_data: str, validation_data: list[str], kl_coefficient: float,
           template: Path = TEMPLATE) -> dict:
    """Pin initialization/reference identity and preserve the split64 engine geometry."""
    raw = yaml.safe_load(template.read_text())
    model, revision = STUDENTS[student]
    raw["pivot"]["mode"] = mode
    raw["trainer"]["policy"]["model"] = {
        "path": model, "revision": revision,
    }
    raw["generator"]["trajectory_retention"]["model_source_identity"] = revision
    raw["trainer"]["algorithm"]["kl_loss_coef"] = kl_coefficient
    raw["data"]["train_data"] = [train_data]
    raw["data"]["val_data"] = validation_data
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=STUDENTS, required=True)
    parser.add_argument("--mode", choices=("profile", "pivotrl", "sft"), required=True)
    parser.add_argument("--train-data", required=True, help="candidates.parquet for profiling; frozen train.parquet otherwise")
    parser.add_argument("--validation-data", nargs="+", required=True)
    parser.add_argument("--kl-coefficient", type=float, choices=(0, .001, .01), default=.001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--smoke", action="store_true", help="One trainer step with the full 64 x 16 geometry")
    args = parser.parse_args()
    raw = recipe(args.student, args.mode, args.train_data, args.validation_data, args.kl_coefficient, args.template)
    if args.smoke:
        if args.mode == "profile":
            parser.error("Bound a profiling smoke with a separate small candidates artifact")
        raw["trainer"].update(max_steps=1, eval_before_train=False, eval_interval=-1, ckpt_interval=1)
    with args.output.open("x") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False)


if __name__ == "__main__":
    main()
