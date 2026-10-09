import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path("scripts/modal-train-six-pendulum-pezzza-chain.py")
OUTPUT_PATH = Path(
    "local-artifacts/outputs/training-checkpoints/pezzza-chain-2link-local-angle-mini.json"
)
INITIAL_POLICY_PATH = Path("app/ailab/six-pendulum-cartpole/sixPendulumPolicy.json")


def load_module():
    spec = importlib.util.spec_from_file_location("pezzza_chain", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = load_module()
    result = module.train_policy.local(True, 426410, 2, 60, 256, 4, INITIAL_POLICY_PATH.read_text())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(result)
    root = json.loads(result)
    print(
        json.dumps(
            {
                "path": str(OUTPUT_PATH),
                "algorithm": root["algorithm"],
                "device": root["training"]["device"],
                "gpu": root["training"]["gpu"],
                "elapsedSeconds": root["training"]["elapsedSeconds"],
                "validation": root["training"]["validation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
