#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def load_generator():
    path = Path(__file__).with_name("generate-six-pendulum-mjcf.py")
    spec = importlib.util.spec_from_file_location("six_pendulum_mjcf_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load MJCF generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_float_pair(value: str) -> tuple[float, float]:
    left, right = value.split()
    return float(left), float(right)


def audit_file(path: Path, profile: str, expected: dict) -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    root = ET.parse(path).getroot()
    option = root.find("option")
    actuator = root.find("actuator/motor")
    slider = root.find(".//joint[@name='slider']")
    joints = root.findall(".//joint")
    hinge_joints = [joint for joint in joints if joint.attrib.get("type") == "hinge"]
    rods = [geom for geom in root.findall(".//geom") if geom.attrib.get("name", "").startswith("rod_")]
    lengths = []
    masses = []
    for geom in rods:
        z0, z1 = float(geom.attrib["fromto"].split()[2]), float(geom.attrib["fromto"].split()[5])
        lengths.append(round(z1 - z0, 10))
        masses.append(float(geom.attrib.get("mass", "nan")))

    ctrlrange = parse_float_pair(actuator.attrib["ctrlrange"]) if actuator is not None else (None, None)
    failures = []
    if abs(float(option.attrib["timestep"]) - expected["timestep"]) > 1e-12:
        failures.append("timestep")
    if len(hinge_joints) != expected["links"]:
        failures.append("hinge-count")
    if any(abs(float(joint.attrib.get("damping", "nan")) - expected["hingeDamping"]) > 1e-12 for joint in hinge_joints):
        failures.append("hinge-damping")
    if any(abs(length - expected["linkLength"]) > 1e-9 for length in lengths):
        failures.append("link-length")
    if expected["linkMass"] is not None and any(abs(mass - expected["linkMass"]) > 1e-9 for mass in masses):
        failures.append("link-mass")
    if ctrlrange != (-expected["forceCap"], expected["forceCap"]):
        failures.append("force-cap")
    slider_limited = slider.attrib.get("limited") if slider is not None else None
    if bool(expected["cartLimited"]) != (slider_limited == "true"):
        failures.append("cart-limit-mode")
    if model.nq != expected["links"] + 1 or model.nv != expected["links"] + 1 or model.nu != 1:
        failures.append("compiled-dimensions")

    return {
        "path": str(path),
        "profile": profile,
        "links": expected["links"],
        "compiled": {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)},
        "timestep": float(option.attrib["timestep"]),
        "gravity": option.attrib["gravity"],
        "hingeDamping": sorted({float(joint.attrib.get("damping", "nan")) for joint in hinge_joints}),
        "linkLengths": sorted(set(lengths)),
        "linkMasses": sorted(set(masses)),
        "cartLimited": slider_limited == "true",
        "forceRange": list(ctrlrange),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }


def run_audit(profile: str, out_dir: Path | None = None) -> dict:
    generator = load_generator()
    config = generator.PROFILE_CONFIGS[profile]
    expected_base = {
        "timestep": float(config["timestep"]),
        "hingeDamping": float(config["hinge_damping"]),
        "linkLength": 0.5 if profile == "thread-matched" else None,
        "linkMass": config["link_mass"],
        "forceCap": float(config["force_cap"]),
        "cartLimited": bool(config["cart_limited"]),
    }

    with tempfile.TemporaryDirectory() as tmp:
        target = out_dir or Path(tmp)
        generated = []
        for links in range(1, int(config["max_links"]) + 1):
            generated.append(generator.write_model(links, target, profile))
        audits = []
        for path in generated:
            links = int(path.stem.split("_")[1])
            expected = {**expected_base, "links": links}
            if expected["linkLength"] is None:
                expected["linkLength"] = round(generator.link_length(0, profile), 10)
            audits.append(audit_file(path, profile, expected))

    failed = [item for item in audits if item["status"] != "pass"]
    return {
        "schema": "six-pendulum-mjcf-profile-audit-v1",
        "profile": profile,
        "status": "pass" if not failed else "fail",
        "filesAudited": len(audits),
        "failedFiles": len(failed),
        "canonical": {
            "maxLinks": int(config["max_links"]),
            "linkLength": 0.5 if profile == "thread-matched" else "profile-dependent",
            "linkMass": config["link_mass"],
            "hingeDamping": float(config["hinge_damping"]),
            "forceCap": float(config["force_cap"]),
            "cartLimited": bool(config["cart_limited"]),
            "dt": float(config["timestep"]),
        },
        "audits": audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile-audit six-pendulum MJCF generator profiles.")
    parser.add_argument("--profile", default="thread-matched", choices=["current", "thread-matched"])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--write-result", type=Path)
    args = parser.parse_args()

    result = run_audit(args.profile, args.out_dir)
    if args.write_result:
        args.write_result.parent.mkdir(parents=True, exist_ok=True)
        args.write_result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
