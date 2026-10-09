#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def add(elem, tag, **attrs):
    return ET.SubElement(elem, tag, {key: str(value) for key, value in attrs.items()})


def build_model(links: int) -> ET.ElementTree:
    if links < 1 or links > 6:
        raise ValueError("links must be between 1 and 6")

    root = ET.Element("mujoco", model=f"cartpole_{links}_link")
    add(root, "compiler", angle="radian", coordinate="local")
    add(root, "option", timestep="0.0025", gravity="0 0 -9.8")

    default = add(root, "default")
    add(default, "joint", damping="0", frictionloss="0", armature="0")
    add(default, "geom", density="650", friction="0.00001 0 0", contype="0", conaffinity="0")

    asset = add(root, "asset")
    add(asset, "material", name="cart", rgba="0.95 0.2 0.16 1")
    add(asset, "material", name="link", rgba="0.1 0.65 0.9 1")
    add(asset, "material", name="track", rgba="0.75 0.75 0.72 1")

    world = add(root, "worldbody")
    add(world, "light", pos="0 -3 4", dir="0 0 -1")
    add(world, "geom", name="rail", type="box", pos="0 0 -0.04", size="3 0.035 0.025", material="track")

    cart = add(world, "body", name="cart", pos="0 0 0")
    add(cart, "joint", name="slider", type="slide", axis="1 0 0", limited="true", range="-2.4 2.4")
    add(cart, "geom", name="cart_box", type="box", size="0.16 0.10 0.08", material="cart")

    parent = cart
    z_offset = 0.08
    for index in range(links):
        length = 0.62 + index * 0.035
        radius = 0.018
        color = "link"
        body = add(parent, "body", name=f"link_{index + 1}", pos=f"0 0 {z_offset:.5f}")
        add(body, "joint", name=f"hinge_{index + 1}", type="hinge", axis="0 1 0", damping="0", frictionloss="0")
        add(
            body,
            "geom",
            name=f"rod_{index + 1}",
            type="capsule",
            fromto=f"0 0 0 0 0 {length:.5f}",
            size=f"{radius:.5f}",
            material=color,
        )
        add(body, "site", name=f"tip_{index + 1}", pos=f"0 0 {length:.5f}", size="0.018")
        parent = body
        z_offset = length

    actuator = add(root, "actuator")
    add(actuator, "motor", name="cart_force", joint="slider", gear="1", ctrllimited="true", ctrlrange="-240 240")

    sensor = add(root, "sensor")
    add(sensor, "jointpos", name="cart_x", joint="slider")
    add(sensor, "jointvel", name="cart_v", joint="slider")
    for index in range(links):
        add(sensor, "jointpos", name=f"theta_{index + 1}", joint=f"hinge_{index + 1}")
        add(sensor, "jointvel", name=f"omega_{index + 1}", joint=f"hinge_{index + 1}")

    return ET.ElementTree(root)


def indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def write_model(links: int, out_dir: Path) -> Path:
    tree = build_model(links)
    indent(tree.getroot())
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cartpole_{links}_link.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MuJoCo MJCF files for one through six link cartpole.")
    parser.add_argument("--max-links", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=Path("app/ailab/six-pendulum-cartpole/mjcf"))
    args = parser.parse_args()

    if args.max_links < 1 or args.max_links > 6:
        raise SystemExit("--max-links must be between 1 and 6")

    for links in range(1, args.max_links + 1):
        print(write_model(links, args.out_dir))


if __name__ == "__main__":
    main()
