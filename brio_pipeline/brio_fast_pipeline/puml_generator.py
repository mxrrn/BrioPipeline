"""
Generate a PlantUML instance diagram from 3D detections and connections.

Output format matches the ground-truth PUML files in the dataset:

    @startuml
    object "blwo11_1 : Component" as blwo111
    object "nu_1 : Component" as nu1
    blwo111 -- nu1
    @enduml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from triangulator import Detection3D


def _alias(instance_id: str) -> str:
    """PlantUML object alias: strip underscores and hyphens."""
    return instance_id.replace("_", "").replace("-", "")


def generate_puml(instances: list[Detection3D],
                  edges: list[tuple[str, str]],
                  title: str | None = None) -> str:
    """
    Build the PlantUML source string.

    Args:
        instances : ordered list of Detection3D
        edges     : list of (instance_id_A, instance_id_B) connection pairs
        title     : optional diagram title

    Returns PlantUML source as a string.
    """
    lines = ["@startuml"]
    if title:
        lines.append(f"title {title}")
    lines.append("")

    for inst in instances:
        alias = _alias(inst.instance_id)
        lines.append(f'object "{inst.instance_id} : Component" as {alias}')

    if edges:
        lines.append("")
        for a, b in edges:
            lines.append(f"{_alias(a)} -- {_alias(b)}")

    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)


def save_puml(puml_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(puml_text)
    print(f"[PUML] Saved → {output_path}")
