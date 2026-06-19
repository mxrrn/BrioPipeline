"""Parse PUML instance diagram files to extract the component manifest for a sample."""
import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Component:
    cls: str        # e.g. "blwo11"
    instance_id: str  # e.g. "blwo11_1"
    slots: list = field(default_factory=list)  # filled after parse


@dataclass
class SampleManifest:
    sample_id: int
    components: list          # list[Component]
    thjo_pairs: list = field(default_factory=list)
    # thjo_pairs: list of (rod_instance_id, [through_instance_id, ...])
    # A "thjo" (through-joint) rod passes through each listed {op} component.

    @property
    def n_components(self):
        return len(self.components)

    @property
    def class_list(self):
        return [c.cls for c in self.components]


def parse_puml(puml_path: Path) -> SampleManifest:
    """
    Parse InstanceDiagramSN.puml and return the component manifest.
    Only reads the human-readable PUML (not _AI.puml).
    """
    text = Path(puml_path).read_text()

    # Match: object "blwo11_1 : Component" as blwo111
    comp_pattern = re.compile(r'object "(\w+)_(\d+)\s*:\s*Component"')
    components = []
    for cls, num in comp_pattern.findall(text):
        components.append(Component(cls=cls, instance_id=f"{cls}_{num}"))

    # Parse through-joint connections: identify which rod goes through which component.
    # Connection names look like: "blwo11_1&{op}_1#rome_1&{thjo}_1 : Connection"
    conn_pattern = re.compile(
        r'object\s+"(\w+_\d+)&\{(\w+)\}_\d+#(\w+_\d+)&\{(\w+)\}_\d+\s*:\s*Connection"'
    )
    thjo_map: dict[str, set] = {}
    for c1, s1, c2, s2 in conn_pattern.findall(text):
        if s2 == "thjo" and s1 == "op":
            thjo_map.setdefault(c2, set()).add(c1)
        elif s1 == "thjo" and s2 == "op":
            thjo_map.setdefault(c1, set()).add(c2)
    thjo_pairs = [(rod, sorted(through)) for rod, through in thjo_map.items()]

    # Extract sample number from filename (InstanceDiagramS113.puml)
    fname = Path(puml_path).stem  # "InstanceDiagramS113"
    match = re.search(r'S(\d+)', fname)
    sample_id = int(match.group(1)) if match else -1

    return SampleManifest(sample_id=sample_id, components=components, thjo_pairs=thjo_pairs)


def find_puml(sample_id: int) -> Path:
    """Locate the PUML file for a given sample number."""
    from config import BATCH_112_150
    folder_name = f"Sample_{sample_id}_InstanceDiagram"
    puml_name   = f"InstanceDiagramS{sample_id}.puml"
    candidate   = BATCH_112_150 / folder_name / puml_name
    if candidate.exists():
        return candidate
    # Search all batch dirs as fallback
    from config import CONSTRUCTIONS
    for batch_dir in CONSTRUCTIONS.iterdir():
        p = batch_dir / folder_name / puml_name
        if p.exists():
            return p
    raise FileNotFoundError(f"PUML not found for sample {sample_id}")


if __name__ == "__main__":
    import sys
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else 113
    puml = find_puml(sid)
    manifest = parse_puml(puml)
    print(f"Sample {manifest.sample_id}: {manifest.n_components} components")
    for c in manifest.components:
        print(f"  {c.instance_id}")
