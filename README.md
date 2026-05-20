# From Multi-View 2D to 3D Object Detection and Structure Generation in PlantUML

**Bachelor's Thesis:** 3D Object Detection from Multi-View 2D Input Images of Multi-Component Constructions and Knowledge-Aware Structure Diagram Generation in PlantUML

## Problem

BRIO construction toys are assembled from discrete physical components (bolts, nuts, screws, plates, wheels, etc.) that connect through typed attachment points. Given a photograph of a completed construction, the goal is to automatically infer its full structural topology — which components are present, what slots they expose, and how those slots are connected — and represent that structure as a PlantUML instance diagram.

## Domain Model

```
Construction
  └── has one or many ConnectionConfiguration(s)
        └── has one or more Connection(s)
              └── is formed by exactly 2 Slot(s)
                    └── belongs to a Component
```

### Definitions

| Term | Definition |
|------|-----------|
| **Component** | A physical BRIO part (e.g., bolt, nut, plate, wheel). Has one or more slots. |
| **Slot** | A typed attachment point on a component (e.g., opening, pin, thread). |
| **Connection** | A link formed by exactly two slots joining together (e.g., a pin inserted into an opening). |
| **Connection Configuration** | A group of one or more connections that share a common joint point. |
| **Construction** | The complete physical assembly, composed of one or many connection configurations. |

### Component Types (30 known)

| Abbreviation | Part | Abbreviation | Part |
|---|---|---|---|
| `bo` | Bolt | `ro lo` | Roller |
| `nu` | Nut | `rome` | Rod medium |
| `pl` | Plug | `rosm` | Rod small |
| `sl` | Sleeve | `sclo` | Screw long |
| `wa` | Washer | `scme` | Screw medium |
| `ti` | Tire | `scsm` | Screw small |
| `no` | Nose/Nozzle | `whre` | Wheel red |
| `whwh` | Wheel white | `stwo3`-`stwo9` | Star wheel (sizes 3-9) |
| `blwo11` | Block w/ 1x1 openings | `plwo21`-`plwo53` | Plate w/ openings (various) |
| `plpl53` | Plate/Plug 5x3 | `stpl5` | Step plate 5 |

### Slot Types (9 known)

| Type | Meaning | Used By |
|---|---|---|
| `{op}` | Opening | Plates, blocks, star wheels, sleeves, noses |
| `{pi}` | Pin | Bolts |
| `{th}` | Thread | Nuts |
| `{thjo}` | Thread joint | Screws, rods, rollers |
| `{sh}` | Shaft | Plugs |
| `{slsh}` | Slot shaft | Plugs (variant) |
| `{slpi}` | Slot pin | Bolts (variant) |
| `{gr}` | Grip/Groove | Wheels |
| `{to}` | Torus | Tires |

## Data

~150 annotated BRIO construction samples, each containing:

| File | Description |
|------|-------------|
| `Construction.jpg` | Photograph of the physical construction |
| `InstanceDiagramSN.puml` | Human-readable PlantUML instance diagram with component names and slot types |
| `InstanceDiagramSN_AI.puml` | Anonymized PlantUML diagram (all names replaced with sequential integer IDs) |
| `InstanceDiagramSN.png` | Rendered human-readable diagram |
| `InstanceDiagramSN_AI.png` | Rendered anonymized diagram |
| `Mapping.drawio` | Visual mapping between photo regions and abstract components |
| `Mapping.png` | Rendered mapping image |

### Naming Conventions in PlantUML

- **Component**: `abbreviation_instanceNum` (e.g., `stwo3_1`, `bo_2`)
- **Slot**: `component&{slotType}_slotNum` (e.g., `stwo3_1&{op}_2`)
- **Connection**: `slotA#slotB` (e.g., `stwo3_1&{op}_1#bo_1&{pi}_1`)
- **ConnectionConfig**: `~connection1 * connection2 * ...~` (grouped by shared joint)

### Real Example (Sample 55 — 4 components)

A construction with a plug, bolt, washer, and sleeve:

```
Components: pl_1, bo_1, wa_1, sl_1
Slots:      pl_1&{slsh}_1, bo_1&{pi}_1, bo_1&{slpi}_1, wa_1&{op}_1, sl_1&{op}_1

Connections:
  pl_1&{slsh}_1 # bo_1&{slpi}_1    (plug shaft into bolt slot-pin)
  bo_1&{pi}_1   # wa_1&{op}_1      (bolt pin into washer opening)
  bo_1&{pi}_1   # sl_1&{op}_1      (bolt pin into sleeve opening)

Connection Configurations:
  CC1: pl_1&{slsh}_1#bo_1&{slpi}_1
  CC2: bo_1&{pi}_1#wa_1&{op}_1 * bo_1&{pi}_1#sl_1&{op}_1
```

### Complexity Range

- **Smallest**: 2 components, 2 slots, 1 connection, 1 ConnectionConfig
- **Largest**: 10+ components, 18+ slots, 13+ connections, 4+ ConnectionConfigs
- **Typical**: 4-7 components, 5-11 slots, 3-9 connections, 2-3 ConnectionConfigs

## Approach

### Input
- **Multi-view 2D images** of the construction (currently 1 photo per sample)
- **PlantUML instance diagrams** as ground-truth structural labels (training only)

### Output
- **Predicted connection topology**: which components are present, their slots, and how they connect

### Method (Under Investigation)

Investigating whether the **PETR** (Position Embedding Transformation) architecture — designed for multi-view 3D object detection — can be adapted for structural relationship prediction:

1. Backbone CNN extracts 2D features from input image(s)
2. Position embedding transforms 2D features into 3D-aware representations
3. Decoder predicts structural relationships instead of bounding boxes

The key question: can a pipeline built for detecting 3D objects (cars, pedestrians) be repurposed to detect **structural connections** between construction components?

## References

1. Liu et al. (2022) — *PETR: Position Embedding Transformation for Multi-View 3D Object Detection*
2. Chen et al. (2023) — *Viewpoint Equivariance for Multi-View 3D Object Detection*
3. Yang & Wang (2019) — *Learning Relationships for Multi-View 3D Object Recognition*
