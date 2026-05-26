# Thesis Title & Abstract

---

## Title Alternatives

**Option 1** 
> Multi-View Component Detection from 2D-RGB images and Structural Diagram Generation Using SAM-Assisted YOLOv8 and Rule-Based PlantUML Architecture

**Option 2** 
> From Multi-View Images to Instance Diagrams: A Multi-View Detection Pipeline for Structural Relationship Inference in Small Assemblies

**Option 3**
> Component Detection and Connection Inference of Small Constructions from Multi-View 2D Images

---

## Abstract

Recognizing the full structural 3-dimensional architecture of a physical assembly from multi-view images is a challenging object detection task that requires both reliable component detection and the inference of how detected parts connect to one another despite occlusion. This research addresses that challenge in the context of BRIO (toy?) constructions: given a set of multi-view RGB images taken at multiple elevation angles around a construction, the goal is to automatically produce a PlantUML instance diagram that represents every component and every inter-component (or component to component) connection.
 
The proposed pipeline consists of three stages. First, a Segment Anything Model (SAM) is used offline as an annotation accelerator: we utilize the ground-truth component manifest extracted from existing PlantUML diagrams that represent the component to slot to connection relationship of a structure, followed by SAM-generated instance masks that are matched to known component classes, yielding a labeled dataset of the annotated samples without manual labeling. Second, a YOLOv8n detector is trained on this dataset and applied to all available views per construction; a majority-voting aggregation scheme across approximately 78 images per sample consolidates per-view detections into a robust component list while respecting a fast paced inference constraint. Third, a rule-based slot-compatibility engine translates the detected component list into valid connections using the fixed slot-type vocabulary of the BRIO domain, and serializes the result as a PlantUML diagram.
 
Experiments on the 150-sample dataset demonstrate (result here.....).
