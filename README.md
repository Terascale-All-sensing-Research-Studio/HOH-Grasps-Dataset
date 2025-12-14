# HOH-Grasps Dataset

## Contents

[Description](#description)

[Dataset Location](#dataset-location)

[Contributors](#contributors)

[Requirements](#requirements)

[Installation](#installation)

[Citation](#citation)

## Description

We present HOH-Grasps, derived from the [HOH dataset](https://tars-home.github.io/hohdataset/). The original HOH dataset comprises a large-scale collection of 3D point cloud data from 2,720 handover interactions involving 136 distinct objects and 20 unique pairs of givers and receivers. HOH captures data via multi-view markerless motion capture technology and records detailed handover interactions where a giver picks up an object from a table and transfers it to a receiver.

Our HOH-Grasps dataset extracts segmented point clouds of objects, giver hands, and receiver hands from the HOH dataset and aligns them at two critical frames: the moment right before human grasping and the moment of handover. We define a standardized coordinate system different from the original HOH dataset, placing the origin above the scene with the Z-axis oriented downward toward the object center, aligning with typical camera viewpoints used in robotic grasping experiments. For simulating realistic scenarios, each object point cloud is combined with a dummy planar surface beneath it to emulate a tabletop setting, along with binary masks distinguishing objects from the dummy surface. Additionally, we provide ground truth centroids of objects and giver/receiver hands aligned at both frames. Corresponding transformations are also extracted.

Grasp labels for each object are generated using an analytical method proposed by [GraspNet](https://graspnet.net/). Grasp points are sampled across the surface of each 3D mesh, and 14,400 grasps per sample point. Collision indicators, grasp scores, and tolerance labels are computed for each grasp.

## Dataset Location

[https://huggingface.co/datasets/tars-home/HOH-Grasps](https://huggingface.co/datasets/tars-home/HOH-Grasps)

## Contributors

Xinchao Song, Ava Megyeri, Noah Wiederhold, Sean Banerjee, and Natasha Kholgade Banerjee

[Terascale All-sensing Research Studio](https://tars-home.github.io)

## Requirements

- Python 3.8
- NumPy
- Open3d
- graspnetAPI

## Installation

Get the code:

```bash
git clone https://github.com/Terascale-All-sensing-Research-Studio/HOH-Grasps-Dataset.git
cd HOH-Grasps-Dataset
```

Greate and activate a virtual environment using virtualenv:

```bash
virtualenv -p python3.8 .venv
source .venv/bin/activate
```

Or using Conda:

```bash
conda create -n hoh_grasps_dataset python=3.8 -y
conda activate hoh_grasps_dataset
```

Install packages via Pip:

```bash
pip install -r requirements.txt
```

## Citation
Please cite our paper in your publications if it helps your research:
```
@inproceedings{song2025higrasp,
  title={HI-Grasp: Human-Inspired Grasp Network for Intuitive and Stable Robotic Grasp},
  author={Song, Xinchao and Megyeri, Ava and Wiederhold, Noah and Banerjee, Sean and Kholgade Banerjee, Natasha},
  booktitle={2025 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)},
  year={2025},
  organization={IEEE}
}
```
