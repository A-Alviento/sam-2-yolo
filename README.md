# Sam2Yolo-Tool

This project leverages the capabilities of the Segment Anything Model (SAM) to automate the process of creating segmentation datasets for YOLOv8.

## Description

As of May 31, 2023, the capabilities of this tool include:

- Accepts a video input (file path) and outputs frames to be used as a dataset. 
- Splits the dataset into an 80/20 train/test set.
- Utilizes the SAM model to obtain the segmentation masks from these images.
- Allows the user to specify desired labels.
- Prompts the user to input labels for each mask.
- Stores the polygon coordinates of these masks.
- Automates the process of creating the `.txt` files for YOLO segmentation dataset format, along with the yaml file.

### Future Plans:

- Convert to a script-based program.
- Add the ability to filter masks based on relative size or actual size.
- Further optimization of functions.
- Develop a comprehensive GUI.

## Getting Started

This project has been tested and run using Python 3.10.11 on a Windows PC equipped with an Nvidia RTX 3080 Ti, using CUDA=11.8.

### Prerequisites and Installation

1. Start by cloning the repository:
```bash
git clone https://github.com/A-Alviento/sam-2-yolo
cd sam-2-yolo
pip install -e .
```

2. Install the required packages:
- numpy 
- torch 
- matplotlib 
- opencv-python  
- ultralytics  
- pillow
- scikit-learn ipython 
- pyyaml


## Running the Program

1. Open the `sam2yolo-tool.ipynb` notebook and follow the instructions.
2. A powerful Nvidia GPU is recommended for optimal performance. The tool currently exhibits slow performance with the Apple M1 CPU and may encounter bugs when MPS is utilized.

## Acknowledgements

This project was inspired by and made possible through the following resources:

- [Segment Anything Model (SAM) by Facebook Research](https://github.com/facebookresearch/segment-anything/blob/main/README.md)
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
