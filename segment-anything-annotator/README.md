# segment-anything-annotator

## Installation
  1. Python>=3.8
  2. Download compatible [Pytorch](https://pytorch.org/)
  3. pip install -r requirements.txt

## Usage
### 1. Start the Annotation Platform

```
python annotator.py --app_resolution 1000,1600 --model_type vit_b --keep_input_size True --max_size 720
```
`--model_type`: vit_b, vit_l, vit_h

`--keep_input_size`: `True`: keep the origin image size for SAM; `False`: resize the input image to `--max_size` (save GPU memory)

### 2. Load the category list file if you want to annotate object categories.
Click the `Category File` on the top tool bar and choose your own one, such as the `categories.txt` in this repo. This is a .txt file with your label names separated by newline.

### 3. Specify the image and save folds
Click the 'Image Directory' on the top tool bar to specify the folder containing images (in .jpg or .png).
Click the 'Save Directory' on the top tool bar to specify the folder for saving the annotations. The annotations of each image will be saved as json file in the following format
```
[
  #object1
  {
      'label':<category>, 
      'group_id':<id>,
      'shape_type':'polygon',
      'points':[[x1,y1],[x2,y2],[x3,y3],...]
  },
  #object2
  ...
]
```

### 4. Load SAM model
Click the "Load SAM" on the top tool bar to load the SAM model. The model will be automatically downloaded at the first time. Please be patient. Or you can manually download the [models](https://github.com/facebookresearch/segment-anything#model-checkpoints) and put them in the root directory named `vit_b.pth`, `vit_l.pth` and `vit_h.pth`.

### 5. Annotating Functions
`Manual Polygons`: manually add masks by clicking on the boundary of the objects, just like the Labelme (Press right button and drag to draw the arcs easily).

`Point Prompt`: generate mask proposals with clicks. The mouse leftpress/rightpress represent positive/negative clicks respectively.
You can see several mask proposals below in the boxes: `Proposal1-4`, and you could choose one by clicking or shortcuts `1`,`2`,`3`,`4`.

`Box Prompt`: generate mask proposals with boxes.

`Accept`(shortcut:`a`): accept the chosen proposal and add to the annotation dock.

`Reject`(shortcut:`r`): reject the proposals and clean the workspace.

`Save`(shortcut:'s'): save annotations to file. Do not forget to save your annotation for each image, or it will be lost when you switch to the next image.

`Edit Polygons`: in this mode, you could modify the annotated objects, such as changing the category labels or ids by double click on object items in the
annotation dock. And you can modify the boundary by draging the points on the boundary.

`Delete`(shortcut:'d'): under `Edit Mode`, delete selected/hightlight objects from annotation dock.

`Reduce Point`: under `Edit Mode`, if you find the polygon is too dense to edit, you could use this button to reduce the points on the selected polygons. But this will slightly reduce the annotation quality.

`Zoom in/out`: press 'CTRL' and scroll wheel on the mouse

`Class On/Off`: if the Class is turned on, a dialog will show after you accept a mask to record category and id, or the catgeory will be default value "Object".

### 6. Notebooks
`extract-frames.ipynb`
- Notebook to extract images from videos. 

`yolo-format.ipynb`
- Once done annotating with the program, you can use this notebook to convert the .json labels into .txt YOLO labels and to collect all labels into a single folder and then to subsequently split them into the appropriate dataset structure for training with YOLO.

`yolo-training.ipynb`
- This notebook is to train the YOLO model on the dataset collated. This notebook also has a function to test the inference of a model on a video and to output the inference into a .mp4 video.
- This notebook also has a function to automate annotation on a set of images for the segment-anything-annotator.

`modify-labels.ipynb`
- This notebook is used to batch change label names for a directory of .json labels generated from the program. It can also be used to merge label names.
- Furthemore, this notebook can also identify specific images in the directory with specified labels, for debugging purposes.

`segment-to-bb.ipynb`
- This notebook is used to convert a directory of YOLO segmentation .txt labels into bounding box .txt labels

**Note: Refer to the individual notebooks for instructions**

## Acknowledgement 
This repo is built on [SAM](https://github.com/facebookresearch/segment-anything) and [Labelme]().


