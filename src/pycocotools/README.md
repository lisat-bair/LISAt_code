# cocobetter

Customized version of pycocotools. Should be a drop-in replacement for the official pycocotools, just with more features.

## What's different from the official?

* Provides support for custom `max_dets` settings
* Adds metrics AP and AR at IoU threshold of 25% (official version does AP50, AP75, and AP)
* Various helper classes for building, converting, shrinking, inspecting, etc, COCO formated data.

## Wishlist / TODO

* [x] Add per-class version of `COCOeval.stats` (also make it a dict as described in previous bullet)
* [x] Improve the `.stats` output to make it easier to pull out individual stats without using hardcoded ordinals/indexes. You can access the stats via `COCOeval.stats_dict`
* [ ] Make `COCOeval.stats` backwards compatible with original pycocotools. Right now it returns a customized array of values, so it's not full drop-in replacement. Existing code that assumes which metric is in each index of the `.stats` property will get the wrong values. Easy to workaround for now is to switch to using `COCOeval.stats_dict`.
* [ ] Pull in the faster eval code from detectron2 (if the license allows for it)
  * <http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html>
* [ ] Add PR curve generation
* [x] Add tools and classes for manipulating coco formatted json
* [ ] Add tests (along with eval results and json's)
* [ ] Publish to pypi
* [ ] Test the anchor box clustering
* [ ] Merge in latest changes from pycocotools (2.0.2>=)
* [ ] Add more useful notebooks
* [ ] Add documentation / examples of how to use some of the features

## Installing (choose one method)

### Pre-requisites

```bash
pip install cython pybind11 setuptools
```

### Quick Install (Non-editable)

Use this method if you don't need to customize the code, and just want to install it into your python environment:

```bash
conda activate "YOUR_ENV_NAME"
python -m pip install git+https://github.com/GiscardBiamby/cocobetter.git#subdirectory=PythonAPI
```

`requirements.txt` method:

```text
pycocotools @ git+https://github.com/GiscardBiamby/cocobetter.git#subdirectory=PythonAPI
```

### Quick Install (Editable)

This will pull the repo into your local folder under `./src`, and install it into your python environment in develop mode. The conda package `pycocotools` will point to `./src` inside the folder you run this command from. Editing the code in `./src` it will immediately reflect in your python env (no need to reinstall).

```bash
conda activate "YOUR_ENV_NAME"
python -m pip install -e \
    pycocotools @ git+https://github.com/GiscardBiamby/cocobetter.git#subdirectory=PythonAPI
```

OR, add this line to your `requirements.txt`:

```bash
-e git+https://github.com/GiscardBiamby/cocobetter.git#egg=pycocotools\&subdirectory=PythonAPI
```

### Method 3: Clone and install

```bash
conda activate "YOUR_ENV_NAME"
git clone git@github.com:GiscardBiamby/cocobetter.git
cd cocobetter/PythonAPI
pip install -e .
```

### Check the install location

Useful for when  you want to make sure your environment is using cocobetter instead of pycocotools. Use this command to check where pycocotools is being loaded from:

```bash
conda activate "YOUR_ENV_NAME"
python -c "import pycocotools as pcc; print(pcc.__file__)"
```

Sample output:

```bash
> /home/username/my_project/cocobetter/PythonAPI/pycocotools/__init__.py
```

## Usage

From your python project:

```python
from pycocotools.coco import coco
from pycocotools.cocoeval import COCOeval
```

## COCO API - <http://cocodataset.org/>

COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. This package provides Matlab, Python, and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO. Please visit <http://cocodataset.org/> for more information on COCO, including for the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website. The Matlab and Python APIs are complete, the Lua API provides only basic functionality.

In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
\-Please download, unzip, and place the images in: coco/images/
\-Please download and place the annotations in: coco/annotations/
For substantially more details on the API please see <http://cocodataset.org/#download>.

After downloading the images and annotations, run the Matlab, Python, or Lua demos for example usage.

To install:
\-For Python, run "make" under coco/PythonAPI
