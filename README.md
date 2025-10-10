##  ALIV-torch
Adaptive Layered Image Vectorization




ALIV is able to explicitly presents a editable representation for vectorized images. 

## Installation
```bash
pip3 install torch torchvision
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
pip install scikit-fmm
pip install opencv-python==4.5.4.60 
pip install easydict


```
Next, please refer DiffVG to install [pydiffvg](https://github.com/BachiLi/diffvg)


## Run
```bash
python main.py --config config/base.yaml --color_num 20 --color_dis 1000 --experiment experiment_1 --signature e-0 --target data/test/e-0.png --log_dir results
```
Please modify the config files to change configurations.
