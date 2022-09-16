# Understanding 3D Object Articulation in Internet Videos

Code release for our paper

```
Understanding 3D Object Articulation in Internet Videos
Shengyi Qian, Linyi Jin, Chris Rockwell, Siyi Chen, David Fouhey
CVPR 2022
```

![teaser](docs/teaser.png)

Please check the [project page](https://jasonqsy.github.io/Articulation3D/) for more details and consider citing our paper if it is helpful:

```
@inproceedings{Qian22,
    author = {Shengyi Qian and Linyi Jin and Chris Rockwell and Siyi Chen and David F. Fouhey},
    title = {Understanding 3D Object Articulation in Internet Videos},
    booktitle = {CVPR},
    year = 2022
}
```

## Setup

We are using [pyenv](https://github.com/pyenv/pyenv) to set up the anaconda environment. It is tested on pytorch 1.7.1, detectron2 0.4, and pytorch3d 0.4.0.

```bash
VERSION_ALIAS="articulation3d" PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install anaconda3-2020.11

# pytorch and pytorch3d
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d

# detectron2 with pytorch 1.7, cuda 10.2
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html
```

Alternatively, we have tested the anaconda virtual environment. It is tested on pytorch 1.12.1, detectron2 0.6, and pytorch3d 0.7.0.

```bash
conda create -n articulation3d python=3.8
conda activate articulation3d
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

To install python packages,

```
# other packages
pip install scikit-image matplotlib imageio plotly opencv-python
pip install mapbox-earcut
pip install numpy-quaternion
pip install imageio-ffmpeg
pip install scikit-learn

# install articulation3d
cd articulation3d
pip install -e .
```

If necessary, download our [pretrained model](https://www.dropbox.com/s/50uderl5ynan2yt/model_0059999.pth?dl=0) and put it at `exps/model_0059999.pth`

```bash
mkdir exps
cd exps
wget https://www.dropbox.com/s/50uderl5ynan2yt/model_0059999.pth?dl=0
```

## Inference

To run the model and temporal optimization on a video,

```bash
python tools/inference.py --config config/config.yaml --input example.mp4 --output output
```

To save the 3d model, add `--save-obj` and `--webvis` flags,

```bash
python tools/inference.py --config config/config.yaml --input example.mp4 --output output --save-obj --webvis
```

## Training 

Our training consists of three stages.

In the first stage, we train the bounding box on Internet videos.

```bash
python tools/train_net.py --config-file config/step1_bbox.yaml
```

In the second stage, we train articulation axis on Internet videos while freezing the backbone.

```bash
python tools/train_net.py --config-file config/step2_axis.yaml
```

In the final stage, we train the plane head on ScanNet images.

```bash
python tools/train_net.py --config-file config/step3_plane.yaml
```

## Acknowledgment

We reuse the codebase of [SparsePlanes](https://github.com/jinlinyi/SparsePlanes) and [Mesh R-CNN](https://github.com/facebookresearch/meshrcnn).
