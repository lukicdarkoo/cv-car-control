
## Training

### Make dataset
Install `labelImg` and open it:
```bash
pip3 install labelImg
labelImg
```

In `labelImg` make sure to click `YOLO` and annotate all the images.
Once all the images annotated move them, together with the corresponding `*.txt` files, to `dataset/train` and `dataset/valid`.
It is very important to move 10-20 percent of the images to `dataset/valid` directory and the rest, move to `dataset/train`.
YOLO also needs a list of images for validation and training written to `*.txt` files.
You can make the files by invoking the following commands from the project's root:
```bash
cd ./dataset
find train -name '*.jpg' > train.txt
find valid -name '*.jpg' > valid.txt
```

### Start training locally

If you have GPU and CUDA available then you may prefer to train the YOLO locally.

First, install `darknet` with GPU and CUDA support:
```bash

git clone https://github.com/pjreddie/darknet
cd darknet
sed -i 's/CUDNN=0/CUDNN=1/g' darknet/Makefile
sed -i 's/GPU=0/GPU=1/g' darknet/Makefile
make -j8
```

To accelerate training it is usually a good idea to start from pre-trained model.
Download the pre-trained model as:
```bash
wget https://pjreddie.com/media/files/darknet53.conv.74
```

Finally, you can initiate the training as:
```bash
cd ./dataset
$HOME/darknet/darknet detector train ./custom.data ../yolov3-tiny_train.cfg ../darknet53.conv.74 
```
and your models will be available in `./dataset/backup`.

### Start training on Google Colab

In case you don't have GPU it probably a good idea to use [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xQ2eHRcVIB8GCBsp85jxWabH-tavZIE3?usp=sharing).
After the Google Colab is ready, upload `yolov3-tiny_train.cfg` and `./dataset` folder compressed as `dataset.tar.xz`.

> You can create `dataset.tar.xz` as `tar -cf ./dataset ./dataset.tar.xz`

Run all cells and after 1 hour of training the model will be ready waiting in `./dataset/backup`.
