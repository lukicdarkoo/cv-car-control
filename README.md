
## Training

### Make dataset
Install `labelImg` and open it:
```bash
pip3 install labelImg
labelImg
```

In `labelImg` make sure to click `YOLO` and annotate all the images.
Once all the images annotated move them, together with the corresponding `*.txt` files, to `dataset/train` and `dataset/valid`.
It is very important to move 10-20 percent `dataset/valid` directory and the rest, move to `dataset/train`.
YOLO needs list of images for validation and training written to files.
You can make the files by invoking the following commands from the project's root:
```bash
cd ./dataset
find train -name '*.jpg' > train.txt
find valid -name '*.jpg' > valid.txt
```

### Start training locally

Download pretrained weights for the transfer learning:
```bash
wget https://pjreddie.com/media/files/darknet53.conv.74
```

Install `darknet`

```bash
cd ./dataset
$HOME/darknet/darknet detector train ./custom.data ../yolov3-tiny_train.cfg ../darknet53.conv.74 
```

### Start training on Google Colab

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xQ2eHRcVIB8GCBsp85jxWabH-tavZIE3?usp=sharing)
