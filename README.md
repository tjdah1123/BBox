
Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

## Quick Start
shell command
1. `conda create -n keypoint python==3.7`
2. `conda activate keypoint`
3. `git clone -b keypoints_infer --single-branch http://192.168.1.126:9000/lab/keypoints_detection.git`
   - login is required (username, password)
4. `cd keypoints_detection`
5. setup
   ```
   pip install -r requirement.txt
   python setup.py build develop
   ```
7. execution
   ```
   python tools/magic.py --config_file $CONFIG_FILE_PATH --image_path $IMAGE_PATH --result_path $RESULT_PATH
   
   # ex.
   python tools/magic.py --config_file configs/COCO-Keypoints/keypoint_sports.yaml --image_path outputs/soccer.jpg
   ```

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).


# GPU RTX 3080 Error
1. cuda install
```
wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
sudo sh cuda_11.2.2_460.32.03_linux.run
# only CUDA toolkit install
```

2. upload cudnn in dms/etc to RTX3080 server and unzip it.
```
tar -xvf cudnn-11.2-linux-x64-v8.1.1.33
```

3. copy cudnn related files to the place where cuda toolkit is installed
```
cp ./cuda/include/* /usr/local/cuda-11.2/include
cp -P ./cuda/lib64/* /usr/local/cuda-11.2/lib64
chmod a+r /usr/local/cuda-11.2/lib64/libcudnn*
```

4. pytorch install
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

5. fvcore install
```
pip install fvcore==0.1.3.post20210204
```

6. iopath install
```
pip install iopath==0.1.3
```

7. detectron2 setup & execution