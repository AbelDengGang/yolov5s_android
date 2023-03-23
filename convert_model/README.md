# Model Conversion (PyTorch -> ONNX -> OpenVino -> Tensorflow -> TfLite)
## Why we convert the model via OpenVino format?
As you know, PyTorch have the `NCHW` layout, and Tensorfloww have `NHWC` layout.  
[`onnx-tf`](https://github.com/onnx/onnx-tensorflow) supports model conversion from onnx to tensorflow, but the converted model includes a lot of `Transpose` layer because of the layout difference.  
By using OpenVino's excellent model optimizer and [`openvino2tensorflow`](https://github.com/PINTO0309/openvino2tensorflow), we can obtain a model without unnecessary transpose layers.  
For more information, please refer this article by the developer of `openvino2tensorflow` : [Converting PyTorch, ONNX, Caffe, and OpenVINO (NCHW) models to Tensorflow / TensorflowLite (NHWC) in a snap](https://qiita.com/PINTO/items/ed06e03eb5c007c2e102)
  
## Docker build
```sh
git clone --recursive git@github.com:AbelDengGang/yolov5s_android.git
cd yolov5s_android
docker build ./ -f ./docker/Dockerfile  -t yolov5s_android
docker run -it --gpus all -v `pwd`:/workspace yolov5s_android bash
```
The following process is performed in docker container.  

If docker report can not find GPU, need install nvidia-container-toolkit
```
sudo apt-get install -y nvidia-container-toolkit
```

### modify default pip source
open  /home/developer/.config/pip/pip.conf add:
```
[global]
index-url = http://pypi.douban.com/simple/
trusted-host = pypi.douban.com
```
or run command to set:
```
pip config set global.index-url http://pypi.douban.com/simple/
pip config set global.trusted-host  pypi.douban.com
```
Then need install onnxruntime and cormtools in docker container and then commit the container
```
/usr/bin/python3 -m pip install --user onnxruntime   -i https://pypi.douban.com/simple/
/usr/bin/python3 -m pip install --user coremltools   -i https://pypi.douban.com/simple/
```

## PyTorch -> ONNX
Download the pytorch pretrained weights and export to ONNX format.  
默认情况下，下载最新release的模型。如果要指定模型版本，则修改 yolov5/utils/downloads.py
```
diff --git a/utils/downloads.py b/utils/downloads.py
index 6b2c374..3a65da5 100644
--- a/utils/downloads.py
+++ b/utils/downloads.py
@@ -52,7 +52,7 @@ def attempt_download(file, repo='ultralytics/yolov5'):  # from utils.downloads i
         # GitHub assets
         file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
         try:
-            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
+            response = requests.get(f'https://api.github.com/repos/{repo}/releases/v5.0').json()  # github api
             assets = [x['name'] for x in response['assets']]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
             tag = response['tag_name']  # i.e. 'v1.0'
         except:  # fallback plan
```

```sh
cd workspace
cd yolov5
./data/scripts/download_weights.sh #modify 'python' to 'python3' if needed
python3 export.py --weights ./yolov5s.pt --img-size 640 640 --simplify
```
after this command , you can get yolov5s.mlmodel

## ONNX -> OpenVino
setup openvino
```
/opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer/install_prerequisites
sudo ./install_prerequisites.sh
```
add openvino env var  in ~/.bashrc
```
source /opt/intel/openvino_2021.3.394/bin/setupvars.sh 
```

```sh
python3 /opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer/mo.py \
 --input_model yolov5s.onnx \
 --input_shape [1,3,640,640] \
 --output_dir ./openvino \
 --data_type FP32 \
 --output Conv_245,Conv_325,Conv_405
```
You will get `yolov5s.bin  yolov5s.mapping  yolov5s.xml` as OpenVino model.  

在转换时出现问题：
```
[ ERROR ]  Exception occurred during running replacer "REPLACEMENT_ID" (<class 'extensions.front.user_data_repack.UserDataRepack'>): No node with name Conv_325
```
不知道该如何解决。

If you use the other verion yolov5, you have to check the output layer IDs in netron.  
The output layers are three most bottom Convolution layers. 
```sh
netron yolov5s.onnx
```
<img src="https://github.com/lp6m/yolov5s_android/raw/media/onnx_output_layers.png" width=50%> 
  
In this model, the output layer IDs are `Conv_245,Conv_325,Conv_405`.  
**We convert the ONNX model without detect head layers.**
### Why we exclude detect head layers?
NNAPI does not support some layers included in detect head layers.  
For example, The number of dimension supported by [ANEURALNETWORKS_MUL](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0ab34ca99890c827b536ce66256a803d7a) operator for multiply layer is up to 4.  
The input of multiply layer in detect head layers has 5 dimension, so NNAPI delegate cannot load the model.  
We tried to include detect head layers into tflite [in other method](https://github.com/lp6m/yolov5s_android/issues/2), but not successful yet.
  
For the inference, the calculation of detect head layers are implemented outside of the tflite model.  
For Android, the detect head layer is [implemented in C++ and executed on the CPU through JNI](https://github.com/lp6m/yolov5s_android/blob/host/app/tflite_yolov5_test/app/src/main/cpp/postprocess.cpp).  
For host evaluation, we use [PyTorch model](https://github.com/lp6m/yolov5s_android/blob/host/host/detector_head.py) ported from original yolov5 repository.


## OpenVino -> TfLite
Convert OpenVino model to Tensorflow and TfLite by using `openvino2tensorflow`.
```sh
source /opt/intel/openvino_2021/bin/setupvars.sh 
export PYTHONPATH=/opt/intel/openvino_2021/python/python3.6/:$PYTHONPATH
openvino2tensorflow \
--model_path ./openvino/yolov5s.xml \
--model_output_path tflite \
--output_pb \
--output_saved_model \
--output_no_quant_float32_tflite 
```
You will get `model_float32.pb, model_float32.tflite`.  

### Quantize model
Load the tensorflow frozen graph model (pb) obtained by the previous step, and quantize the model.  
The precision of input layer is `uint8`. The precision of the output layer is `float32` for commonalize the postprocess implemented in C++(JNI). 
For calibration process in quantization, you have to prepare coco dataset in tfds format.  
```sh
cd ../convert_model
usage: quantize.py [-h] [--input_size INPUT_SIZE] [--pb_path PB_PATH]
                   [--output_path OUTPUT_PATH] [--calib_num CALIB_NUM]
                   [--tfds_root TFDS_ROOT] [--download_tfds]

optional arguments:
  -h, --help            show this help message and exit
  --input_size INPUT_SIZE
  --pb_path PB_PATH
  --output_path OUTPUT_PATH
  --calib_num CALIB_NUM
                        number of images for calibration.
  --tfds_root TFDS_ROOT
  --download_tfds       download tfds. it takes a lot of time.
```
```sh
python3 quantize.py --input_size 640 --pb_path /workspace/yolov5/tflite/model_float32.pb \
--output_path /workspace/yolov5/tflite/model_quantized.tflite
--calib_num 100
```
