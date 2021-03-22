# LogisticsOCR

## Introduction

LogisticsOCR makes use of [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to detect and recognize text tagged on Steel Stock.

## Installation

**1. Install PaddlePaddle Fluid v2.0**
```
pip3 install --upgrade pip

# If you have cuda9 or cuda10 installed on your machine, please run the following command to install
python3 -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple

# If you only have cpu on your machine, please run the following command to install
python3 -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple
```
For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/install/quick) for operation.

**2. Clone LogisticsOCR repo**
```
git clone https://github.com/dannylzj/Image_Classification_OCR/LogisticsOCR
```

**3. Install third-party libraries**
```
cd LogisticsOCR
pip3 install -r requirements.txt
```

If you getting this error `OSError: [WinError 126] The specified module could not be found` when you install shapely on windows.

Please try to download Shapely whl file using [http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely).

Reference: [Solve shapely installation on windows](https://stackoverflow.com/questions/44398265/install-shapely-oserror-winerror-126-the-specified-module-could-not-be-found)

## Inference Models

The inference models being used are the Universal Chinese OCR model (143M) and consist of:
1) Text Detection Model - [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar)
2) Angle Classification Model - [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar)
3) Text Recognition Model - [Inference Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar)

Create `./inference` folder in the main directory and uncompress the models downloaded above. Or alternatively run the following commands below:

```
# Create inference folder and cd into directory
mkdir inference && cd inference
# Download the detection model and unzip
wget {https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar} && tar xf {ch_ppocr_server_v2.0_det_infer.tar}
# Download the recognition model and unzip
wget {https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar} && tar xf {ch_ppocr_mobile_v2.0_cls_infer.tar}
# Download the direction classifier model and unzip
wget {https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar} && tar xf {ch_ppocr_server_v2.0_rec_infer.tar}
# Return to main directory
cd ..
```

After decompression, the file structure should be as follows:

```
├── ch_ppocr_mobile_v2.0_cls_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── ch_ppocr_mobile_v2.0_det_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── ch_ppocr_mobile_v2.0_rec_infer
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

## Usage

### For Single Image / Image Set Visualization

* The following code implements text detection, angle classification and text recognition process.
1) When performing the prediction, you need to specify the path of a single image or image set through the parameter `image_dir`.
2) You need to also specify the path of the inference model
	- The parameter `det_model_dir` specifies the path to the text detection inference model
	- the parameter `rec_model_dir` specifies the path to the text recognition inference model
	- the parameter `cls_model_dir` specifies the path to identify the direction classifier model
	- the parameter `use_angle_cls` specifies whether to use the direction classifier 
	- the parameter `use_space_char` specifies whether to predict the space char
3) The visualization results are saved to the `./inference_results` folder by default.

```bash

# Predict a single image specified by image_dir
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_ppocr_mobile_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v2.0_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True

# Predict imageset specified by image_dir
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/" --det_model_dir="./inference/ch_ppocr_mobile_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v2.0_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True

# If you want to use the CPU for prediction, you need to set the use_gpu parameter to False
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_ppocr_mobile_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v2.0_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True --use_gpu=False
```

### For Image Set Prediction to .csv
* The following script implements text detection, angle classification and text recognition process for Steel Plate Tags, Angle Bar Tags and Bulb Flat Tags
1) The default image storage location for Steel Plates is `./project/B384/steelplate`
2) The default image storage location for Angle Bars is `./project/B384/anglebar`
3) The default image storage location for Bulb Flats is `./project/B384/bulbflat`
4) The .csv file will be saved as `table.csv` in the respective folders after running the commands below.

```bash

# Predict images of steel plates
python3 tools/infer/predict_steelplate.py

# Predict images of angle bars
python3 tools/infer/predict_anglebar.py

# Predict images of bulb flats
python3 tools/infer/predict_bulbflat.py
```