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



