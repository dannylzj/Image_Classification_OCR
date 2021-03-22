import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

from paddleocr import PaddleOCR
from ppocr.utils.utility import get_image_file_list
import os
import pandas as pd

# Get image file list directory
image_file_list = get_image_file_list("./project/B384/bulbflat/")
print("Total {0} images found.".format(str(len(image_file_list))))

# Instantiate PaddleOCR with server inference model
s_ocr = PaddleOCR(use_angle_cls=True, det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer/", rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/", cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/", use_space_char=True, use_gpu=True)

# Run images through server inference model
filename = []
line1_coord = []
line1 = []
line2_coord = []
line2 = []
line1_confidence = []
line2_confidence = []
dict1 = {}
print("Starting model inference ...")
for image in image_file_list:
    print("Processing " + str(image) + " ...")
    result = s_ocr.ocr(image, cls=True)
    dict1[os.path.basename(image)] = result
print("Model inference completed!")

# Code to extract information
filename =[]
heat_number = []
hn_conf = []
dimension = []
dim_conf = []
grade = []
grade_conf = []
for key,value in dict1.items():
    filename.append(key)
    dim_count = 0
    hn_count = 0
    grade_count = 0
    for i in value:
        # Extract dimension from image. Assume dimensions contain with "HP".
        if (("X" in i[1][0]) or ("*" in i[1][0]) or ("×" in i[1][0]) or ("x" in i[1][0])) and ("HP" in i[1][0]):
            dimension.append(i[1][0])
            dim_conf.append(i[1][1])
            dim_count+=1
    for i in value:
        # Extract heat number from image
        if (len(i[1][0]) == 9) and (i[1][0] not in dimension):
            if (i[1][0][0].isdigit()):
                heat_number.append(i[1][0])
                hn_conf.append(i[1][1])
                hn_count+=1
    for i in value:
        # Extract grade from image ('S355' grade only). Grades should be standard, need to know other grades
        if 'S355' in i[1][0]:
            grade.append(i[1][0])
            grade_conf.append(i[1][1])
            grade_count+=1
    if dim_count == 0:
        dimension.append('Cannot detect Dimension. Please check image directly.')
        dim_conf.append('')
    elif hn_count == 0:
        heat_number.append('Cannot detect Heat Number. Please check image directly.')
        hn_conf.append('')
    elif grade_count == 0:
        grade.append('Cannot detect Grade. Please check image directly.')
        grade_conf.append('')

df = pd.DataFrame([filename, heat_number, hn_conf, dimension, dim_conf, grade, grade_conf]).T
df.columns = ["Filename", "Heat Number", "HN Confidence", "Dimension", "Dim Confidence", "Grade", "Grade Confidence"]

df.to_csv("./project/B384/bulbflat/table.csv")
print("Successfully saved output in ./project/B384/bulbflat/table.csv")