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
image_file_list = get_image_file_list("./project/B384/steelplate/")
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
print("Starting model inference ...")
for image in image_file_list:
    print("Processing " + str(image) + " ...")
    result = s_ocr.ocr(image, cls=True)
    line1_coord.append(result[0][0])
    line1.append(result[0][1][0])
    line2_coord.append(result[1][0])
    line2.append(result[1][1][0])
    line1_confidence.append(result[0][1][1])
    line2_confidence.append(result[1][1][1])
    filename.append(os.path.basename(image))
print("Model inference completed!")

s_df = pd.DataFrame([filename, line1_coord, line1, line1_confidence, line2, line2_confidence]).T
s_df.columns = ["filename", "line1_coord", "line1", "line1_confidence", "line2", "line2_confidence"]

# Function to extract heat number
def get_heat_number(row):
    if ' ' in row:
        return row.rsplit(None, 1)[0]
    else:
        if len(row) < 10:
            return "Cannot detect heat number, please check image directly"
        else:
            return row[0:10]

# Function to extract dimension
def get_dimension(row):
    if '*' not in row:
        return "Cannot detect dimension, please check image directly"
    elif ' ' in row:
        return row.rsplit(None, 1)[1]
    else:
        return row[10:]
    
# Function to extract grade
def get_grade(row):
    if 'S' in row:
        s_index = [i for i,char in enumerate(row) if char == 'S']
        if any(char.isdigit() for char in row[s_index[0]:]):
            return row[s_index[0]:]
        else:
            return "Cannot detect grade, please check image directly"
    else:
        return "Cannot detect grade, please check image directly"

# Replace all instances of '米' to '*'
# Note: To remove this rule if chinese character recognition is required
s_df['line1'] = s_df['line1'].str.replace('米', '*')

s_df['heat number'] = s_df['line1'].apply(get_heat_number)
s_df['dimension'] = s_df['line1'].apply(get_dimension)
s_df['grade'] = s_df['line2'].apply(get_grade)
df = s_df[['filename', 'heat number', 'dimension', 'line1_confidence', 'grade','line2_confidence']]

df.to_csv("./project/B384/steelplate/table.csv")
print("Successfully saved output in ./project/B384/steelplate/table.csv")