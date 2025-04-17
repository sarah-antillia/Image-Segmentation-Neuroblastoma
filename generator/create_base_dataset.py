# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2023/08/16 to-arai 
# create_base_dataset.py

import os
import sys
import glob
import shutil

import traceback

from svglib.svglib import svg2rlg

from reportlab.graphics import renderPM

#pip install svglib
#pip install rlPyCairo
# create base images_masks_dataset


def create_base_dataset(input_dir, output_dir):
  svg_files = glob.glob(input_dir + "/*.svg")
  
  for svg_file in svg_files:
 
    drawing = svg2rlg(svg_file)
    basename = os.path.basename(svg_file)
    category = basename.split("_")[0]

    categoried_images_dir = os.path.join(output_dir, category + "/images/")
    categoried_masks_dir  = os.path.join(output_dir,  category + "/masks/")

    if not os.path.exists(categoried_images_dir):
     os.makedirs(categoried_images_dir)

    if not os.path.exists(categoried_masks_dir):
     os.makedirs(categoried_masks_dir)

    jpg_filename     = basename.split(".")[0] + ".jpg"
    
    output_mask_file = os.path.join(categoried_masks_dir, jpg_filename)

    renderPM.drawToFile(drawing, output_mask_file, fmt="JPG")
    print("Saved {}".format(output_mask_file))

    jpg_filepath = os.path.join(input_dir, jpg_filename)

    shutil.copy2(jpg_filepath, categoried_images_dir)
    print("Copied {}".format(jpg_filepath))


if __name__ == "__main__":
  try:
    input_dir  = "./groundtruth_svgs"
    output_dir = "./Nuclei_base/"

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    create_base_dataset(input_dir, output_dir)

  except:
    traceback.print_exc()
