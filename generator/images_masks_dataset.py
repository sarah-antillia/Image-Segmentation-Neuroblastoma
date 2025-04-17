import os
import sys
import glob
import shutil

import traceback

from svglib.svglib import svg2rlg

from reportlab.graphics import renderPM

#pip install svglib
#pip install rlPyCairo
# images_masks_dataset


def create_images_masks_dataset(input_dir, output_images_dir, output_masks_dir):
  svg_files = glob.glob(input_dir + "/*.svg")
  
  for svg_file in svg_files:
 
    drawing = svg2rlg(svg_file)
    basename = os.path.basename(svg_file)
    category = basename.split("_")[0]

    categoried_images_dir = os.path.join(output_images_dir, category + "/images/")
    categoried_masks_dir  = os.path.join(output_masks_dir,  category + "/masks/")

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
    input_dir = "./groundtruth_svgs"
    output_images_dir = "./Nuclei/"
    output_masks_dir  = "./Nuclei/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    #svg2jpg(input_dir, output_dir)
    create_images_masks_dataset(input_dir, output_images_dir, output_masks_dir)

  except:
    traceback.print_exc()
