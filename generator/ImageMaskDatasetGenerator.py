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
# 2023/08/15 to-arai 
# ImageMaskDatasetGenerator.py

import os
import sys
import shutil
import cv2
from PIL import Image, ImageOps

import glob
import numpy as np
import math
import traceback

class ImageMaskDatasetGenerator:
  def __init__(self,  shrinks= [0.4, 0.6, 0.8, 1.0], angles=[90, 180, 270], debug=False):
    self.SHRINKS = shrinks
    self.RESIZE  = 512
    self.DEBUG    = debug
    self.ANGLES   = angles

  def augment(self, image, output_dir, filename):

    if len(self.ANGLES) > 0:
      for angle in self.ANGLES:
        rotated_image = image.rotate(angle)

        w, h = rotated_image.size
        for resize in self.SHRINKS:
          rw = int (w * resize)
          rh = int (h * resize)
     
          resized = rotated_image.resize((rw, rh))
          squared = self.past_to_background(resized)
          ratio   = str(resize).replace(".", "_")
          output_filename = "rotated_" + str(angle) + "_" + "shrinked_" + ratio + "_" + filename
          image_filepath  = os.path.join(output_dir, output_filename)
          squared.save(image_filepath)
          print("=== Saved {}".format(image_filepath))
      
    # Create mirrored image
    mirrored = ImageOps.mirror(image)
    output_filename = "mirrored_" + filename
    image_filepath = os.path.join(output_dir, output_filename)
    
    mirrored.save(image_filepath)
    print("=== Saved {}".format(image_filepath))
        
    # Create flipped image
    flipped = ImageOps.flip(image)
    output_filename = "flipped_" + filename

    image_filepath = os.path.join(output_dir, output_filename)

    flipped.save(image_filepath)
    print("=== Saved {}".format(image_filepath))

  def past_to_background(self, image):
     w, h = image.size
     pixel = image.getpixel((w-20, h-20))
     background = Image.new("RGB", (self.RESIZE, self.RESIZE), pixel)
     x = (self.RESIZE - w)//2
     y = (self.RESIZE - h)//2

     background.paste(image, (x, y))
     return background

  def resize_to_square(self, image, RESIZE=512):
     image = Image.fromarray(image)
     w, h = image.size
     bigger = w
     if h >bigger:
       bigger = h
     background = Image.new("RGB", (bigger, bigger))
     x = (bigger - w)//2
     y = (bigger - h)//2

     background.paste(image, (x, y))
     background = background.resize((RESIZE, RESIZE))
     return background

  def generate(self, images_dir, masks_dir, output_images_dir, output_masks_dir):
    image_files = glob.glob(images_dir + "/*.jpg")
    mask_files  = glob.glob(masks_dir  + "/*.jpg")
    num_image_files = len(image_files)
    num_mask_files  = len(mask_files)
    print("=== num_image_files {}".format(num_image_files))
    print("=== num_mask_files {}".format(num_mask_files))

    for mask_file in mask_files:
      print("=== mask_file {}".format(mask_file))
      mask = cv2.imread(mask_file)
      mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
      mask = self.resize_to_square(mask)

      basename = os.path.basename(mask_file)
      image_file = os.path.join(images_dir, basename)
      image = cv2.imread(image_file)
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      image = self.resize_to_square(image)
 
      output_maskfile  = os.path.join(output_masks_dir, basename)
      output_imagefile = os.path.join(output_images_dir, basename)
      mask.save(output_maskfile)
      image.save(output_imagefile)

      self.augment(mask,  output_masks_dir, basename)
      self.augment(image, output_images_dir, basename)




if __name__ == "__main__":
  try:
    images_dir = "./Nuclei_base/Neuroblastoma/images/"
    masks_dir  = "./Nuclei_base/Neuroblastoma/masks/"
  
    output_images_dir = "./Neuroblastoma-master/images/"
    output_masks_dir  = "./Neuroblastoma-master/masks/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    shrinks= [0.4, 0.6, 0.8, 1.0]
    angles = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260,280, 300, 320, 340]

    generator = ImageMaskDatasetGenerator(shrinks=shrinks, angles=angles, debug=False)
    generator.generate(images_dir, masks_dir, output_images_dir, output_masks_dir)

  except:
    traceback.print_exc()


