# Image-Segmentation-Neuroblastoma (2023/08/16)
<h2>
1 Image-Segmentation-Neuroblastoma 
</h2>
<p>
This is an experimental project for Image-Segmentation of Neuroblastoma by using
 <a href="https://github.com/atlan-antillia/Tensorflow-Slightly-Flexible-UNet">Tensorflow-Slightly-Flexible-UNet</a> Model,
which is a typical classic Tensorflow2 UNet implementation <a href="./TensorflowUNet.py">TensorflowUNet.py</a> 
<p>
The image dataset used here has been taken from the following web site.
</p>
<b>An annotated fluorescence image dataset for training nuclear segmentation methods</b><br>
<pre>
Sabine Taschner-Mandl, M. Ambros 1Peter F. Ambros,Beiske 2Allan Hanbury,
Doerr,Weiss,Berneder, Magdalena Ambros,Eva Bozsaky

Description of the Biostudies dataset S-BSST265
Title: An annotated fluorescence image dataset for training nuclear segmentation methods
The dataset is assigned with an open data license (CC0)
Author: Florian Kromp, 15.04.2020
Children's Cancer Research Institute, Vienna, Austria
florian.kromp@ccri.at
</pre>
<pre>
https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST265
</pre>



<br>
<br>
<h2>
2. Install Image-Segmentation-Neuroblastoma 
</h2>
Please clone Image-Segmentation-Neuroblastoma.git in a folder <b>c:\google</b>.<br>
<pre>
>git clone https://github.com/sarah-antillia/Image-Segmentation-Neuroblastoma.git<br>
</pre>
You can see the following folder structure in your working folder.<br>

<pre>
Image-Segmentation-Neuroblastoma 
├─asset
└─projects
    └─-Neuroblastoma
        ├─eval
        ├─generator
        ├─test
        ├─models
        ├─Neuroblastoma
        │   ├─test
        │   │  ├─images
        │   │  └─masks
        │   ├─train
        │   │  ├─images
        │   │  └─masks
        │   └─valid
        │       ├─images
        │       └─masks
        ├─test_output
        └─test_output_merged
</pre>

<h2>
3 Prepare dataset
</h2>

<h3>
3.1 Download master dataset
</h3>
  Please download the original image and mask dataset from the following website <br>
<b>An annotated fluorescence image dataset for training nuclear segmentation methods</b><br>

<pre>
https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST265
</pre>

</pre>
The <b>S-BSST265/dataset</b> has the following folder structure.<br>
<pre>
./S-BSST265/dataset
├─groundtruth
├─groundtruth_svgs
├─rawimages
├─singlecell_groundtruth
├─visualized_groundtruth
└─visualized_singlecell_groundtruth
</pre>
For example, <b>groundtruth_svgs</b> folder contains SVG-Files for each annotated masks and corresponding raw image in JPEG format.
It contains the files corresponding to the following categories.<br>
<pre>
├─Ganglioneuroblastoma
├─Neuroblastoma
├─normal
└─otherspecimen
</pre>.

<h3>
3.2 Create base dataset
</h3>
By using Python script <a href="./projects/Neuroblastoma/generator/create_base_dataset.py">create_base_dataset.py</a>,
we have created jpg images and masks <b>Nuclei_base</b> dataset from <b>groundtruth_svgs</b> dataset.<br> 
<pre>
./Nuclei_base
├─Ganglioneuroblastoma
│  ├─images
│  └─masks
├─Neuroblastoma
│  ├─images
│  └─masks
├─normal
│  ├─images
│  └─masks
└─otherspecimen
    ├─images
    └─masks
</pre>
For example, Nuclei_base/Neuroblastoma/images folder contains only 18 images, which is apparently too few to use for training of our UNet model.<br>
<b>Nuclei_base/Neuroblastoma/images</b><br>
<img src="./asset/Nuclei_base_Neuroblastoma_images.png" width="1024" height="auto"><br>
<br>
<b>Nuclei_base/Neuroblastoma/masks</b><br>
<img src="./asset/Nuclei_base_Neuroblastoma_masks.png" width="1024" height="auto"><br>
<br>

<h3>
3.3 Augment image and mask dataset for Neuroblastoma
</h3>
By using Python script <a href="./projects/Neuroblastoma/generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a>,
 we have created <b>Neuroblastoma-master</b> dataset from the <b>Nuclei_base/Neuroblastoma</b>.<br>
The script performs the following image processings.<br>
<pre>
1 Create 512x512 square images from Neuroblastoma image files.
2 Create 512x512 square mask  corresponding to the Neuroblastoma image files. 
3 Create rotated, resized, flipped and mirrored images and masks of size 512x512 to augment the resized square images and masks.
</pre>

The created <b>Neuroblastoma-master</b> dataset has the following folder structure.<br>
<pre>
./Neuroblastoma-master
├─images
└─masks
</pre>

<h3>
3.3 Split master to test, train and valid 
</h3>
By using Python script <a href="./projects/Neuroblastoma/generator/split_master.py">split_master.py</a>,
 we have finally created <b>Neuroblastoma</b> dataset from the Neuroblastoma-master.<br>
<pre>
./Neuroblastoma
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
    ├─images
    └─masks
</pre>
<b>train/images samples:</b><br>
<img src="./asset/train_images_samples.png" width="1024" height="auto">
<br>
<b>train/masks samples:</b><br>
<img src="./asset/train_masks_samples.png"  width="1024" height="auto">
<br>
<b>Dataset Inspection</b><br>
<img src="./asset/dataset_inspection.png" width="640" height="auto">
<br>
<br>
<h2>
4 Train TensorflowUNet Model
</h2>
 We have trained Neuroblastoma TensorflowUNet Model by using the following
 <b>train_eval_infer.config</b> file. <br>
Please move to ./projects/Neuroblastoma directory, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
, where train_eval_infer.config is the following.
<pre>
; train_eval_infer.config
; Dataset of Neuroblastoma
; 2023/08/16 (C) antillia.com

[model]
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 1
base_filters   = 16
base_kernels   = (7,7)
num_layers     = 7
dropout_rate   = 0.07
learning_rate  = 0.0001
clipvalue      = 0.5
dilation       = (2,2)
loss           = "bce_iou_loss"
;metrics        = ["iou_coef", "sensitivity", "specificity"]
metrics        = ["iou_coef"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 4
patience      = 10
metrics       = ["iou_coef", "val_iou_coef"]
model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "./Neuroblastoma/train/images"
mask_datapath  = "./Neuroblastoma/train/masks"
create_backup  = True

[eval]
image_datapath = "./Neuroblastoma/valid/images"
mask_datapath  = "./Neuroblastoma/valid/masks"
output_dir     = "./eval_output"
batch_size     = 4

[infer] 
images_dir = "./test/images"
output_dir = "./test_output"
merged_dir = "./test_output_merged"

[mask]
blur      = True
binarize  = True
threshold = 74

</pre>

Please note that the input image size and base_kernels size of this Neuroblastoma TensorflowUNet model are slightly large.<br>
<pre>
[model]
image_width    = 512
image_height   = 512
base_kernels   = (7,7)
</pre>

The training process has been stopped at epoch 100.<br><br>
<img src="./asset/train_console_output_at_epoch_100_0816.png" width="720" height="auto"><br>
<br>
<br>
<b>Train metrics line graph</b>:<br>
<img src="./asset/train_metrics.png" width="720" height="auto"><br>
<br>
<b>Train losses line graph</b>:<br>
<img src="./asset/train_losses.png" width="720" height="auto"><br>


<h2>
5 Evaluation
</h2>
 We have evaluated prediction accuracy of our Pretrained Neuroblastoma Model by using <b>valid</b> dataset.<br>
Please move to ./projects/Neuroblastoma/ directory, and run the following bat file.<br>
<pre>
>2.evalute.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../TensorflowUNetEvaluator.py ./train_eval_infer.config
</pre>
The evaluation result is the following.<br>
<img src="./asset/evaluate_console_output_at_epoch_100_0816.png" width="720" height="auto"><br>
<br>

<h2>
5 Inference 
</h2>
We have also tried to infer the segmented region for 
<pre>
images_dir    = "./test/images" 
</pre> dataset defined in <b>train_eval_infer.config</b>,
 by using our Pretrained Neuroblastoma UNet Model.<br>
This <b>./test/images</b> dataset has just been taken from the original <b>groundtruth_svgs</b> folder, which contains
18 jpg files of 1280x1024 pixel size.
<pre>
./S-BSST265/dataset
├─groundtruth_svgs
</pre>
Please move to ./projects/Neuroblastoma/ directory, and run the following bat file.<br>
<pre>
>3.infer.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../TensorflowUNetInferencer.py ./train_eval_infer.config
</pre>

<b><a href="./projects/Neuroblastoma/test/images">Test input images</a> </b><br>
<img src="./asset/test_images.png" width="1024" height="auto"><br>
<br>
<b><a href="./projects/Neuroblastoma/test/masks">Test input ground truth mask</a> </b><br>
<img src="./asset/test_masks.png" width="1024" height="auto"><br>
<br>

<b><a href="./projects/Neuroblastoma/test_output/">Inferred images </a>test output</b><br>
<img src="./asset/test_output.png" width="1024" height="auto"><br>
<br>
<br>


<b><a href="./projects/Neuroblastoma/test_output_merged">Inferred merged images (blended test/images with 
inferred images)</a></b><br>
<img src="./asset/test_output_merged.png" width="1024" height="auto"><br><br>

<b>Some enlarged input images and inferred merged images</b><br>
<table>
<tr>
<td>Input Neuroblastoma_0.jpg</td><td>Inferred-merged Neuroblastoma_0.jpg</td>
</tr>
<tr>
<td><img src = "./projects/Neuroblastoma/test/images/Neuroblastoma_0.jpg" width="640" height="auto"></td>
<td><img src = "./projects/Neuroblastoma/test_output_merged/Neuroblastoma_0.jpg"  width="640" height="auto"></td>
</tr>

<tr>
<td>Input Neuroblastoma_2.jpg</td><td>Inferred-merged Neuroblastoma_2.jpg</td>
</tr>
<tr>
<td><img src = "./projects/Neuroblastoma/test/images/Neuroblastoma_2.jpg" width="640" height="auto"></td>
<td><img src = "./projects/Neuroblastoma/test_output_merged/Neuroblastoma_2.jpg"  width="640" height="auto"></td>
</tr>

<tr>
<td>Input Neuroblastoma_4.jpg</td><td>Inferred-merged Neuroblastoma_4.jpg</td>
</tr>
<tr>
<td><img src = "./projects/Neuroblastoma/test/images/Neuroblastoma_4.jpg" width="640" height="auto"></td>
<td><img src = "./projects/Neuroblastoma/test_output_merged/Neuroblastoma_4.jpg"  width="640" height="auto"></td>
</tr>

<tr>
<td>Input Neuroblastoma_6.jpg</td><td>Inferred-merged Neuroblastoma_6.jpg</td>
</tr>
<tr>
<td><img src = "./projects/Neuroblastoma/test/images/Neuroblastoma_6.jpg" width="640" height="auto"></td>
<td><img src = "./projects/Neuroblastoma/test_output_merged/Neuroblastoma_6.jpg"  width="640" height="auto"></td>
</tr>

<tr>
<td>Input Neuroblastoma_10.jpg</td><td>Inferred-merged Neuroblastoma_10.jpg</td>
</tr>
<tr>
<td><img src = "./projects/Neuroblastoma/test/images/Neuroblastoma_10.jpg" width="640" height="auto"></td>
<td><img src = "./projects/Neuroblastoma/test_output_merged/Neuroblastoma_10.jpg"  width="640" height="auto"></td>
</tr>

<tr>
<td>Input Neuroblastoma_16.jpg</td><td>Inferred-merged Neuroblastoma_16.jpg</td>
</tr>
<tr>
<td><img src = "./projects/Neuroblastoma/test/images/Neuroblastoma_16.jpg" width="640" height="auto"></td>
<td><img src = "./projects/Neuroblastoma/test_output_merged/Neuroblastoma_16.jpg"  width="640" height="auto"></td>
</tr>


</table>

<br>
<h3>
References
</h3>
<b>1. An annotated fluorescence image dataset for training nuclear segmentation methods</b><br>
Sabine Taschner-Mandl, M. Ambros 1Peter F. Ambros,Beiske 2Allan Hanbury, <br>
Doerr,Weiss,Berneder, Magdalena Ambros,Eva Bozsaky<br>
<pre>
# Description of the Biostudies dataset S-BSST265
# Title: An annotated fluorescence image dataset for training nuclear segmentation methods
# The dataset is assigned with an open data license (CC0)
# Author: Florian Kromp, 15.04.2020
# Children's Cancer Research Institute, Vienna, Austria
# florian.kromp@ccri.at

</pre>
<pre>
https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST265
</pre>



