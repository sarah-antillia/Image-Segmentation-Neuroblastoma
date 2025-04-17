<h2>Tensorflow-Image-Segmentation-Neuroblastoma (Updated: 2025/04/18)</h2>

Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>

<li>2025/04/20: Updated to use the latest Tensorflow-Image-Segmentation-API</li>
<br>
This is the first experiment of Image Segmentation for Neuroblastoma 
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and  <a href="https://drive.google.com/file/d/1VAySrFfcHS9LwqfdId-nL7pt03eLV9tm/view?usp=sharing">
Neuroblastoma-ImageMask-Dataset.zip</a>, which was derived by us from
<a href="https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST265">
<b>An annotated fluorescence image dataset for training nuclear segmentation methods</b>
</a>
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 1280x1024 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/images/Neuroblastoma_0.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/masks/Neuroblastoma_0.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test_output/Neuroblastoma_0.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/images/Neuroblastoma_3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/masks/Neuroblastoma_3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test_output/Neuroblastoma_3.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/images/Neuroblastoma_9.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/masks/Neuroblastoma_9.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test_output/Neuroblastoma_9.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this NeuroblastomaSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
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
<b>License: CC0</b>

<br>
<h3>
<a id="2">
2 Neuroblastoma ImageMask Dataset
</a>
</h3>
 If you would like to train this Neuroblastoma Segmentation model by yourself,
 please download our 512x512 pixels dataset from the google drive  
<a href="https://drive.google.com/file/d/1VAySrFfcHS9LwqfdId-nL7pt03eLV9tm/view?usp=sharing">
Neuroblastoma-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Neuroblastoma
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
On the derivation of this dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py.</a></li>
<br>
<br>
<b>Neuroblastoma Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/Neuroblastoma_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We trained Neuroblastoma TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Neuroblastomaand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Enabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = True
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (93,94,95)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 27  by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/asset/train_console_output_at_epoch_27.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/eval/train_losses.png" width="520" height="auto"><br>
<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Neuroblastoma.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/asset/evaluate_console_output_at_epoch_27.png" width="720" height="auto">
<br><br>Image-Segmentation-Neuroblastoma

<a href="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/evaluation.csv">evaluation.csv</a><br>
The loss (bce_dice_loss) to this Neuroblastoma/test was low, and dice_coef high as shown below.
<br>
<pre>
loss,0.0528
dice_coef,0.9363
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Neuroblastoma.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/asset/mini_test_output.png" width="1024" height="auto"><br>

<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/images/Neuroblastoma_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/masks/Neuroblastoma_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test_output/Neuroblastoma_1.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/images/Neuroblastoma_3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/masks/Neuroblastoma_3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test_output/Neuroblastoma_3.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/images/Neuroblastoma_5.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/masks/Neuroblastoma_5.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test_output/Neuroblastoma_5.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/images/Neuroblastoma_9.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/masks/Neuroblastoma_9.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test_output/Neuroblastoma_9.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/images/Neuroblastoma_13.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/masks/Neuroblastoma_13.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test_output/Neuroblastoma_13.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/images/Neuroblastoma_15.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test/masks/Neuroblastoma_15.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Neuroblastoma/mini_test_output/Neuroblastoma_15.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
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



