# Human feature perception and computer vision estimates of melanoma features

Here we describe the process we followed to generate perceptual strength estimates of the ABC features for 10,000 images based on human perception and a computer vision algorithm. We then used these features to train a set of SVM models to examine the diagnostic utility of these features and found a benefit of combining human and computer perception.

Please follow the guide below to setup the project, extract the same features, conduct the analyses, and see the results!

## Assumptions, requirements and notes

- Python >=3.12.9

## Git setup

```
git clone https://github.com/MurraySBennett/melanoma-identification.git
```

## Venv setup

Create and activate your python virtual environment however you like to do it. Or skip to the final code-block of this section to install the requirements globally if you like the thrill of potentially conflicting packages/versions. To each their own.

```
cd melanoma-identification
python -m venv mel_venv

# then, depending on your system, activate it
# mac/Linux
source .mel_venv/bin/activate

# PowerShell
.mel_venv\Scripts\Activate.ps1

# CommandPrompt
.mel_venv\Scripts\activate.bat

```

Finally, install requirements, the appropriate file is located within the 'pwc' directory, so can be called from within that directory.

```
cd pwc
pip install -r requirements.txt
cd ..
```

## Computer vision estimation

### Image segmentation

We will start off segmenting the images - this involves (the attempt of) identifying the lesion within each image and saving the binary mask to the images/masks directory. These masks are used for shape and colour estimation in the following section.

From within the melanoma-identification directory:

```
python -m pwc.cv.00_segmentation.image_segmentation
```

### Shape and colour estimates

We can then begin with the shape and colour estimates. These scripts generate symmetry and border regularity measures into one file and colour variance into another.
Estimates are generated for each image based off the generated masks, output as `melanoma-identification/pwc/cv/01_feature-analysis/[cv_shape][cv_colour].txt`. We then merge the two files into the `melanoma-identification/pwc/data/estimates/cv-data.csv` for later wrangling.

```
python -m pwc.cv.01_feature-analysis.00_shape_analysis
python -m pwc.cv.01_feature-analysis.01_colour_continuous
python -m pwc.cv.01_feature-analysis.02_merge_cv_data
```

**Note:** _for those of you snooping around the directories, you will notice the 03_select_images, 04_createImageList, and 05_select_practice_images. These were originally required when we were filtering down from ~75,000 images to the final image set. For brevity/space, we only provide the final 10,000 images used in the analysis, so these final scripts aren't strictly necessary, but are included for completeness._

## BTL estimation

### Human data processing

At this point, we then head out to collect our data using a pairwise choice experiment (hence the 'pwc' you are seeing in the directory names). The experiment file can be reviewed and run by uploading the `melanoma-identification/pwc/btl_melanoma.study.json` file to
[lab.js](https://labjs.felixhenninger.com/).

We hosted our images online to allow rapid online data collection, but have since paused this hosting mechanism. You can still run the task locally, but will need to alter the `img_root` and `practice_root` path variables in the 2nd custom script window of the 'Welcome' screen to point to the image location (assuming you're working in the lab.js builder, you may need to scroll down).

Following data collection, we saved the raw data to `melanoma-identification/pwc/data/raw/`. We will now work through the scripts that cleaned and processed this data to generate a set of more managable master files for analysis `melanoma-identification/pwc/data/cleaned/[btl-asymmetry][btl-border][btl-colour].csv`:

```
python -m pwc.scripts.00_read_data
```

For those with a preference for R, the same preprocessing can be run with `00_read_data.R`

### BTL model estimation

The Bradley-Terry-Luce model estimates the 'perceptual strength' of each image along the prompted perceptual dimension. These are estimated via logistic regression. The scores for each image are output to the `melanoma-identification/pwc/data/estimates` directory.

Higher scores reflect greater 'severity' of the feature. For example, high scores on a dimension indicate greater asymmetry, border irregularity, or colour variance.

We then need to merge the BTL and CV estimates for the next steps (BTL estimates are also saved separately).

```
python -m pwc.scripts.01_apply_BTL
python -m pwc.scripts.02_merge_btl_cv
```

## Descriptive Plots

Here we generate face-validity plots for the BTL estimates for a visual inspection and confirmation that images with high estimates tend to be 'greater/stronger/more-er' along the relevant dimension. We save `btl_facevalidity.pdf` to `melanoma-identification/pwc/figures/`.
An optional plot of the feature variance can also be generated. This plot was generated for presentations and really is just another version of the `btl_facevalidity` figure.

```
python -m pwc.scripts.03_btl_fv
python -m pwc.scripts.06_feature_variance
```

_Optional:_ We also used the following script for a quick visual check (of what is just a logistic function) of the raw strength estimates.

```
python -m pwc.scripts.03a_plt_BTL_estimates
```

Next, we generate a scatter plot of the BTL estimates against the CV estimates, including measures of their correlation. Correlations are calculated with the complete data, but plots use a subset of points to aid image size and rendering. Plots are saved to `melanoma-identification/pwc/figures/`.

```
python -m pwc.scripts.04_plt_correlations
```

## SVM analysis

We now assess the diagnostic utility of the human perception and computer vision algorithm derived estimates by training 3 Support Vector Machine (SVM) models. Each model is trained and evaluated on a 5-fold cross validation and make use of different feature sets:

1. BTL
2. Computer Vision
3. BTL and Computer Vision

The complete process involves three scripts.
The first two scripts determine model configurations and run permutation tests to assess significant differences between the model predictions. If you're willing to take our word for it (particularly in terms of the 'different from random performance', which takes the longest, then skip to the third script: 05c_svm_plot.py.

We start by performing a grid search on the regularisation parameter, $C$, and the radial basis function's kernel coefficient, $\gamma$, for each of the models. We save the grid search outputs for a visual inspection and determination of the best configuration across all models.

```
python -m pwc.scripts.05a_svm_parameter_search
```

We then load the model configurations, train each model, and conduct permuation tests to compare performance against chance and performance between models. Navigate to the end of the script to toggle the exectuion of each test. The randomness test (`test_random`) takes a while to run, but the more interesting _between_ model test is rather fast (`test_between`).

```
python -m pwc.scripts.05b_svm_tests
```

Finally, let's plot the ROC curve. Shaded error is based on the variance across the 5-fold cross validation.

```
python -m pwc.scripts.05c_svm_plot
```
