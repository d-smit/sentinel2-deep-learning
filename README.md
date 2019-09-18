# Novel Training Methodologies for Land Classification of Sentinel-2 Imagery

## Project Overview

The goal of this project was to develop training methodologies for land-cover prediction using Sentinel-2 imagery. A presentation, given to a technical and non-technical audience, can be found [here](presentation.pdf).

We were aiming to consider spatial and textural elements of a scene, in order to assess their impact on classification. We had access to a single Sentinel-2 tile of South-West England. Our ground-truth was Corine Land-Cover mapping for the same area.  

## Data Preparation 

Running ```scene_prep.py``` will load in our test image. We were only considering bands R,G,B and NIR. The script merges our 4 band tifs into one raster.  

Initially having access to the 13 spectral bands captured by Sentinel-2, we merged RGB and Near Infrared bands:

![](/notes/s2_aoi1.png)

We then cropped and filtered. We increased the contrast and reduced the image to approximately 40km by 11km:

![](/notes/s2_newaoi.png)

Our ground-truth was Corine Land-Cover mapping for the same area.

## Segmentation 

First we wanted to consider the image texture, or colour. By setting ```Segment = True``` in ```scene_prep.py```, we could load in our image, import ```segment.py``` and segment the AOI. By splitting the AOI into segmented objects and running statistics on each segment, we could assign a value to each:

![](/notes/s2_seg1.png)

We would then use the segment values, band values and Corine labels to form a training set. A pixel was given the segment value of the segment that contained it. We sampled our pixels using the ```sample_raster``` function, which randomly samples ```n``` pixels from the AOI. 

![](/notes/s2_seg2.png)

We were aiming to predict the Corine label for each pixel. The inclusion of the segment values would ideally improve the classification, based on the assumption that two separate pixels with the same segment type would more likely belong to the same class, making segment type a useful predictor. In ```test_segment.py```, we load in our dataset of pixel band values, segment types and Corine labels. We partition into training splits, and construct a triple-layered MLP. We also can run a baseline Random Forest, and an MLP trained without the segment type variable, to observe its effect.

We can see the results of this below:

### Predictions

![](/notes/s2_segresults.png)

Our results showed slight increases in accuracy when including the segment tree level as a feature on our dataset of pixels. This was positive, as it meant textural considerations would improve accuracy. 

## Patch-Based Training 

Next we wanted to experiment with spatial relations in the image. To do this, we wanted to train on patches of pixels, which meant exploring CNNs. Using ```scene_prep.py``` with ```Segment=False```, we sampled individual pixels. By entering ```sample_raster``` and applying a ```buffer``` to each sampled pixel, we could sample patches instead of pixels. 

![](/notes/s2_patch1.png)

The values of each patch then formed individuals of a new dataset. This dataset is loaded in ```patches.py```. We wanted to learn a sampled pixels surrounding neighbourhood to improve classification of that individual pixel. Therefore our target was the label of each patch centre pixel. In ```patches.py``` we construct our CNN model:

![](/notes/s2_patchcnn.png)
![](/notes/s2_patchtables.png)

The model was trained on up to 400,000 patches taken from the AOI. The patch-size ranged from 3x3 to 61x61, with the two top-performing models kept. We then wanted to predict over the entire scene. In ```test_patches.py``` we partition our AOI into overlapping tiles. This allows us to make a prediction for the entire image, shown below for a CNN trained on 7x7 patches.

### Predictions

![](/notes/s2_patchpreds1.png)

We can take a closer look at the different patch size model predictions: 

![](/notes/s2_patchcomp2.png)

## Transfer Learning with BigEarthNet

We also wanted to experiment with transfer learning for the patch-based approach. BigEarthNet is one of the largest archives of Sentinel-2 data currently available. Each image is a 120x120 patch, with 13 spectral bands available per location. We load in the dataset in ```read_bigearth.py```. Choosing the same 4 bands we had used for the previous approaches, we reduced the dataset to images of only the classes represented in our AOI. We could then train a model in ```train_bigearth.py```. 

Our model struggled to get above 40% accuracy when testing on our AOI. 

![](/notes/s2_bigearth.png)

This could be due to the large patch sizes in the BigEarth dataset, which can overload the model with a high number of pixel values. 

# Discussion 

We saw that considering patches of pixels produces excellent shape detection and a moderate accuracy (highest over AOI was 66%). This was a greater accuracy than considering textural elements alone, making spatial considerations more influential for classifying images of this nature. We also discovered that the size of patches is a strong determinant for accuracy.
