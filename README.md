# Novel Training Methodologies for Land Classification of Sentinel-2 Imagery

## Project Overview

The goal of this project was to develop training methodologies for land-cover prediction using Sentinel-2 imagery. We had access to a single Sentinel-2 tile of South-West England. Our ground-truth was Corine Land-Cover mapping for the same area.  

## Data Preparation 

Running ```scene_prep.py``` will load in our test image. The script merges our 4 band tifs into one raster, and samples a training set of pixels from the raster.  

Initially having access to the 13 spectral bands captured by Sentinel-2, we merged RGB and Near Infrared bands:

![](/notes/s2_aoi1.png)

We then cropped and filtered. We increased the contrast and reduced the image to approximately 40km by 11km:

![](/notes/s2_aoi2.png)

Our ground-truth was Corine Land-Cover mapping for the same area:

![](/notes/s2_corine1.png)

## Segmentation 

Our first methodology was segmenting the AOI using graph-based segmentation. Zonal statistics were acquired for each segment, and a hierarchy was formed. The process, shown below, was carried out using the functions in ```segment.py```. This was run on our AOI by running ```scene_prep.py``` with ```Segment = True```.

![](/notes/s2_seg2.png)


We were considering NN architectures. We can see this below. 

** SEG MODEL **

Our results showed slight increases in accuracy when including the segment tree level as a feature on our dataset of pixels. We now wanted to experiment with spatial relations in the AOI.

## Patch-Based Training 

To do this, we wanted to train on patches of pixels, which also opened up the potential classification increase of convolutional layers from CNNs. We sampled individual pixels and constructed patches, as seen below.

** PATCH CONSTRUCTION DIAGRAM **

The values of each patch then formed individuals of a new dataset. We wanted to learn a sampled pixels surrounding neighbourhood to improve classification of that individual pixel. By partitioning our AOI into overlapping tiles, we would be 
able to train on patches of pixels and still provide pixel-level classification. 

We can visualise the partitioning below. 

** SLIDING WINDOW METHOD **

We were also able to experiment with CNNs. Using VGG-like structural considerations, we formed a model of repeating conv blocks, with no pooling. This was due to the small size of our patches. Using deep blocks, we constructed the model below.

** PATCH MODEL 1 **

We can see the results of the patch-based training over a range of patch-sizes below, compared against pixel only classifiers. 

** TABLE OF PATCHES **
** AOI PREDS ** 
** AOI CONFIDENCE **

## Transfer Learning with BigEarthNet

We also wanted to experiment with transfer learning for the patch-based approach. BigEarthNet is one of the largest archives of Sentinel-2 data currently available. Each image is a 120x120 patch, with 13 spectral bands available per location. Choosing the same 4 bands we had used for the previous approaches, we reduced the dataset to images of only the classes represented in our AOI. We could then train a model on this and use it to predict over our AOI. We can see the model choice and results of the AOI-wide predictions below. 

** big earth model **

** big earth AOI preds **

# Discussion 
