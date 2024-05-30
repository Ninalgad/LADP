# LADP: Label Aware Denoising Pretraining for Lesion Segmentation

The training scrips and algorithm docker container source code for the LADP alpgorithim.

# Overview
LADP is a denoising (self-)pretraining technique that incorporates state-of-the-art techniques and depends on label information. LADP uses the region-of-interest extraction method from CarveMix in order to impart increasing levels of noise to regions surrounding lesion contours. 

![Screenshot](overview.png)
