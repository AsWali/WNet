# Reproducing a Deep Learning Paper
### By Asror Wali & Erwin Russel

![Header image](https://i.ibb.co/Cwhjm9T/zwaan-crf2.png)

This is a short blog about a reproduction of the paper "W-Net: A Deep Model for Fully Unsupervised Image Segmentation".
This reproduction is part of the CS4240 Deep Learning course at the TU Delft.

We were asked to choose from a list of papers, where this was the one we found most interesting.
There was a Github repository attached to the paper which we could use, in this blog we will describe the process of using this code repository and adapting it to achieve the results that were displayed in the paper.
Next to reproducing the cool segmentation abilities of the WNet architecture, we also want to see how our reproduction benchmarked against the model in the paper. There is a table explaining various benchmarks that we will go into in this blog. 

The dataset is discussed following the losses and postprocessing. We are showing preliminary results on a small dataset, since the original paper trained the model 50000 epochs on 17000 images  this was not attainable given the resources given (google cloud and educational credit). We therefore show one of the results training on a miniscule dataset size of only 10 images running with 100 epochs. 

## Dataset

W-Net uses two different datasets, they train the network using the PASCAL VOC2012 dataset and evaluate the network using the Berkeley Segmentation Database (BSDS 300 and BSDS 500).  The PASCAL VOC2012 dataset is a large visual object classes challenge which contains 11,530 images and 6,929 segmentations. BSDS300 and BSDS500 have 300 and 500 images including human-annotated segmentationas ground truth. 

## The Model

The W-Net code is publicly available on GitHub. Although it is provided in an incomplete state, the 2 U-nets have been implemented but it's missing the Relu activation and the dropout mentioned in the W-Net paper. The script provided to train the model is also not implemented. There is no logic for importing images and training the network. But even if that was present the loss functions used in the paper are not implemented. Instead they used kernels for vertical and horizontal edge detection to optimize the encoder and an unusual mean function for the decoder.

## Losses

W-Net uses 2 loss functions, a reconstruction loss and a soft normalized cut loss.  The reconstruction loss is used to train the decoder so it can generate the original input image X using the segmented output of the encoder. This loss forces the encoder representations to contain as much information of the original image X as possible. And the soft n-cut loss maximizes the total association within the groups created by the encoder, so pixels that are similar in region space and value are grouped together more often. 

Both of these losses converge showing that their approach which balances minimizing the reconstruction loss and maximizes the total association within groups in the encoder.

## Post-processing

Since the output of the encoder shows the hidden representation, but is still rough. The postprocessing algorithm can be found in the image under Algorithm 2.
We apply postprocessing to the encoder output. A fully conditional random field is applied to smooth the encoder output and provide a more simple segmentation with larger segments. Furthermore, we see that the probability boundary is computed with the contour2ucm method. We have not implemented this postprocessing step as was discussed with the supervisor for this paper. 

## Results

## Benchmarking

The postprocessed output of the encoder will be benchmarked against the groundtruth of the BSD300 and BSD500 dataset. The benchmarks will include Segmentation Covering, Variation of information and Probabilistic Rand Index. These were not thoroughly explained in the paper and there is additional rescaling and segmentation clustering involved. 

## Conclusion

We have demonstrated a reproduction of the paper "W-net: A Deep Model for Fully Unsupervised Image Segmentation.". This poster is a preview of the final blogpost we are writing for the course CS4240 Deep Learning. We have explained the dataset, the model, losses and postprocessing. Furthermore, we have demonstrated our preliminary results and explained our benchmarking approach of the project. As well as some shortcomings of the paper and the enclosed repository. 



