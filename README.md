# DocOrNot?

DocOrNot is an image classification model that detect if an image is
likely to be a text document or not. The model is trained on a dataset
that is composed of images of text documents and images that are not.

## The dataset

The DocOrNot dataset was built using:

- RVL CDIP (Small) - https://www.kaggle.com/datasets/uditamin/rvl-cdip-small
- Flickr8k - https://www.kaggle.com/datasets/adityajn105/flickr8k

8k images were taken from the 8k Flickr dataset and from the RVL CDIP one.

## The model

The model was fine-tuned using `facebook/deit-base-distilled-patch16-224`
