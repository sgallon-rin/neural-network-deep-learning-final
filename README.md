# FDU DATA130011 神经网络与深度学习 / Neural Network and Deep Learning - Final Project

## Identification and Synthesizing of Raphael’s paintings from the forgeries

The following [data](https://drive.google.com/folderview?id=0B-yDtwSjhaSCZ2FqN3AxQ3NJNTA&usp=sharing) is provided by Prof. Yang WANG from HKUST. Since this link is in google drive, you can download the file from our [course webpage](http://www.sdspeople.fudan.edu.cn/fuyanwei/course/projects/final_project/Raphael.zip). The data contains 28 digital paintings of Raphael or forgeries. Note that there are both jpeg and tiff files, so be careful with the bit depth in digitization. The following [file](https://docs.google.com/document/d/1tMaaSIrYwNFZZ2cEJdx1DfFscIfERd5Dp2U7K1ekjTI/edit) contains the labels of such paintings.

#### Questions

1. Can you exploit the known Raphael v.s. Not Raphael data to predict the identity of those 6 disputed paintings (maybe Raphael)? The following papers ([1](http://www.sdspeople.fudan.edu.cn/fuyanwei/course/projects/final_project/artistic_poster.pdf), [2](http://dx.doi.org/10.1016/j.acha.2015.11.005)) might be some references for you.

2. We need to synthesize the painting of Raphael. That is, given one photo, we need to generate/synthesize a new photo that makes it look like Raphael. There are lots of recent works on this topic. You can try it by either (1) Generative Adversarial Networks, or Convolutional Neural Networks, or Attribute transfer. The testing images are downloaded from [here](http://www.sdspeople.fudan.edu.cn/fuyanwei/course/projects/final_project/test_images.zip). You can use the images in [here](http://www.sdspeople.fudan.edu.cn/fuyanwei/course/projects/final_project/Raphael.zip) as well as other available images online to train your model. Your report should show the effects of synthesized testing images.

#### Minimum requirements

The minimum requirements include:

1. classification tasks in Question-1;

2. synthesizing tasks in Question-2.
