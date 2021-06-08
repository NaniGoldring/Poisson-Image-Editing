# Poisson-Image-Editing
Image blending is a process of transferring an image from the source domain to the target domain while ensuring the transformed pixels conform to the target domain to ensure consistency. Poisson Image Blending was introduced by Perez et al. to perform seamless blending of images. The idea utilizes the sensitivity of human observers to gradients in an image. By exploiting this we obtain a Poisson equation the solution to which yields the algorithm for seamless blending.

This code implements gradient domain fusion of images via three techniques which are discussed in [[1]](#doc1) and [[2]](#doc2):

1. Poisson Blending
2. Mixed Gradients
3. Shepard Blending

The results are shown below:

![alt text](/results/im1.png "Example 1 - Poisson Blending") 

<a id="doc1"></a>
[1] - [P. Perez, M. Ganget, A. Blake - Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)

<a id="doc2"></a>
[2] - [D. Shepard - A two-dimensional interpolation function for irregularly-spaced data](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.154.6880&rep=rep1&type=pdf)
