# DRA-Net
This article aims to improve the segmentation accuracy of white blood cells and proposes a deep
learning network called DRA-Net based on U-Net. DRA-Net is a U-shaped neural network based on
a residual dual-channel mechanism, utilizing an improved encoder-decoder structure to enhance the
interdependence between channels and spatial positions. In the encoding module, the Efficient Channel Attention (ECA) module is connected to the lower layers of the convolutional blocks and residual
blocks to effectively extract feature information. In the decoding module, the Triple Vision module
is connected to the upper layers of the convolutional blocks, eliminating the correspondence between
channels and weights, thereby better extracting and fusing multi-scale features, which enhances the
performance and efficiency of the network. This article uses publicly available Kaggle dataset from
the Core Laboratory of Hospital Clinic in Barcelona and a self-built DML-LZWH (Liuzhou Workers’
Hospital Medical Laboratory) dataset to conduct experiments on medical image segmentation tasks.
In the self-built DML-LZWH dataset, compared to the U-Net network model, the IoU improved by
3% and the Dice improved by 2.3%.In the Kaggle public dataset from the Core Laboratory of Hospital
Clinic in Barcelona, the IoU improved by 4.3% and the Dice improved by 3.1%. These results validate
the effectiveness of the DRA-Net algorithm, and the experimental results indicate that the performance
of the DRA-Net algorithm is significantly better than existing segmentation algorithms. Furthermore,
when compared to the state-of-the-art (DA-TransUNet) model, DRA-Net also shows a significant
performance improvement. The experimental methods and related data in this article will be opensourced at: https://github.com/W-JFenf/DRA-Net
