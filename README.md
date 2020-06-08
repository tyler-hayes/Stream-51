# Stream-51
The Stream-51 dataset for streaming classification and novelty detection from videos. [Paper link](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Roady_Stream-51_Streaming_Classification_and_Novelty_Detection_From_Videos_CVPRW_2020_paper.pdf)

## Training Protocol
![Stream-51](./repo_images/Stream_protocol.png)

The Stream-51 protocol poses unique challenges, requiring agents to learn from temporally correlated datastreams and recognize unlearned concepts as novel. Training data can be ordered either just by instance or by both class and instance. Evaluation data includes a set of novel examples from classes unseen during training.

## Dataset Coming Soon!

## Citation
If using this code, please cite our paper.
```
@InProceedings{Roady_2020_Stream51,
author = {Roady, Ryne and Hayes, Tyler L. and Vaidya, Hitesh and Kanan, Christopher},
title = {Stream-51: Streaming Classification and Novelty Detection From Videos},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}
