# Stream-51
The Stream-51 dataset for streaming classification and novelty detection from videos.

## Training Protocol
![Stream-51](./repo_images/Stream_protocol.png)

The Stream-51 protocol poses unique challenges, requiring agents to learn from temporally correlated datastreams and recognize unlearned concepts as novel. Training data can be ordered either just by instance or by both class and instance. Evaluation data includes a set of novel examples from classes unseen during training.

## Dataset Coming Soon!

## Citation
If using this code, please cite our paper.
```
@inproceedings{roady2020stream-51,
  title={Stream-51: Streaming Classification and Novelty Detection from Videos},
  author={Roady, Ryne and Hayes, Tyler L and Vaidya, Hitesh and Kanan, Christopher},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2020}
}
