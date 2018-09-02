
# Stereo disparity Estimation
The purpose of this project is to implement stereo estimation with Semi-Global Method. At the current stage, a few publications apply this method to stereo estimation. The pixel-wise cost function with Census transformation is involved. During cost computation, potential targets are selected based on mapping of corresponding epipolar geometry. In particular, the geometrical representation gives 1-D pixel-wise optimization. Afterward, aggregating cost by sweeping
the image domain is performed. 

The implementation posted here is much for understanding sharing and education purpose. All work is from my interest of computer vision and without carefully refactoring.

## Basic Knowledge
Variational methods, formulating energy in terms of data differences and regularizations, are popular in estimating the motion of observed structures such as optical flow. By minimizing the energy function, a two-dimension vector $(u,v)_p$ representing the flow of pixel $p$ can be found. Following the same idea, the matching-based algorithm used in this thesis particularly measures intensity differences in paired points, which can be thought of the data term. The regularization term usually depends on a transformation of points.

The relevant mathematical models of the semi-global method will be presented in this section. For outdoor scenes, a cost function proposed by K. Yamaguchi[1] behaves potentially accurate on the KITTI benchmark. As already mentioned, the cost function can be replaced with the original one from Hirschmuller[2].

The pipeline of the semi-global method basically contains three parts. The first part is preprocessing, which includes computations of derivatives, Census transformation, as well as Hamming distance. This part prepares all requirements to
the cost function. The reader could review the implementation and figure out transformations required in the preprocessing. The second part implements the central body of the algorithm. It includes computing costs locally and being aware of global information by path-wise aggregations. The final part operates postprocessing by consistency check and optional interpolation. It is worth to note that the pipeline can be performed in both ways, using left frames as the reference
then to match right frames or the opposite. If applications have a higher tolerance in increased runtime or hold more hardware resource for high performance computing, implementing both ways and running consistency check are highly recommended. The postprocessing is not just to recognize outliers, but also to reconstruct occluded points for labeling objects’ boundaries andsegmentations.

## Dependencies
This implementation depends on OpenCV library.

## Building
```sh
mkdir build
cd build
cmake ..
make -j
```

## Reference
[1]http://ttic.uchicago.edu/~dmcallester/SPS/index.html

[2]https://elib.dlr.de/73119/1/180Hirschmueller.pdf
