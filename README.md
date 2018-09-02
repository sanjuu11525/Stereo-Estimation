
# Stereo disparity Estimation
The purpose of this project is to implement stereo estimation with
Semi-Global Method. At the current stage, a few publications apply this method to stereo esti-
mation. The pixel-wise cost function with Census transformation
is involved. During cost computation, potential targets are selected based
on mapping of corresponding epipolar geometry. In particular, the geometrical representation
gives 1-D pixel-wise optimization. Afterward, aggregating cost by sweeping
the image domain is performed. 

The implementation posted here is much for understanding sharing and education purpose. 
All work is from my interest of computer vision and without carefully refactoring.

## Getting Started

### Basic Knowledge
Variational methods, formulating energy in terms of data differences and regularizations, are
popular in estimating the motion of observed structures such as optical flow. By minimizing the
energy function, a two-dimension vector $(u,v)_p$ representing the flow of pixel $p$ can be found.
Following the same idea, the matching-based algorithm used in this thesis particularly measures
intensity differences in paired points, which can be thought of the data term. The regulariza-
tion term usually depends on a transformation of points.

The relevant mathematical models of the semi-global method will be presented in this section.
For outdoor scenes, a cost function proposed by K. Yamaguchi[1] behaves potentially accurate
on the KITTI benchmark. As already mentioned, the cost function can be replaced with the
original one from Hirschmuller[2].

$$\begin{equation*}
C(p, d)=\sum_{s\in\Sigma_p}|D_{img}(s, e(s))−D_{img´}(s´(s, d), e(s))| + \rho_{census}H(\Phi_l(s), \Phi_r(s (s, d)))
$$\end{equation*}

where $$C(p, d)$$ represents the evaluated cost by $p$ to the point with distance $d$. For a better
quality of matching, a local window $Ω$ surrounding pixel $p$ is created and $s$ represents the sub-
pixel. $D_l(s, e(s))$ is the directional derivative of s along epipolar line $e(s)$ in the left image
and $D_r(s{´}(s, d))$ represents candidate $s{´}(s, d)$ in the right image. Census transformation and
Hamming distance are denoted by $\Phi$ and $H$ respectively. The main idea of this equation is that the set of costs of pixel $p$ in the left image is calculated by knowing candidate $p$ with distance $d$ in the right image. The exact $p$ is decided by the epipolar line $e(p)$. Meanwhile, Hamming distance with coefficient $ρ$ is added after Census transformation has been computed. As being reminded, the stereo case does encode the epipolar information implicitly. In contrast, $e(p)$ must be computed in the flow estimation.

After calculating matching costs, the next step is to get information by aggregating costs from
different directions with extra penalties.

\begin{equation*}
L_j(p, d) = C(p, d) + min\{L_j(p − j, i) + Penalty\}
\end{equation*}

where $L_j(p, d)$ denotes the aggregated cost of pixel $p$ with disparity $d$ and $j$ represents the
directions of aggregations.
After running each direction of aggregations, the total energy of pixel p with d can be gathered
simply by

\begin{equation*}
S(p, d)=\sum_j L_j (p, d)
\end{equation*}

Consequently, disparity D map of pixel p can be determined by

\begin{equation*}
D_{map}(p, d_min)=\min S(p, d)
\end{equation*}

It is worth to note that the computation can be performed in both ways, using left frames as the reference
then to match right frames or the opposite. If applications have a higher tolerance in increased
runtime or hold more hardware resource for high performance computing, implementing both
ways and running consistency check are highly recommended. 


### Dependencies
This implementation depends on OpenCV library.

## Building
```sh
mkdir build; cd build; cmake ..; make -j
```

## Reference
[1]http://ttic.uchicago.edu/~dmcallester/SPS/index.html

[2]https://elib.dlr.de/73119/1/180Hirschmueller.pdf
