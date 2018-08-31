
#Running Stereo disparity

1. Check CMake for OpenCV libray  directory

2. Change image directory in sgm_stereo.cpp(be careful of gray image used)

3. Change "visualDisparity" at top of SGM.h for visualization.(currently being 3. Shall be 1 for the regular segamentation)

4. For previous disparity imgs on gitLab, just read those images and divided each pixels by 
   "visualDisparity" and then store back.

5. The code is very slow because of O(WHD). W: width; H:height; D:disparity depth;  The consistency check involves
   computeCost and adggregation twice for stereo images, being the host and target image alternatively.

6. For some stereo pairs, the code doesn't perform very well , for example, training/00061.

7. cd build; cmake ../; make; ./sgm_stereo 
