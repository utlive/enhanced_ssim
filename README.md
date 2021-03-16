# Enhanced SSIM
This repository is a C implementation of SSIM based on the recommendations made in "A Hitchhiker's Guide to SSIM". A major part of this code has been forked from the implementation of SSIM in Netflix's VMAF repository [1], and we have made changes to reflect our recommendations which improve performance and efficiency.

List of improvements over basline SSIM.
1. Uses Rectangular windows, with a configurable size and stride, instead of Gaussian windows, to compute local statistics.
2. Uses integral images (aka summed area tables) to implement rectangular windows, leading to improvement in efficiency.
3. Uses Coefficient of Variation for spatial pooling, which we found to perform better than the baseline arithmetic mean pooling.
4. Viewing distance and device type are accounted for by using a Self-Adaptive Scale Transform (SAST) [2], instead of a constant scaling factor.

This code accompanies the following paper.

A. K. Venkataramanan, C. Wu, A. C. Bovik, I. Katsavounidis and Z. Shahid, "A Hitchhiker’s Guide to Structural Similarity," in IEEE Access, vol. 9, pp. 28872-28896, 2021, doi: 10.1109/ACCESS.2021.3056504.

# References:
[1] https://www.github.com/Netflix/vmaf/

[2] K. Gu, G. Zhai, X. Yang and W. Zhang, "Self-adaptive scale transform for IQA metric," 2013 IEEE International Symposium on Circuits and Systems (ISCAS), Beijing, 2013, pp. 2365-2368, doi: 10.1109/ISCAS.2013.6572353.
