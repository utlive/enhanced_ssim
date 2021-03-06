/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 * (12/24/2020) Updated by abhinaukumar (abhinaukumar@utexas.edu) to add pooling
 * options and integral image option.
 */

#pragma once

#ifndef IQA_OPTIONS_H_
#define IQA_OPTIONS_H_

/* Whether to use 1D separable convolution */
#define IQA_CONVOLVE_1D

/* Whether to use integral images for custom square windows */
#define USE_IQA_INTEGRAL_IMAGE_MEAN
// #define USE_IQA_CONVOLVE

/* Spatial aggregation methods */
#define COV_POOLING 0
#define MEAN_POOLING 1

#endif /* IQA_OPTIONS_H_ */
