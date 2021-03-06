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
 */

int compute_ssim(const float *ref, const float *cmp, int w, int h,
                 int ref_stride, int cmp_stride, int window_type, int window_len, int window_stride, float d2h, int spatial_aggregation_method, double *score,
                 double *l_score, double *c_score, double *s_score, int close_windows_on_end);
