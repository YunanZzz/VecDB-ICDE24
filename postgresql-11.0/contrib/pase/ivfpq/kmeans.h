// Copyright (C) 2019 Alibaba Group Holding Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ===========================================================================
// copy from https://pgxn.org/dist/kmeans/
//

#ifndef PASE_IVFPQ_KMEANS_H_
#define PASE_IVFPQ_KMEANS_H_

typedef float4 *myvector;
extern myvector
basic_kmeans_impl(int dim, int k, int N, myvector inputs,
        bool initial_mean_supplied, myvector initial_mean, int *r);

extern myvector
residual_impl(int dim, int N, myvector inputs, myvector mean, int *r, int subdim);

extern myvector
pq_kmeans_impl(int dim, int N, myvector inputs, int partition_num, int pq_centroid_num);

#endif  // PASE_IVFPQ_KMEANS_H_
