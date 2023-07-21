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

#include "postgres.h"

#include "access/genam.h"
#include "access/generic_xlog.h"
#include "catalog/index.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/indexfsm.h"
#include "storage/smgr.h"
#include "utils/memutils.h"
#include "utils/builtins.h"
#include "utils/rel.h"
#include "utils/array.h"
#include "utils/varlena.h"
#include "access/stratnum.h"
#include "common/base64.h"
#include <float.h>

#include "utils/string_util.h"
#include "ivfpq.h"
#include "kmeans.h"

// temporary clustering data for clustering
typedef struct {
  int    count;
  int    max_count;
  float4 *values;
  float4 *mean;
  int    *k_pos;
  float4 *residual_vec;
  float4 *pq_mean;
} ClusteringData;

typedef ClusteringData *Clustering;

 // State of pase ivfpq index build.  We accumulate one page data here before
 // flushing it to buffer manager.
typedef struct {
  IvfpqState   ivf_state;          // ivfpq index state
  int64          ind_tuples;         // total number of tuples indexed
  MemoryContext  tmp_ctx;            // temporary memory context reset after each tuple
  MemoryContext  init_ctx;           // memory context for initializing
  PqCentroidsData  centroids;          // centroids memory struct
  Buffer         *buf_list;          // buffer list for inverted list 
  char           *centroid_path;     // centroid file path
  int    clustering_sample_ratio;    // clustering_sample_ratio: sample ratio for clustering, just for clustering type 1
  int            k;                  // k: cluster count
  Clustering     clustering;         // clustering data for clustering
} IvfpqBuildState;

//Initialize centroids data 
static bool
InitCentroids(IvfpqBuildState *buildState) {
  PqCentroids       centroids;
  long            size;
  char            *str;
  List            *splits;
  ListCell        *cell;
  PqCentroidTuple   *ctup;
  PqSubvectorTuple  *pqtup;
  IvfpqOptions  *opts;
  FILE            *fp;
  int             loop, i, dim, count, vc, total_sub, subdim;
  float           val;

  loop = 0;
  centroids = &(buildState->centroids);
  opts = &(buildState->ivf_state.opts);

  if (opts->clustering_type == 0) {
    // open file
    if (!(fp = fopen(buildState->centroid_path, "r"))) {
      elog(ERROR, "open centroid file[%s] failed",
          buildState->centroid_path);
    }
    // get file size
    fseek(fp, 0L, SEEK_END);
    size = ftell(fp);
    rewind(fp);
    // read whole data to string
    str = palloc0(size);
    if (!fgets(str, size, fp)) {
      pfree(str);
      elog(ERROR, "read file data failed, size:%ld", size);
    }
    if (!SplitGUCList(pstrdup(str), '|', &splits) ||
        splits->length != 3) {
      list_free(splits);
      pfree(str);
      elog(ERROR, "centroid file format error, splits size:%d",
          splits->length);
    }
    foreach (cell, splits) {
      const char *split = (const char *) lfirst(cell);
      if (0 == loop) {
        dim = -1;
        dim = atoi(split);
        if (dim <= 0) {
          list_free(splits);
          pfree(str);
          elog(ERROR, "dim format[%s] error", split);
        }
        centroids->dim = dim;
        if (centroids->dim != opts->dimension) {
          list_free(splits);
          pfree(str);
          elog(ERROR, "centroids dim[%d] in file not equal to opts[%d]",
              centroids->dim, opts->dimension);
        }
      } else if (1 == loop) {
        count = -1;
        count = atoi(split);
        if (count <= 0) {
          list_free(splits);
          pfree(str);
          elog(ERROR, "count format[%s] error", split);
        }
        centroids->count = count;
      } else if (2 == loop) {
        char *pos = (char*)split;
        centroids->ctups = (PqCentroidTuple *) palloc0(
            buildState->ivf_state.size_of_centroid_tuple * centroids->count);
        i = 0;
        for (; i < centroids->count; ++i) {
          vc = 0;
          while (*pos != '\0' && vc < centroids->dim) {
            if (*pos == ',') {
              pos ++;
              continue;
            }

            val = 0.0f;
            if (!StringToFloat(pos, &val, &pos)) {
              list_free(splits);
              pfree(str);
              pfree(centroids->ctups);
              elog(ERROR, "val[%s] format error", pos);
            }
            ctup = PqCentroidTuplesGetTuple(buildState, i);
            ctup->vector[vc] = val;
            vc ++;
          }
          ctup->head_ivl_blkno = 0;
          ctup->inverted_list_size = 0;
        }
      }
      loop ++;
    }
    list_free(splits);
    pfree(str);
  } else {
    centroids->dim = opts->dimension;
    centroids->count = buildState->k;
    centroids->ctups = (PqCentroidTuple *) palloc0(
        buildState->ivf_state.size_of_centroid_tuple * centroids->count);
    for (i = 0; i < centroids->count; ++i) {
      ctup = PqCentroidTuplesGetTuple(buildState, i);
      memcpy((void*)(ctup->vector),
          (void*)&(buildState->clustering->mean[i * centroids->dim]),
          centroids->dim * sizeof(float4)); 
      ctup->head_ivl_blkno = 0;
      ctup->inverted_list_size = 0;
    }
    total_sub = opts->partition_num * opts->pq_centroid_num;
    subdim = opts->dimension / opts->partition_num;
    centroids->pqtups = (PqSubvectorTuple *) palloc0(
        buildState->ivf_state.size_of_subvector_tuple * total_sub);
    for (i = 0; i < total_sub; ++i) {
        pqtup = PqSubvectorTuplesGetTuple(buildState, i);
        memcpy((void*)(pqtup->vector),
          (void*)&(buildState->clustering->pq_mean[i * subdim]),
          subdim * sizeof(float4));
    }
  }
  buildState->buf_list = (Buffer*) palloc0(sizeof(Buffer) * centroids->count);
  return true;
}

static Page
CreateNewInvertedListPage(Relation index, PqInvertedListTuple *tuple,
    Buffer *buffer, bool needLock) {
  Page page;
  *buffer = IvfpqNewBuffer(index, needLock);
  page = BufferGetPage(*buffer);
  IvfpqInitPage(page, 0);
  Assert(!PageIsNew(page) && !IvfpqPageIsDeleted(page));
  return page;
}

static Page
GetBufferPageForAddItem(Relation index, IvfpqState *state,
    PqInvertedListTuple *tuple, Buffer buffer,
    Buffer *newBuffer, bool needLock) {
  Page page, newPage;
  IvfpqPageOpaque opaque;

  page = BufferGetPage(buffer);
  // whether page is full
  if (PqInvertedListPageGetFreeSpace(state, page) <
      state->size_of_invertedlist_tuple) {
    //free space not enough, new page
    newPage = CreateNewInvertedListPage(index, tuple,
        newBuffer, needLock);
    opaque = IvfpqPageGetOpaque(newPage);
    opaque->next = BufferGetBlockNumber(buffer);
    PqFlushBufferPage(index, buffer, needLock);
    page = newPage;
  }
  Assert(!PageIsNew(page) && !IvfpqPageIsDeleted(page));
  return page;
}

static void 
InvertedListPageAddItem(IvfpqState *state, Page page,
    PqInvertedListTuple *tuple) {
  Pointer           ptr;
  IvfpqPageOpaque opaque;
  PqInvertedListRawTuple *itup;

  opaque = IvfpqPageGetOpaque(page);
  itup = PqInvertedListPageGetTuple(state, page, opaque->maxoff + 1);
  memcpy((Pointer) itup, (Pointer) tuple, state->size_of_invertedlist_tuple);
  opaque->maxoff++;
  ptr = (Pointer) PqInvertedListPageGetTuple(state, page, opaque->maxoff + 1);
  ((PageHeader) page)->pd_lower = ptr - page;
  Assert(((PageHeader) page)->pd_lower <= ((PageHeader) page)->pd_upper);
}

static void
InvertedListFormEncodedTuple(IvfpqState *state, PqInvertedListRawTuple *tuple, PqInvertedListTuple *encoded_tuple, 
    PqCentroidTuple *ctup, PqSubvectorTuple *pqtup) {
    //TODO omp for encoding in different subvector space
    float4 *residual;
    float minDistance;
    int i, j;
    int dim = state->opts.dimension;
    int parnum = state->opts.partition_num;
    int pqnum = state->opts.pq_centroid_num; //pqnum <= 256
    int subdim;
    float4 *subvec, *pqvec;
    uint8_t code;
    float dis;

    residual = (float4 *)palloc0(sizeof(float4) * dim);
    for (i = 0; i < dim; i++) 
        residual[i] = tuple->vector[i] - ctup->vector[i];

    if (dim % parnum != 0)
      elog(ERROR, "Partition num is not a factor of dimension");
    
    subdim = dim / parnum;
    //TODO omp 
    for (i = 0; i < parnum; i++) {
      subvec = residual + i * subdim;
      minDistance = FLT_MAX;
      for (j = 0; j < pqnum; j++) {
          pqvec = ((PqSubvectorTuple *)((char*)pqtup + (i * pqnum + j) * state->size_of_subvector_tuple))->vector;
          dis = fvec_L2sqr(subvec, pqvec, subdim);
          if (dis < minDistance) {
            minDistance = dis;
            code = (uint8_t)j;
          }
      }
      encoded_tuple->encoded_vector[i] = code;
    }
    pfree(residual);
}

static bool
AddTupleToInvertedList(Relation index, IvfpqBuildState *buildState,
    PqInvertedListRawTuple *tuple) {
  int     minPos;
  Page    page;
  Buffer  buffer, newBuffer;
  PqInvertedListTuple *encoded_tuple;
  PqCentroidTuple *ctup;
  PqSubvectorTuple *pqtup;

  newBuffer = 0;
  minPos = 0;

  PqSearchNNFromCentroids(&(buildState->ivf_state), tuple,
      &(buildState->centroids), &minPos);
  if (minPos >= buildState->centroids.count) {
    elog(WARNING, "min pos[%d] error", minPos);
    return false;
  }

  encoded_tuple = (PqInvertedListTuple *)palloc0(buildState->ivf_state.size_of_invertedlist_tuple);
  encoded_tuple->heap_ptr = tuple->heap_ptr;

  ctup = (PqCentroidTuple *)((char*)buildState->centroids.ctups + minPos * buildState->ivf_state.size_of_centroid_tuple);

  InvertedListFormEncodedTuple(&(buildState->ivf_state), tuple, encoded_tuple, ctup, buildState->centroids.pqtups);

  if (buildState->buf_list[minPos] == 0) {
    // first item in invertedlist
    page = CreateNewInvertedListPage(index, encoded_tuple, &buffer, false);
    buildState->buf_list[minPos] = buffer;
  } else {
    // get latest buffer page in the inverted list
    buffer = buildState->buf_list[minPos];
    page = GetBufferPageForAddItem(index, &buildState->ivf_state, encoded_tuple, buffer, &newBuffer, false); 
    PqCentroidTuplesGetTuple(buildState, minPos)->inverted_list_size ++;
    if (newBuffer != 0) {
      PqCentroidTuplesGetTuple(buildState, minPos)->head_ivl_blkno =
        BufferGetBlockNumber(newBuffer);
      buildState->buf_list[minPos] = newBuffer;
    }
  }
  InvertedListPageAddItem(&(buildState->ivf_state), page, encoded_tuple); 
  pfree(encoded_tuple);
  return true;
}

#define CENTROID_HEAD_BLKNO_UPDATE                                \
  do {                                                            \
    cbuffer = ReadBuffer(index, items[0].cblkno);                 \
    LockBuffer(cbuffer, BUFFER_LOCK_EXCLUSIVE);                   \
    gxlogState = GenericXLogStart(index);                         \
    cpage = GenericXLogRegisterBuffer(gxlogState, cbuffer, 0);    \
    ctup = PqCentroidPageGetTuple(state, cpage, items[0].offset);   \
    ctup->head_ivl_blkno = BufferGetBlockNumber(buffer);          \
    ctup->inverted_list_size += 1;                                \
    GenericXLogFinish(gxlogState);                                \
    UnlockReleaseBuffer(cbuffer);                                 \
  } while (0)

static bool
AddTupleToInvertedListForInsert(Relation index, IvfpqState *state,
    IvfpqMetaPageData *meta, PqInvertedListRawTuple *tuple) {
  bool                reverse;
  Page                page, cpage;
  Buffer              buffer, newBuffer, cbuffer;
  PqCentroidSearchItem  items[1];
  PqCentroidTuple       *ctup;
  GenericXLogState    *gxlogState;
  PqInvertedListTuple *encoded_tuple;
  PqSubvectorTuple *pqtuples;

  memset(items, 0, sizeof(PqCentroidSearchItem));

  newBuffer = 0;
  reverse = false;

  PqSearchKNNInvertedListFromCentroidPages(index, state, meta, tuple->vector, 1,
      reverse, items, false);
  if (items[0].cblkno == 0) {
    elog(WARNING, "insert item failed");
    return false;
  }

  pqtuples = PqGetSubvectorTuples(index, state, meta);

  encoded_tuple = (PqInvertedListTuple *)palloc0(state->size_of_invertedlist_tuple);
  encoded_tuple->heap_ptr = tuple->heap_ptr;

  InvertedListFormEncodedTuple(state, tuple, encoded_tuple, items[0].ctup, pqtuples);

  if (items[0].head_ivl_blkno == 0) {
    // first item in invertedlist
    page = CreateNewInvertedListPage(index, encoded_tuple, &buffer, true);
    CENTROID_HEAD_BLKNO_UPDATE;
  } else {
    buffer = ReadBuffer(index, items[0].head_ivl_blkno);
    LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
    page = GetBufferPageForAddItem(index, state, encoded_tuple,
        buffer, &newBuffer, true);
    if (newBuffer != 0) {
      buffer = newBuffer;
      CENTROID_HEAD_BLKNO_UPDATE;
    }
  }

  InvertedListPageAddItem(state, page, encoded_tuple); 
  pfree(pqtuples);
  pfree(encoded_tuple);

  PqFlushBufferPage(index, buffer, true);
  return true;
}

#undef CENTROID_HEAD_BLKNO_UPDATE

static bool
GetVectorFromDatum(IvfpqState *state, Datum value,
    float4 *vector) {
  ArrayType *arr;
  text      *rawText;
  char      *rawData;
  char      dest[1024*1024];
  int       len, dim, i;
  float4    *data;

  if (state->opts.base64_encoded) {
    memset(dest, 0, sizeof(dest));
    rawText = DatumGetTextPP(value);
    rawData = VARDATA_ANY(rawText);
    len = VARSIZE_ANY_EXHDR(rawText);

    dim = pg_b64_decode(rawData, len, dest) / sizeof(float4);
    if (dim != state->opts.dimension) {
      elog(WARNING, "data dimension[%d] not equal to configure dimension[%d]",
          dim, state->opts.dimension);
      return false;
    }
    for (i = 0; i < dim; ++i) {
      vector[i] = ((float4*)dest)[i];
    }
  } else {
    arr = DatumGetArrayTypeP(value);
    data = PASE_ARRPTR(arr); 
    // copy vector data
    dim = PASE_ARRNELEMS(arr); 
    if (dim != state->opts.dimension) {
      elog(WARNING, "data dimension[%d] not equal to configure dimension[%d]",
          dim, state->opts.dimension);
      return false;
    }
    memcpy(vector, data, dim * sizeof(float4));
  }
  return true; 
}
    

// Make invertedlist tuple from values.
static PqInvertedListRawTuple *
InvertedListFormTuple(IvfpqState *state, ItemPointer iptr,
    Datum *values, bool *isnull) {
  PqInvertedListRawTuple *res;

  res = (PqInvertedListRawTuple *) palloc0(state->size_of_invertedlist_rawtuple);
  res->heap_ptr = *iptr;
  if (isnull[0]) {
    elog(WARNING, "vector colum is null");
    pfree(res);
    return NULL;
  }

  if (!GetVectorFromDatum(state, values[0], (float4*)(&res->vector))) {
    pfree(res);
    return NULL;
  }
  return res;
}

static void
CentroidPageAddItem(IvfpqState *state, Page page, PqCentroidTuple *tuple)
{
  Pointer           ptr;
  IvfpqPageOpaque opaque;
  PqCentroidTuple     *ctup;

  opaque = IvfpqPageGetOpaque(page);
  ctup = PqCentroidPageGetTuple(state, page, opaque->maxoff + 1);
  memcpy((Pointer) ctup, (Pointer) tuple, state->size_of_centroid_tuple);
  opaque->maxoff++;
  ptr = (Pointer) PqCentroidPageGetTuple(state, page, opaque->maxoff + 1);
  ((PageHeader) page)->pd_lower = ptr - page;
  Assert(((PageHeader) page)->pd_lower <= ((PageHeader) page)->pd_upper);
}

static void
SubvectorPageAddItem(IvfpqState *state, Page page, PqSubvectorTuple *tuple)
{
  Pointer           ptr;
  IvfpqPageOpaque opaque;
  PqSubvectorTuple     *pqtup;

  opaque = IvfpqPageGetOpaque(page);
  pqtup = PqSubvectorPageGetTuple(state, page, opaque->maxoff + 1);
  memcpy((Pointer) pqtup, (Pointer) tuple, state->size_of_subvector_tuple);
  opaque->maxoff++;
  ptr = (Pointer) PqSubvectorPageGetTuple(state, page, opaque->maxoff + 1);
  ((PageHeader) page)->pd_lower = ptr - page;
  Assert(((PageHeader) page)->pd_lower <= ((PageHeader) page)->pd_upper);
}

static Page
CreateNewCentroidPage(Relation index, Buffer *buffer) {
  Page page;
  *buffer = IvfpqNewBuffer(index, true);
  page = BufferGetPage(*buffer);
  IvfpqInitPage(page, 0);
  Assert(!PageIsNew(page) && !IvfpqPageIsDeleted(page));
  return page;
}

static void
BuildCentroidPages(Relation index, IvfpqBuildState *buildState) {
  PqCentroids           centroids;
  Buffer              metaBuffer;
  Page                metaPage;
  Page                tmpPage;
  Buffer              tmpBuf;
  Page                newPage;
  Buffer              newBuf;
  IvfpqPageOpaque   opaque;
  int                 i;
  int                 total_pq;
  GenericXLogState    *state;
  IvfpqMetaPageData *meta;

  state = GenericXLogStart(index);
  centroids = &(buildState->centroids);
  metaBuffer = ReadBuffer(index, IVFPQ_METAPAGE_BLKNO);
  LockBuffer(metaBuffer, BUFFER_LOCK_EXCLUSIVE);
  metaPage = GenericXLogRegisterBuffer(state, metaBuffer, 0);
  if (PageIsNew(metaPage) || IvfpqPageIsDeleted(metaPage)) {
    elog(WARNING, "open meta page failed");
    UnlockReleaseBuffer(metaBuffer);
    GenericXLogAbort(state);
    return;
  }
  meta = IvfpqPageGetMeta(metaPage);
  meta->centroid_page_count = 0;
  meta->centroid_num = centroids->count;
  tmpPage = CreateNewCentroidPage(index, &tmpBuf);
  meta->centroid_head_blkno = BufferGetBlockNumber(tmpBuf);

  for (i = 0; i < centroids->count; ++i) {
    if (PqCentroidPageGetFreeSpace(&(buildState->ivf_state), tmpPage) <
        buildState->ivf_state.size_of_centroid_tuple) {
      newPage = CreateNewCentroidPage(index, &newBuf);
      opaque = IvfpqPageGetOpaque(tmpPage);
      opaque->next = BufferGetBlockNumber(newBuf);
      PqFlushBufferPage(index, tmpBuf, true);
      tmpPage = newPage;
      tmpBuf = newBuf;
      meta->centroid_page_count ++;
    }
    CentroidPageAddItem(&buildState->ivf_state, tmpPage,
        PqCentroidTuplesGetTuple(buildState, i));
  }

  // flush last buffer page
  PqFlushBufferPage(index, tmpBuf, true);
  meta->centroid_page_count ++;

  meta->pq_centroid_page_count = 0;
  total_pq = meta->opts.partition_num * meta->opts.pq_centroid_num;
  tmpPage = CreateNewCentroidPage(index, &tmpBuf);
  meta->pq_centroid_head_blkno = BufferGetBlockNumber(tmpBuf);

  for (i = 0; i < total_pq; ++i) {
    if (PqSubvectorPageGetFreeSpace(&(buildState->ivf_state), tmpPage) <
        buildState->ivf_state.size_of_subvector_tuple) {
      newPage = CreateNewCentroidPage(index, &newBuf);
      opaque = IvfpqPageGetOpaque(tmpPage);
      opaque->next = BufferGetBlockNumber(newBuf);
      PqFlushBufferPage(index, tmpBuf, true);
      tmpPage = newPage;
      tmpBuf = newBuf;
      meta->pq_centroid_page_count ++;
    }
    SubvectorPageAddItem(&buildState->ivf_state, tmpPage,
        PqSubvectorTuplesGetTuple(buildState, i));
  }

  // flush last buffer page
  PqFlushBufferPage(index, tmpBuf, true);
  meta->pq_centroid_page_count ++;

  GenericXLogFinish(state);
  UnlockReleaseBuffer(metaBuffer);
}

// centroids call back
static void
IvfpqCentroidsBuildCallback(Relation index, HeapTuple htup, Datum *values,
    bool *isnull, bool tupleIsAlive, void *state) {
  IvfpqBuildState   *buildState;
  int                 rand;

  buildState = (IvfpqBuildState *) state;

  rand = random() % MAX_CLUSTERING_SAMPLE_RATIO;
  if (rand >= buildState->clustering_sample_ratio ||
      buildState->clustering->count >= buildState->clustering->max_count) {
    return;
  }

  if (!GetVectorFromDatum(&buildState->ivf_state, values[0],
        buildState->clustering->values +
        (buildState->clustering->count *
         buildState->ivf_state.opts.dimension))) {
    return;
  }
  buildState->clustering->count ++;
}

// Per-tuple callback from IndexBuildHeapScan.
static void
IvfpqBuildCallback(Relation index, HeapTuple htup, Datum *values,
    bool *isnull, bool tupleIsAlive, void *state) {
  MemoryContext       oldCtx;
  IvfpqBuildState   *buildState;
  PqInvertedListRawTuple   *itup;

  CHECK_FOR_INTERRUPTS();
  buildState = (IvfpqBuildState *) state;
  oldCtx = MemoryContextSwitchTo(buildState->tmp_ctx);

  itup = InvertedListFormTuple(&buildState->ivf_state,
      &htup->t_self, values, isnull);
  if (!itup) {
    elog(WARNING, "itup is NULL");
    MemoryContextSwitchTo(oldCtx);
    MemoryContextReset(buildState->tmp_ctx);
    return;
  }

  if (!AddTupleToInvertedList(index, buildState, itup)) {
    elog(WARNING, "add tuple to inverted list failed");
    MemoryContextSwitchTo(oldCtx);
    MemoryContextReset(buildState->tmp_ctx);
    return;
  }

  // Update total tuple count
  buildState->ind_tuples += 1;
  if (buildState->ind_tuples % 100000 == 0) {
    elog(NOTICE, "build tuple count[%ld]", buildState->ind_tuples);
  }
  MemoryContextSwitchTo(oldCtx);
  MemoryContextReset(buildState->tmp_ctx);
}

// parse parameters from clustering options
static bool
ParseClusteringParams(IvfpqOptions *opts, IvfpqBuildState *buildState) {
  char            *params;
  List            *splits;
  ListCell        *cell;
  int             loop, clustering_sample_ratio, k;

  params = (char *) opts + opts->clustering_params_offset;
  loop = 0;
  clustering_sample_ratio = -1;
  k = -1;

  if (!SplitGUCList(pstrdup(params), ',', &splits) ||
      splits->length != 2) {
    list_free(splits);
    elog(ERROR, "cluster parameters format error, splits size:%d",
        splits->length);
  }
  foreach (cell, splits) {
    const char *split = (const char *) lfirst(cell);
    if (0 == loop) {
      clustering_sample_ratio = atoi(split); 
      if (clustering_sample_ratio <= 0 ||
          clustering_sample_ratio > MAX_CLUSTERING_SAMPLE_RATIO) {
        elog(ERROR, "clustering_sample_ratio[%s] is illegal, should in (0, 1000]",
            split);
      }
      buildState->clustering_sample_ratio = clustering_sample_ratio;
    } else if (1 == loop) {
      k = atoi(split);
      if (k <= 0) {
        elog(ERROR, "k format error[%s]", split);
      }
      buildState->k = k;
    }
    loop ++;
  }
  elog(NOTICE, "parse clustering parameters succeed, clustering_sample_ratio[%d], k[%d]",
      buildState->clustering_sample_ratio,
      buildState->k);
  return true;
}

// Build a new ivfpq index.
IndexBuildResult *
ivfpq_build(Relation heap, Relation index, IndexInfo *indexInfo) {
  IndexBuildResult     *result;
  double               reltuples;
  IvfpqBuildState    buildState;
  IvfpqOptions       *opts;
  int                  i, maxCount;
  MemoryContext        oldCtx;
  double               beginTime, centroidDoneTime, indexDoneTime;

  if (RelationGetNumberOfBlocks(index) != 0) {
    elog(ERROR, "index \"%s\" already contains data",
        RelationGetRelationName(index));
  }

  IvfpqInitMetapage(index);

  memset(&buildState, 0, sizeof(buildState));
  InitIvfpqState(&buildState.ivf_state, index);

  buildState.tmp_ctx = AllocSetContextCreate(CurrentMemoryContext,
      "pase ivfpq build temporary context",
      ALLOCSET_DEFAULT_SIZES);
  buildState.init_ctx = AllocSetContextCreate(CurrentMemoryContext,
      "pase ivfpq build init temporary context",
      IVFPQ_BUILD_INIT_MEM_SIZE,
      IVFPQ_BUILD_INIT_MEM_SIZE,
      IVFPQ_BUILD_INIT_MEM_SIZE);
  oldCtx = MemoryContextSwitchTo(buildState.init_ctx);

  opts = (IvfpqOptions *)index->rd_options;

  if (NULL == opts) {
    elog(ERROR, "ivfpq index must be created with necessary parameters");
  }

  beginTime = elapsed();
  if (opts->clustering_type == 1) {
    maxCount = MAX_CLUSTERING_SAMPLE_COUNT;
    if (MAX_CLUSTERING_SAMPLE_COUNT * opts->dimension * sizeof(float4) >
        MAX_CLUSTERING_MEM) {
      elog(NOTICE, "vector dimension is huge, parameter (clustering_sample_ratio) should be set to ensure the clustering count lower than %d",
          (int)(MAX_CLUSTERING_MEM / (opts->dimension * sizeof(float4))));
      maxCount = MAX_CLUSTERING_MEM / (opts->dimension * sizeof(float4));
    }
    // Do the heap scan for training centroids
    ParseClusteringParams(opts, &buildState);
    buildState.clustering = (Clustering) palloc(sizeof(ClusteringData));
    buildState.clustering->max_count = maxCount;
    buildState.clustering->values = (float4 *) palloc0(
    buildState.clustering->max_count * opts->dimension * sizeof(float4));
    buildState.clustering->count = 0;
    // train clusters by kmeans
    MemoryContextSwitchTo(oldCtx);
    reltuples = IndexBuildHeapScan(heap, index, indexInfo, true,
        IvfpqCentroidsBuildCallback, (void *) &buildState, NULL);
    oldCtx = MemoryContextSwitchTo(buildState.init_ctx);
    buildState.clustering->k_pos = (int *) palloc0(
        buildState.clustering->count * sizeof(int));
    elog(NOTICE, "begin inner kmeans clustering");
    buildState.clustering->mean = basic_kmeans_impl(opts->dimension,
        buildState.k, buildState.clustering->count,
        buildState.clustering->values, false, (float4*)NULL,
        buildState.clustering->k_pos);

    buildState.clustering->residual_vec = residual_impl(
      opts->dimension,buildState.clustering->count,
        buildState.clustering->values,
        buildState.clustering->mean,
        buildState.clustering->k_pos,
        opts->dimension / opts->partition_num);

    pfree(buildState.clustering->values);

    elog(NOTICE, "begin inner pq kmeans clustering");
    buildState.clustering->pq_mean = pq_kmeans_impl(
      opts->dimension,
      buildState.clustering->count,
      buildState.clustering->residual_vec,
      opts->partition_num,
      opts->pq_centroid_num
    );

    if (!InitCentroids(&buildState))
      elog(ERROR, "index \"%s\" InitCentroids failed",
          RelationGetRelationName(index));
    
    pfree(buildState.clustering->k_pos);
    pfree(buildState.clustering->mean);
    pfree(buildState.clustering->residual_vec);
    pfree(buildState.clustering->pq_mean);
    pfree(buildState.clustering);
    MemoryContextSwitchTo(oldCtx);
  } else {
    buildState.centroid_path = (char *) opts +
      opts->clustering_params_offset;
    if (!InitCentroids(&buildState))
      elog(ERROR, "index \"%s\" InitCentroids failed",
          RelationGetRelationName(index));
    MemoryContextSwitchTo(oldCtx);
  }

  centroidDoneTime = elapsed();

  // Do the heap scan
  elog(NOTICE, "begin, ivfpq index building");
  reltuples = IndexBuildHeapScan(heap, index, indexInfo, true,
      IvfpqBuildCallback, (void *) &buildState, NULL);

  // Flush last page if needed in inverted list
  for (i = 0; i < buildState.centroids.count; ++i) {
    if (buildState.buf_list[i] != 0 &&
        IvfpqPageGetOpaque(BufferGetPage(
            buildState.buf_list[i]))->maxoff > 0) {
      PqCentroidTuplesGetTuple((&buildState), i)->inverted_list_size ++;
      if (PqCentroidTuplesGetTuple((&buildState), i)->head_ivl_blkno == 0)
        PqCentroidTuplesGetTuple((&buildState), i)->head_ivl_blkno =
          BufferGetBlockNumber(buildState.buf_list[i]);
      PqFlushBufferPage(index, buildState.buf_list[i], false);
    }
  }

  // build centroid page
  BuildCentroidPages(index, &buildState);

  indexDoneTime = elapsed();

  MemoryContextDelete(buildState.tmp_ctx);
  MemoryContextDelete(buildState.init_ctx);

  result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
  result->heap_tuples = reltuples;
  result->index_tuples = buildState.ind_tuples;
  elog(NOTICE, "ivfpq index build done, build tuple number[%ld], totalTimeCost[%fs], centroidBuildTimeCost[%fs], indexBuildTimeCost[%fs]",
      buildState.ind_tuples, indexDoneTime - beginTime,
      centroidDoneTime - beginTime, indexDoneTime - centroidDoneTime);
  return result;
}

void
ivfpq_buildempty(Relation index) {
  Page		metapage;

  // Construct metapage.
  metapage = (Page) palloc(BLCKSZ);
  IvfpqFillMetapage(index, metapage);

  // Write the page and log it.  It might seem that an immediate sync would
  // be sufficient to guarantee that the file exists on disk, but recovery
  // itself might remove it while replaying, for example, an
  // XLOG_DBASE_CREATE or XLOG_TBLSPC_CREATE record.  Therefore, we need
  // this even when wal_level=minimal.

  PageSetChecksumInplace(metapage, IVFPQ_METAPAGE_BLKNO);
  smgrwrite(index->rd_smgr, INIT_FORKNUM, IVFPQ_METAPAGE_BLKNO,
      (char *) metapage, true);
  log_newpage(&index->rd_smgr->smgr_rnode.node, INIT_FORKNUM,
      IVFPQ_METAPAGE_BLKNO, metapage, false);

  // An immediate sync is required even if we xlog'd the page, because the
  // write did not go through shared_buffers and therefore a concurrent
  // checkpoint may have moved the redo pointer past our xlog record.
  smgrimmedsync(index->rd_smgr, INIT_FORKNUM);
}

bool
ivfpq_insert(Relation index, Datum *values, bool *isnull,
    ItemPointer ht_ctid, Relation heapRel,
    IndexUniqueCheck checkUnique,
    IndexInfo *indexInfo) {
  IvfpqState         ivfpqState;
  MemoryContext        oldCtx;
  MemoryContext        insertCtx;
  Buffer		         metaBuffer;
  IvfpqMetaPageData  *meta;
  PqInvertedListRawTuple    *itup;

  insertCtx = AllocSetContextCreate(CurrentMemoryContext,
      "ivfpq insert temporary context",
      ALLOCSET_DEFAULT_SIZES);
  oldCtx = MemoryContextSwitchTo(insertCtx);
  InitIvfpqState(&ivfpqState, index);
  itup = InvertedListFormTuple(&ivfpqState, ht_ctid, values, isnull);
  if (!itup) {
    return false;
  }

  metaBuffer = ReadBuffer(index, IVFPQ_METAPAGE_BLKNO);
  LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
  meta = IvfpqPageGetMeta(BufferGetPage(metaBuffer));

  AddTupleToInvertedListForInsert(index, &ivfpqState, meta, itup);
  UnlockReleaseBuffer(metaBuffer);
  MemoryContextSwitchTo(oldCtx);
  return false;
}
