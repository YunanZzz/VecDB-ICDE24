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

#include <pthread.h>
#include <omp.h>
#include "access/relscan.h"
#include "pgstat.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/vector_util.h"
#include "ivfpq.h"

static const bool item_reverse = false;

static int
PairingHeapItemCompare(const pairingheap_node *a, const pairingheap_node *b,
    void *arg) {
  const PqInvertedListSearchItem *ia = (const PqInvertedListSearchItem *)a;
  const PqInvertedListSearchItem *ib = (const PqInvertedListSearchItem *)b;
  bool *reverse = (bool*) arg;
  if (ia->distance > ib->distance) {
    if (*reverse) {
      return 1;
    } else {
      return -1;
    }
  }
  else if (ia->distance < ib->distance) {
    if (*reverse) {
      return -1;
    } else {
      return 1;
    }
  } else {
    return 0;
  }
}

// Begin scan of ivfpq index.
IndexScanDesc
ivfpq_beginscan(Relation r, int nkeys, int norderbys) {
  IndexScanDesc     scan;
  IvfpqScanOpaque so;
  MemoryContext     scanCxt, oldCtx;

  scan = RelationGetIndexScan(r, nkeys, norderbys);
  scanCxt = AllocSetContextCreate(CurrentMemoryContext,
      "ivfpq scan context",
      ALLOCSET_DEFAULT_SIZES);
  oldCtx = MemoryContextSwitchTo(scanCxt);

  so = (IvfpqScanOpaque) palloc0(sizeof(IvfpqScanOpaqueData));
  InitIvfpqState(&so->state, scan->indexRelation);
  so->scan_pase = NULL;
  so->queue = NULL;
  so->first_call = true;
  so->scan_ctx = scanCxt;
  so->queue = pairingheap_allocate(PairingHeapItemCompare, (void*)&item_reverse);

  scan->opaque = so;

  if (scan->numberOfOrderBys > 0) {
    scan->xs_orderbyvals = palloc0(sizeof(Datum) * scan->numberOfOrderBys);
    scan->xs_orderbynulls = palloc(sizeof(bool) * scan->numberOfOrderBys);
    memset(scan->xs_orderbynulls, true, sizeof(bool) * scan->numberOfOrderBys);
  }

  MemoryContextSwitchTo(oldCtx);
  return scan;
}

// Rescan a ivfpq index.
void
ivfpq_rescan(IndexScanDesc scan, ScanKey scankey, int nscankeys,
    ScanKey orderbys, int norderbys) {
  MemoryContext          oldCtx;
  IvfpqScanOpaque      so;
  PqInvertedListSearchItem *item;

  so = (IvfpqScanOpaque) scan->opaque;
  oldCtx = MemoryContextSwitchTo(so->scan_ctx);

  if (so->queue != NULL) {
    if (!pairingheap_is_empty(so->queue)) {
      item = (PqInvertedListSearchItem *) pairingheap_remove_first(so->queue);
      pfree(item);
    }
    pairingheap_free(so->queue);
    so->queue = NULL;
  }
  so->scan_pase = NULL;
  so->queue = pairingheap_allocate(PairingHeapItemCompare, (void*)&item_reverse);
  so->first_call = true;

  if (scankey && scan->numberOfKeys > 0) {
    memmove(scan->keyData, scankey,
        scan->numberOfKeys * sizeof(ScanKeyData));
  }
  if (orderbys && scan->numberOfOrderBys > 0) {
    memmove(scan->orderByData, orderbys,
        scan->numberOfOrderBys * sizeof(ScanKeyData));
  }
  MemoryContextSwitchTo(oldCtx);
}

// End scan of ivfpq index.
void
ivfpq_endscan(IndexScanDesc scan) {
  IvfpqScanOpaque so = (IvfpqScanOpaque) scan->opaque;
  MemoryContextDelete(so->scan_ctx);
}

static float
CalDistanceForEncoded_L2sqr(IvfpqState *state, IvfpqMetaPageData *meta, float4 *residual, 
uint8_t *encoded_vector, PqSubvectorTuple *pqtups, float4 *precomputedTable) {
    int partition_num = meta->opts.partition_num;
    int pq_centroid_num = meta->opts.pq_centroid_num;
    int dim = meta->opts.dimension;
    int subdim;
    int i;
    float4 *generated_vector;
    float4 result=0;
    float4 * table = precomputedTable;

    Assert(dim % partition_num == 0);
    subdim = dim / partition_num;

    if (meta->opts.use_precomputedtable)
    {
      Assert(precomputedTable != NULL);
      for (i = 0; i < partition_num; i++)
      {
        //result += table[i * pq_centroid_num + encoded_vector[i]];
        result += table[encoded_vector[i]];
        table += pq_centroid_num;
      }
    }
    else
    {
      generated_vector = (float4 *)palloc0(sizeof(float4) * dim);
      for (i = 0; i < partition_num; i++) {
          Assert(encoded_vector[i] < pq_centroid_num);
          memcpy((void *)(generated_vector + i * subdim), 
          (void *)(((PqSubvectorTuple*)((char*)pqtups+(i * pq_centroid_num + encoded_vector[i])*(state->size_of_subvector_tuple)))->vector),
          subdim * sizeof(float4));
      }

      result = fvec_L2sqr(residual, generated_vector, dim); 

      pfree(generated_vector);
    }

    return result;
    
}

static void
ScanInvertedListAndCalDistance(Relation index, IvfpqMetaPageData *meta,
    IvfpqState *state, BlockNumber headBlkno,
    float4 *queryVec, PqCentroidTuple *ctup, PqSubvectorTuple *pqtuples, float4 *precomputedTable, float4 *residual,
    pairingheap *queue, pthread_mutex_t *mutex) {
  BlockNumber            blkno;
  Buffer                 buffer;
  Page                   page;
  IvfpqPageOpaque      opaque;
  PqInvertedListTuple      *itup;
  int                    i;
  float                  dis;
  PqInvertedListSearchItem *item;
  //float4                 *residual;
  int                    dim;

  blkno = headBlkno;

  dim = meta->opts.dimension;
  //residual = (float4 *)palloc0(sizeof(float4) * dim);

  for (i = 0; i < dim; i++) 
    residual[i] = queryVec[i] - ctup->vector[i];

  for (;;) {
    // to the end of inverted list
    if (blkno == 0) {
      break;
    }

    buffer = ReadBuffer(index, blkno);
    LockBuffer(buffer, BUFFER_LOCK_SHARE);
    page = BufferGetPage(buffer);
    opaque = IvfpqPageGetOpaque(page);

    for (i = 0; i < opaque->maxoff; ++i) {
      itup = PqInvertedListPageGetTuple(state, page, i + 1); 
      //dis = fvec_L2sqr(queryVec, itup->vector, meta->opts.dimension); 
      dis = CalDistanceForEncoded_L2sqr(state, meta, residual, 
      itup->encoded_vector, pqtuples, precomputedTable);
      if (mutex) {
        pthread_mutex_lock(mutex);
      }
      item = (PqInvertedListSearchItem *) palloc0(
          sizeof(PqInvertedListSearchItem));
      item->heap_ptr = itup->heap_ptr;
      item->distance = dis;
      pairingheap_add(queue, &item->ph_node);
      if (mutex) {
        pthread_mutex_unlock(mutex);
      }
    }
    UnlockReleaseBuffer(buffer); 
    blkno = opaque->next;
  }

  //pfree(residual);
}

static void
computePrecomputeTable(IvfpqMetaPageData *meta, IvfpqState *state, float4 *queryVec, PqCentroidTuple *ctup, 
                        PqSubvectorTuple *pqtuples,float4 *precomputedTable)
{
  
  int dim;
  int partition_num;
  int pq_centroid_num;
  int i,j;
  int subdim;
  float4 * residual;

  dim = meta->opts.dimension;
  partition_num = meta->opts.partition_num;
  pq_centroid_num = meta->opts.pq_centroid_num;
  subdim = dim / partition_num;
  residual = (float4 *)palloc0(sizeof(float4) * dim);

  for (i = 0; i < dim; i++) 
    residual[i] = queryVec[i] - ctup->vector[i];
#pragma omp for
  for (i = 0; i < partition_num; i++)
  {
    for (j = 0; j < pq_centroid_num; j++)
    {
      precomputedTable[i * pq_centroid_num + j] = fvec_L2sqr(
        ((PqSubvectorTuple*)(((char*)pqtuples) + (i*pq_centroid_num+j)*(state->size_of_subvector_tuple)))->vector,
        residual + i * subdim,
        subdim
      );
    }
  }
  pfree(residual);
}


// ivfpq_gettuple() -- Get the next tuple in the scan
bool
ivfpq_gettuple(IndexScanDesc scan, ScanDirection dir) {
  IvfpqScanOpaque      so;
  MemoryContext          oldCtx;
  bool                   reverse;
  IvfpqMetaPageData    *meta;
  Buffer		         metaBuffer;
  uint32                 scanRatio;
  uint32                 scanCentroidNum;
  PqInvertedListSearchItem *item;
  PqCentroidSearchItem     *citems;
  PqSubvectorTuple         *pqtuples;
  int                    i;

  if (dir != ForwardScanDirection) {
    elog(WARNING, "ivfpq only supports forward scan direction");
    return false;
  }

  reverse = false;
  so = (IvfpqScanOpaque) scan->opaque;
  if (!scan->orderByData) {
    elog(WARNING, "orderByData is invalid");
    return false;
  }
  if (!scan->orderByData->sk_argument) {
    elog(WARNING, "orderBy value is invalid");
    return false;
  }
  oldCtx = MemoryContextSwitchTo(so->scan_ctx);

  if (so->first_call) {
    so->scan_pase = DatumGetPASE(scan->orderByData->sk_argument);

    // get info from meta
    metaBuffer = ReadBuffer(scan->indexRelation, IVFPQ_METAPAGE_BLKNO);
    LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
    meta = IvfpqPageGetMeta(BufferGetPage(metaBuffer));

    if (meta->centroid_num == 0) {
      elog(WARNING, "centroid count is 0");
      MemoryContextSwitchTo(oldCtx);
      UnlockReleaseBuffer(metaBuffer);
      return false;
    }

    if (PASE_DIM(so->scan_pase) != meta->opts.dimension) {
      elog(WARNING, "query dimension(%u) not equal to data dimension(%u)",
          PASE_DIM(so->scan_pase), meta->opts.dimension);
      MemoryContextSwitchTo(oldCtx);
      UnlockReleaseBuffer(metaBuffer);
      return false;
    }

    scanRatio = PASE_EXTRA(so->scan_pase);
    if (scanRatio > MAX_SCAN_RATION) {
      elog(WARNING, "scanRatio[%u] is illegal, should in (0, 1000]", scanRatio);
      MemoryContextSwitchTo(oldCtx);
      UnlockReleaseBuffer(metaBuffer);
      return false;
    }
    if (scanRatio == 0) {
      scanRatio = DEFAULT_SCAN_RATIO;
    }
    scanCentroidNum = (scanRatio * meta->centroid_num) / MAX_SCAN_RATION;
    // scan one inverted list at least
    if (scanCentroidNum == 0) {
      scanCentroidNum = 1;
    }
    citems = (PqCentroidSearchItem*) palloc0(sizeof(PqCentroidSearchItem) * scanCentroidNum);

    scan->xs_recheck = false;
    scan->xs_recheckorderby = false;
    PqSearchKNNInvertedListFromCentroidPages(scan->indexRelation,
        &so->state, meta, so->scan_pase->x, scanCentroidNum,
        reverse, citems, true); //citems(selected centriod)
    pqtuples = PqGetSubvectorTuples(scan->indexRelation, &so->state, meta);
    if (meta->opts.open_omp) {
      pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
      omp_set_num_threads(meta->opts.omp_thread_num);
#pragma omp for 
      for (i = 0; i < scanCentroidNum; ++i) { 
        if (citems[i].cblkno == 0) {
          continue;
        }
        // inverted list is empty
        if (citems[i].head_ivl_blkno == 0) {
          continue;
        }
        float4 * precomputedTable = NULL;
        if (meta->opts.use_precomputedtable)
        {
          precomputedTable=(float4*)palloc0(sizeof(float4) * meta->opts.partition_num * meta->opts.pq_centroid_num);
          computePrecomputeTable(meta, &so->state, so->scan_pase->x, citems[i].ctup, pqtuples, precomputedTable);
        }
        float4 * residual = (float4 *)palloc0(sizeof(float4) * meta->opts.dimension);
        ScanInvertedListAndCalDistance(scan->indexRelation, meta,
            &so->state, citems[i].head_ivl_blkno,
            so->scan_pase->x, citems[i].ctup, pqtuples, precomputedTable, residual, so->queue, &mutex);
        pfree(citems[i].ctup);
        if (meta->opts.use_precomputedtable)
          pfree(precomputedTable);
        pfree(residual);
      }
      pthread_mutex_destroy(&mutex);
    } else {
      for (i = 0; i < scanCentroidNum; ++i) {
        if (citems[i].cblkno == 0) {
          continue;
        }
        // inverted list is empty
        if (citems[i].head_ivl_blkno == 0) {
          continue;
        }
        float4 * precomputedTable = NULL;
        if (meta->opts.use_precomputedtable)
        {
          precomputedTable=(float4*)palloc0(sizeof(float4) * meta->opts.partition_num * meta->opts.pq_centroid_num);
          computePrecomputeTable(meta, &so->state, so->scan_pase->x, citems[i].ctup, pqtuples, precomputedTable);
        }
        float4 * residual = (float4 *)palloc0(sizeof(float4) * meta->opts.dimension);
        ScanInvertedListAndCalDistance(scan->indexRelation, meta,
            &so->state, citems[i].head_ivl_blkno,
            so->scan_pase->x, citems[i].ctup, pqtuples, precomputedTable, residual, so->queue, (pthread_mutex_t *)NULL);
        pfree(citems[i].ctup);
        if (meta->opts.use_precomputedtable)
          pfree(precomputedTable);
        pfree(residual);
      }
    }
    if (!pairingheap_is_empty(so->queue)) {
      item = (PqInvertedListSearchItem*) pairingheap_remove_first(
          so->queue);
      scan->xs_ctup.t_self = item->heap_ptr;
      if (scan->numberOfOrderBys > 0) {
        scan->xs_orderbyvals[0] = Float4GetDatum(item->distance);
        scan->xs_orderbynulls[0] = false;
      }
      pfree(item);
    }
    so->first_call = false;
    pfree(citems);
    pfree(pqtuples);
    UnlockReleaseBuffer(metaBuffer);
  }
  else {
    if (!pairingheap_is_empty(so->queue)) {
      item = (PqInvertedListSearchItem*) pairingheap_remove_first(
          so->queue);
      scan->xs_ctup.t_self = item->heap_ptr;
      if (scan->numberOfOrderBys > 0) {
        scan->xs_orderbyvals[0] = Float4GetDatum(item->distance);
        scan->xs_orderbynulls[0] = false;
      }
      pfree(item);
    }
    else {
      elog(WARNING, "not enough data to pop for queue"); 
      MemoryContextSwitchTo(oldCtx);
      return false;
    }
  }
  MemoryContextSwitchTo(oldCtx);
  return true;
}

// ivfpq_gettuple() -- Get the next tuple in the scan
int64
ivfpq_getbitmap(IndexScanDesc scan, TIDBitmap *tbm) {
  elog(NOTICE, "ivfpq_getbitmap begin");
  return 0;
}
