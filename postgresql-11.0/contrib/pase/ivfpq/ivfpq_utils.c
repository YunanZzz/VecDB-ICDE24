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

#include <stdio.h>
#include <float.h>
#include <omp.h>
#include "access/amapi.h"
#include "access/generic_xlog.h"
#include "catalog/index.h"
#include "storage/lmgr.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/indexfsm.h"
#include "utils/memutils.h"
#include "access/reloptions.h"
#include "storage/freespace.h"
#include "storage/indexfsm.h"
#include "lib/pairingheap.h"

#include "utils/string_util.h"
#include "utils/vector_util.h"
#include "ivfpq.h"

// Construct a default set of Bloom options.
static IvfpqOptions *
makeDefaultIvfpqOptions(void) {
  IvfpqOptions *opts;

  opts = (IvfpqOptions *) palloc0(sizeof(IvfpqOptions));
  opts->distance_type = 0;
  opts->dimension = 256;
  opts->partition_num = 16;
  opts->pq_centroid_num = 256;
  SET_VARSIZE(opts, sizeof(IvfpqOptions));
  return opts;
}

// Fill IvfpqState structure for particular index.
void
InitIvfpqState(IvfpqState *state, Relation index) {
  state->nColumns = index->rd_att->natts;

  // Initialize amcache if needed with options from metapage
  if (!index->rd_amcache)
  {
    Buffer		buffer;
    Page		page;
    IvfpqMetaPageData *meta;
    IvfpqOptions *opts;

    opts = MemoryContextAlloc(index->rd_indexcxt, sizeof(IvfpqOptions));

    buffer = ReadBuffer(index, IVFPQ_METAPAGE_BLKNO);
    LockBuffer(buffer, BUFFER_LOCK_SHARE);

    page = BufferGetPage(buffer);

    if (!IvfpqPageIsMeta(page))
      elog(ERROR, "Relation is not a pase ivfpq index");
    meta = IvfpqPageGetMeta(BufferGetPage(buffer));

    if (meta->magick_number != IVFPQ_MAGICK_NUMBER)
      elog(ERROR, "Relation is not a pase ivfpq index");

    *opts = meta->opts;
    UnlockReleaseBuffer(buffer);
    index->rd_amcache = (void *) opts;
  }

  memcpy(&state->opts, index->rd_amcache, sizeof(state->opts));
  state->size_of_centroid_tuple = PQCENTROIDTUPLEHDRSZ +
    sizeof(float4) * state->opts.dimension;
  state->size_of_subvector_tuple = PQSUBVECTORTUPLEHDRSZ + 
    sizeof(float4) * state->opts.dimension / state->opts.partition_num;
  state->size_of_invertedlist_tuple = PQINVERTEDLISTTUPLEHDRSZ +
    sizeof(uint8_t) * state->opts.partition_num;
  state->size_of_invertedlist_rawtuple = PQINVERTEDLISTRAWTUPLEHDRSZ + 
    sizeof(float4) * state->opts.dimension;
}

float
PqSearchNNFromCentroids(IvfpqState *state, PqInvertedListRawTuple *tuple,
    PqCentroids centroids, int *minPos) {
  // TODO(yangwen.yw): omp for
  float minDistance;
  PqCentroidTuple *ctup;
  int i;
  float dis;

  minDistance = FLT_MAX;
  *minPos = centroids->count;
  for (i = 0; i < centroids->count; ++i) {
    // TODO(yangwen.yw): support other metric type
    if (state->opts.distance_type == 0) {
      ctup = (PqCentroidTuple *)((char*)centroids->ctups + i * state->size_of_centroid_tuple);
      dis = fvec_L2sqr(tuple->vector, ctup->vector,
          centroids->dim);
      if (dis < minDistance) {
        minDistance = dis;
        *minPos = i;
      }
    }
  }
  return minDistance;
}

int
PqPairingHeapCentroidCompare(const pairingheap_node *a,
    const pairingheap_node *b, void *arg) {
  const PqCentroidSearchItem *ca = (const PqCentroidSearchItem *)a;
  const PqCentroidSearchItem *cb = (const PqCentroidSearchItem *)b;
  bool *reverse = (bool*) arg;
  if (ca->distance > cb->distance) {
    if (*reverse)
      return 1;
    else
      return -1;
  }
  else if (ca->distance < cb->distance) {
    if (*reverse)
      return -1;
    else
      return 1;
  }
  else
    return 0;
}

void
PqSearchKNNInvertedListFromCentroidPages(Relation index, IvfpqState *state,
    IvfpqMetaPageData *meta, float4 *tuple_vector,
    int count, bool reverse, PqCentroidSearchItem *items,
    bool isScan) {
  // TODO(yangwen.yw): omp for
  BlockNumber cblkno;
  Buffer cbuffer;
  Page cpage;
  PqCentroidTuple *ctup;
  BufferAccessStrategy bas;
  pairingheap *queue;
  PqCentroidSearchItem *item;
  int i;
  bas = GetAccessStrategy(BAS_BULKREAD);

  cblkno = meta->centroid_head_blkno;
  queue = pairingheap_allocate(PqPairingHeapCentroidCompare, &reverse);
  for (; cblkno < meta->centroid_head_blkno + meta->centroid_page_count; ++cblkno) {
    cbuffer = ReadBufferExtended(index, MAIN_FORKNUM, cblkno,
        RBM_NORMAL, bas);
    LockBuffer(cbuffer, BUFFER_LOCK_SHARE);
    cpage = BufferGetPage(cbuffer); 
    if (!PageIsNew(cpage) && !IvfpqPageIsDeleted(cpage)) {
      OffsetNumber offset,
                   maxOffset = IvfpqPageGetMaxOffset(cpage);
      for (offset = 1; offset <= maxOffset; ++offset) {
        ctup = PqCentroidPageGetTuple(state, cpage, offset);
        if (isScan && ctup->head_ivl_blkno == 0)
          continue;
        // TODO(yangwen.yw): support other metric type
        if (state->opts.distance_type == 0) {
          float dis = fvec_L2sqr(tuple_vector, ctup->vector,
              meta->opts.dimension);
          item = (PqCentroidSearchItem *) palloc0(
              sizeof(PqCentroidSearchItem));
          item->cblkno = cblkno;
          item->offset = offset;
          item->head_ivl_blkno = ctup->head_ivl_blkno;
          item->distance = dis;
          item->ctup = (PqCentroidTuple *)palloc0(state->size_of_centroid_tuple);
          memcpy((Pointer)item->ctup, (Pointer)ctup, state->size_of_centroid_tuple);
          pairingheap_add(queue, &item->ph_node);
        }
      }
    }
    UnlockReleaseBuffer(cbuffer);
  }

  for (i = 0; i < count; ++i) {
    if (!pairingheap_is_empty(queue)) {
      item = (PqCentroidSearchItem*) pairingheap_remove_first(queue);
      items[i] = *item;
      pfree(item);
    }
  }
  for(;;) {
    if (!pairingheap_is_empty(queue)) {
      item = (PqCentroidSearchItem*) pairingheap_remove_first(queue);
      pfree(item->ctup);
      pfree(item);
    }
    else
      break;
  }
  pairingheap_free(queue);
  FreeAccessStrategy(bas);
}

PqSubvectorTuple *
PqGetSubvectorTuples(Relation index, IvfpqState *state,
  IvfpqMetaPageData *meta) {
  BlockNumber pqblkno;
  Buffer pqbuffer;
  Page pqpage;
  BufferAccessStrategy bas;
  PqSubvectorTuple *pqtup;
  PqSubvectorTuple *pqtups;
  int i;
  bas = GetAccessStrategy(BAS_BULKREAD);

  pqtups = (PqSubvectorTuple *)palloc0(state->size_of_subvector_tuple * state->opts.partition_num * state->opts.pq_centroid_num);

  pqblkno = meta->pq_centroid_head_blkno;
  i = 0;
  for (; pqblkno < meta->pq_centroid_head_blkno + meta->pq_centroid_page_count; ++pqblkno) {
    pqbuffer = ReadBufferExtended(index, MAIN_FORKNUM, pqblkno,
        RBM_NORMAL, bas);
    LockBuffer(pqbuffer, BUFFER_LOCK_SHARE);
    pqpage = BufferGetPage(pqbuffer); 
    if (!PageIsNew(pqpage) && !IvfpqPageIsDeleted(pqpage)) {
      OffsetNumber offset,
                   maxOffset = IvfpqPageGetMaxOffset(pqpage);
      for (offset = 1; offset <= maxOffset; ++offset) {
        pqtup = PqSubvectorPageGetTuple(state, pqpage, offset);
        memcpy((Pointer)((char*)pqtups + (i++) * state->size_of_subvector_tuple),(Pointer)pqtup, state->size_of_subvector_tuple);
      }
    }
    UnlockReleaseBuffer(pqbuffer);
  }
  Assert(i == state->opts.partition_num * state->opts.pq_centroid_num);

  return pqtups;
}

void PqFlushBufferPage(Relation index, Buffer buffer, bool needUnLock) {
  GenericXLogState *state;

  if (!needUnLock)
    LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
  state = GenericXLogStart(index);
  GenericXLogRegisterBuffer(state, buffer, GENERIC_XLOG_FULL_IMAGE);
  GenericXLogFinish(state);
  UnlockReleaseBuffer(buffer);
}

// Allocate a new page (either by recycling, or by extending the index file)
// The returned buffer is already pinned and exclusive-locked when used in
// inserting, but not exclusive-locked in building
// Caller is responsible for initializing the page by calling IvfpqInitBuffer
Buffer
IvfpqNewBuffer(Relation index, bool needLock) {
  Buffer		buffer;
  bool		inNeedLock;

  /// First, try to get a page from FSM
  for (;;) {
    BlockNumber blkno = GetFreeIndexPage(index);

    if (blkno == InvalidBlockNumber)
      break;

    buffer = ReadBuffer(index, blkno);

    // We have to guard against the possibility that someone else already
    // recycled this page; the buffer may be locked if so.
    if (ConditionalLockBuffer(buffer)) {
      Page		page = BufferGetPage(buffer);

      if (PageIsNew(page))
        return buffer;	// OK to use, if never initialized

      if (IvfpqPageIsDeleted(page))
        return buffer;	// OK to use

      LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
    }

    // Can't use it, so release buffer and try again
    ReleaseBuffer(buffer);
  }

  // Must extend the file
  inNeedLock = !RELATION_IS_LOCAL(index);
  if (inNeedLock)
    LockRelationForExtension(index, ExclusiveLock);

  buffer = ReadBuffer(index, P_NEW);
  if (needLock)
    LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);

  if (inNeedLock)
    UnlockRelationForExtension(index, ExclusiveLock);

  return buffer;
}

// Initialize any page of a ivfpq index.
void
IvfpqInitPage(Page page, uint16 flags) {
  IvfpqPageOpaque opaque;
  PageInit(page, BLCKSZ, sizeof(IvfpqPageOpaqueData));
  opaque = IvfpqPageGetOpaque(page);
  memset(opaque, 0, sizeof(IvfpqPageOpaqueData));
  opaque->flags = flags;
}

// Fill in metapage for ivfpq index.
void
IvfpqFillMetapage(Relation index, Page metaPage) {
  IvfpqOptions *opts;
  IvfpqMetaPageData *metadata;

  // Choose the index's options.  If reloptions have been assigned, use
  // those, otherwise create default options.
  opts = (IvfpqOptions *) index->rd_options;
  if (!opts)
    opts = makeDefaultIvfpqOptions();

  // Initialize contents of meta page, including a copy of the options,
  // which are now frozen for the life of the index.
  IvfpqInitPage(metaPage, IVFPQ_META);
  metadata = IvfpqPageGetMeta(metaPage);
  memset(metadata, 0, sizeof(IvfpqMetaPageData));
  metadata->magick_number = IVFPQ_MAGICK_NUMBER;
  metadata->opts = *opts;
  ((PageHeader) metaPage)->pd_lower += sizeof(IvfpqMetaPageData);
  Assert(((PageHeader) metaPage)->pd_lower <= ((PageHeader) metaPage)->pd_upper);
}

// Initialize metapage for ivfpq index.
void
IvfpqInitMetapage(Relation index)
{
  Buffer      metaBuffer;
  Page        metaPage;
  GenericXLogState *state;

  // Make a new page; since it is first page it should be associated with
  // block number 0 (IVFPQ_METAPAGE_BLKNO).
  metaBuffer = IvfpqNewBuffer(index, true);
  Assert(BufferGetBlockNumber(metaBuffer) == IVFPQ_METAPAGE_BLKNO);

  // Initialize contents of meta page
  state = GenericXLogStart(index);
  metaPage = GenericXLogRegisterBuffer(state, metaBuffer,
      GENERIC_XLOG_FULL_IMAGE);
  IvfpqFillMetapage(index, metaPage);
  GenericXLogFinish(state);

  UnlockReleaseBuffer(metaBuffer);
}

// Parse reloptions for ivfpq index, producing a IvfpqOptions struct.
bytea *
ivfpq_options(Datum reloptions, bool validate) {
  relopt_value *options;
  int         numoptions;
  IvfpqOptions *rdopts;

  static const relopt_parse_elt ivfpq_relopt_tab[] = {
    {"clustering_type", RELOPT_TYPE_INT, offsetof(IvfpqOptions, clustering_type)},
    {"distance_type", RELOPT_TYPE_INT, offsetof(IvfpqOptions, distance_type)},
    {"dimension", RELOPT_TYPE_INT, offsetof(IvfpqOptions, dimension)},
    {"open_omp", RELOPT_TYPE_INT, offsetof(IvfpqOptions, open_omp)},
    {"omp_thread_num", RELOPT_TYPE_INT, offsetof(IvfpqOptions, omp_thread_num)},
    {"base64_encoded", RELOPT_TYPE_INT, offsetof(IvfpqOptions, base64_encoded)},
    {"partition_num", RELOPT_TYPE_INT, offsetof(IvfpqOptions, partition_num)},
    {"pq_centroid_num", RELOPT_TYPE_INT, offsetof(IvfpqOptions, pq_centroid_num)},
    {"use_precomputedtable", RELOPT_TYPE_INT, offsetof(IvfpqOptions, use_precomputedtable)},
    {"clustering_params", RELOPT_TYPE_STRING, offsetof(IvfpqOptions, clustering_params_offset)}
  };

  // Parse the user-given reloptions
  options = parseRelOptions(reloptions, validate, ivfpq_relopt_kind, &numoptions);
  if (numoptions < 4) {
    elog(ERROR, "options format error");
  }
  rdopts = allocateReloptStruct(sizeof(IvfpqOptions), options, numoptions);
  fillRelOptions((void *) rdopts, sizeof(IvfpqOptions), options, numoptions,
      validate, ivfpq_relopt_tab, lengthof(ivfpq_relopt_tab));

  if (rdopts->open_omp) {
    elog(NOTICE, "using openmp to speed up");
    if (rdopts->omp_thread_num == 0) {
      elog(NOTICE, "not set omp thread number, set core number[%d] to it",
          omp_get_num_procs());
      omp_set_num_threads(omp_get_num_procs());
    } else {
      elog(NOTICE, "set omp thread number[%d]", rdopts->omp_thread_num);
      omp_set_num_threads(rdopts->omp_thread_num);
    }
  }

  pfree(options);
  return (bytea *) rdopts;
}
