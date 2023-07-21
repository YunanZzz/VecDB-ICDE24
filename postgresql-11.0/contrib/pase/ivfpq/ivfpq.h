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
// Impaseentation of ivfpq index
//
#ifndef PASE_IVFPQ_IVFPQ_H_
#define PASE_IVFPQ_IVFPQ_H_

#include "access/amapi.h"
#include "access/generic_xlog.h"
#include "access/itup.h"
#include "access/xlog.h"
#include "access/reloptions.h"
#include "nodes/relation.h"
#include "catalog/index.h"
#include "lib/pairingheap.h"
#include "fmgr.h"
#include "type/pase_data.h"
#include "pase.h"

////////////////////////////////////////////////////////////////////////////////
// Opaque for centroid and inverted list pages
typedef struct IvfpqPageOpaqueData {
  OffsetNumber maxoff;		// number of index tuples on page
  uint16      flags;      // see bit definitions below
  BlockNumber next;       // refer to next centroid page block
} IvfpqPageOpaqueData;

typedef IvfpqPageOpaqueData *IvfpqPageOpaque;

// ivfpq index options
typedef struct IvfpqOptions {
  int32  vl_len_;                   // varlena header (do not touch directly!)
  int    clustering_type;           // clustering type:0 centroid_file, 1 inner clustering
  int	 distance_type;	            // distance metric type:0 l2, 1 inner proudct, 2 cosine
  int    dimension;                 // vector dimension

  int    partition_num;             //num of subvectors from a partitioned vector. Should be a factor of dimension.
  int    pq_centroid_num;          //num of refined centroids in each subspace, no more than 256

  int    open_omp;                  // whether open omp, 0:close, 1:open
  int    omp_thread_num;            // omp thread number
  int    use_precomputedtable;      // whether to use precomputedtable
  int    base64_encoded;            // data whether base64 encoded
  int	 clustering_params_offset;  // clustering parameters offset
} IvfpqOptions;

// Metadata of ivfpq index
typedef struct IvfpqMetaPageData
{
  uint32		  magick_number;
  uint32      centroid_num;
  BlockNumber centroid_head_blkno;
  BlockNumber centroid_page_count;
  BlockNumber pq_centroid_head_blkno;
  BlockNumber pq_centroid_page_count;
  IvfpqOptions opts;
} IvfpqMetaPageData;

typedef struct IvfpqState {
  IvfpqOptions opts;			// copy of options on index's metapage
  int32          nColumns;
  Size           size_of_centroid_tuple;
  Size           size_of_subvector_tuple;
  Size           size_of_invertedlist_tuple;
  Size           size_of_invertedlist_rawtuple;
} IvfpqState;

// Tuple for centroid
typedef struct PqCentroidTuple {
  BlockNumber     head_ivl_blkno;
  uint32          inverted_list_size;
  float4          vector[FLEXIBLE_ARRAY_MEMBER];
} PqCentroidTuple;

typedef struct PqSubvectorTuple {
  uint8           is_deleted;
  float4          vector[FLEXIBLE_ARRAY_MEMBER];
} PqSubvectorTuple;

// Tuple for inverted list
typedef struct PqInvertedListTuple {
  ItemPointerData heap_ptr;
  uint8           is_deleted;
  /*256 PG centroids at most*/
  uint8_t         encoded_vector[FLEXIBLE_ARRAY_MEMBER];
} PqInvertedListTuple;

// Tuple for inverted list
typedef struct PqInvertedListRawTuple {
  ItemPointerData heap_ptr;
  uint8           is_deleted;
  float4          vector[FLEXIBLE_ARRAY_MEMBER];
} PqInvertedListRawTuple;

// centroid data
typedef struct PqCentroidsData {
  int    dim;
  int    count;
  int    partition_num;             
  int    pq_centroid_num; 
  PqCentroidTuple *ctups;
  PqSubvectorTuple *pqtups;
} PqCentroidsData;

typedef PqCentroidsData *PqCentroids;

typedef struct PqCentroidSearchItem {
  pairingheap_node ph_node;
  BlockNumber cblkno;
  OffsetNumber offset;
  BlockNumber head_ivl_blkno;
  PqCentroidTuple *ctup;   //ivfflat does not need centroid vectors in searchitem, but ivfpq need.
  float distance;
} PqCentroidSearchItem;

typedef struct PqInvertedListSearchItem {
  pairingheap_node ph_node;
  ItemPointerData heap_ptr;
  float distance;
} PqInvertedListSearchItem;

// Opaque data structure for ivfpq index scan
typedef struct IvfpqScanOpaqueData {
  PASE *scan_pase;
  MemoryContext scan_ctx;
  pairingheap *queue;
  IvfpqState state; 
  bool first_call;
} IvfpqScanOpaqueData;

typedef IvfpqScanOpaqueData *IvfpqScanOpaque;

////////////////////////////////////////////////////////////////////////////////
// ivfpq page flags
#define IVFPQ_META		(1<<0)
#define IVFPQ_DELETED	    (2<<0)

// build initializing memory size
#define IVFPQ_BUILD_INIT_MEM_SIZE 500 * 1024 * 1024
#define MAX_CLUSTERING_MEM          300 * 1024 * 1024
#define MAX_CLUSTERING_SAMPLE_COUNT 1000000
#define DEFAULT_SCAN_RATIO          20
#define MAX_SCAN_RATION             1000
#define MAX_CLUSTERING_SAMPLE_RATIO 1000

// Macros for accessing ivfpq page structures
#define IvfpqPageGetOpaque(_page) ((IvfpqPageOpaque) PageGetSpecialPointer(_page))
#define IvfpqPageGetMaxOffset(_page) (IvfpqPageGetOpaque(_page)->maxoff)
#define IvfpqPageIsMeta(_page) \
  ((IvfpqPageGetOpaque(_page)->flags & IVFPQ_META) != 0)
#define IvfpqPageIsDeleted(_page) \
  ((IvfpqPageGetOpaque(_page)->flags & IVFPQ_DELETED) != 0)
#define IvfpqPageSetDeleted(_page) \
  (IvfpqPageGetOpaque(_page)->flags |= IVFPQ_DELETED)
#define IvfpqPageSetNonDeleted(_page) \
  (IvfpqPageGetOpaque(_page)->flags &= ~IVFPQ_DELETED)
#define PqCentroidPageGetData(_page)		((PqCentroidTuple *)PageGetContents(_page))
#define PqCentroidPageGetTuple(_state, _page, _offset) \
  ((PqCentroidTuple *)(PageGetContents(_page) \
    + (_state)->size_of_centroid_tuple * ((_offset) - 1)))
#define PqSubvectorPageGetTuple(_state, _page, _offset) \
  ((PqSubvectorTuple *)(PageGetContents(_page) \
    + (_state)->size_of_subvector_tuple * ((_offset) - 1)))
#define PqCentoridPageGetNextTuple(_state, _tuple) \
  ((PqCentroidTuple *)((Pointer)(_tuple) + (_state)->size_of_centroid_tuple))
#define PqCentroidTuplesGetTuple(_buildState, _offset) \
  ((PqCentroidTuple *)((char*)_buildState->centroids.ctups + \
    _offset * (_buildState->ivf_state.size_of_centroid_tuple)))
#define PqSubvectorTuplesGetTuple(_buildState, _offset) \
  ((PqSubvectorTuple *)((char*)_buildState->centroids.pqtups + \
    _offset * (_buildState->ivf_state.size_of_subvector_tuple)))
#define PqInvertedListPageGetData(_page)	((PqInvertedListTuple *)PageGetContents(_page))
#define PqInvertedListPageGetTuple(_state, _page, _offset) \
  ((PqInvertedListTuple *)(PageGetContents(_page) \
    + (_state)->size_of_invertedlist_tuple * ((_offset) - 1)))
#define PqInvertedListPageGetNextTuple(_state, _tuple) \
  ((PqInvertedListTuple *)((Pointer)(_tuple) + (_state)->size_of_invertedlist_tuple))
#define IvfpqPageGetMeta(_page) ((IvfpqMetaPageData *) PageGetContents(_page))

// Preserved page numbers
#define IVFPQ_METAPAGE_BLKNO	(0)
#define IVFPQ_HEAD_BLKNO		(1) // first data page

// Default and maximum Ivfpq centroid file path length.
#define MAX_CENTROID_PATH_LEN   256
#define DEFAULT_DIMENSION       256

// Magic number to distinguish ivfpq pages among anothers
#define IVFPQ_MAGICK_NUMBER (0xDBAC0DEE)

#define PqCentroidPageGetFreeSpace(_state, _page) \
  (BLCKSZ - MAXALIGN(SizeOfPageHeaderData) \
   - IvfpqPageGetMaxOffset(_page) * (_state)->size_of_centroid_tuple \
   - MAXALIGN(sizeof(IvfpqPageOpaqueData)))
#define PqSubvectorPageGetFreeSpace(_state, _page) \
  (BLCKSZ - MAXALIGN(SizeOfPageHeaderData) \
   - IvfpqPageGetMaxOffset(_page) * (_state)->size_of_subvector_tuple \
   - MAXALIGN(sizeof(IvfpqPageOpaqueData)))
#define PqInvertedListPageGetFreeSpace(_state, _page) \
  (BLCKSZ - MAXALIGN(SizeOfPageHeaderData) \
   - IvfpqPageGetMaxOffset(_page) * (_state)->size_of_invertedlist_tuple \
   - MAXALIGN(sizeof(IvfpqPageOpaqueData)))

#define PQCENTROIDTUPLEHDRSZ offsetof(PqCentroidTuple, vector)
#define PQSUBVECTORTUPLEHDRSZ offsetof(PqSubvectorTuple, vector)
#define PQINVERTEDLISTTUPLEHDRSZ offsetof(PqInvertedListTuple, encoded_vector)
#define PQINVERTEDLISTRAWTUPLEHDRSZ offsetof(PqInvertedListRawTuple, vector)

////////////////////////////////////////////////////////////////////////////////
// ivfpq_utils.c
//extern void _PG_init(void);
extern void InitIvfpqState(IvfpqState *state, Relation index);
extern void IvfpqFillMetapage(Relation index, Page metaPage);
extern void IvfpqInitMetapage(Relation index);
extern void IvfpqInitPage(Page page, uint16 flags);
extern Buffer IvfpqNewBuffer(Relation index, bool needLock);
extern bool IvfpqPageAddItem(IvfpqState *state, Page page,
    PqInvertedListTuple *tuple);
extern void PqFlushBufferPage(Relation index, Buffer buffer, bool needUnLock);
extern bytea *ivfpq_options(Datum reloptions, bool validate);
extern float PqSearchNNFromCentroids(IvfpqState *state, PqInvertedListRawTuple *tuple,
    PqCentroids centroids, int *minPos);
extern int PqPairingHeapCentroidCompare(const pairingheap_node *a,
    const pairingheap_node *b, void *arg);
extern void PqSearchKNNInvertedListFromCentroidPages(
    Relation index, IvfpqState *state,
    IvfpqMetaPageData *meta, float4 *tuple_vector,
    int count, bool reverse, PqCentroidSearchItem *items, bool isScan);
extern PqSubvectorTuple *PqGetSubvectorTuples(Relation index, IvfpqState *state,
  IvfpqMetaPageData *meta);

// ivfpq_build.c
extern IndexBuildResult *ivfpq_build(Relation heap, Relation index, IndexInfo *indexInfo);
extern bool ivfpq_insert(Relation index, Datum *values, bool *isnull,
    ItemPointer ht_ctid, Relation heapRel,
    IndexUniqueCheck checkUnique, IndexInfo *indexInfo);
extern void ivfpq_buildempty(Relation index);

// ivfpq_vacuum.c
extern IndexBulkDeleteResult *ivfpq_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
    IndexBulkDeleteCallback callback, void *callback_state);
extern IndexBulkDeleteResult *ivfpq_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);

// ivfpq_scan.c
extern IndexScanDesc ivfpq_beginscan(Relation r, int nkeys, int norderbys);
extern void ivfpq_rescan(IndexScanDesc scan, ScanKey scankey, int nscankeys,
    ScanKey orderbys, int norderbys);
extern void ivfpq_endscan(IndexScanDesc scan);
extern bool ivfpq_gettuple(IndexScanDesc scan, ScanDirection dir);
extern int64 ivfpq_getbitmap(IndexScanDesc scan, TIDBitmap *tbm);

// ivfpq_cost.c
extern void
ivfpq_costestimate(PlannerInfo *root, IndexPath *path, double loop_count,
    Cost *indexStartupCost, Cost *indexTotalCost,
    Selectivity *indexSelectivity, double *indexCorrelation,
    double *indexPages);

// pase_handler.c
extern relopt_kind ivfpq_relopt_kind;

#endif  // PASE_IVFPQ_IVFPQ_H_
