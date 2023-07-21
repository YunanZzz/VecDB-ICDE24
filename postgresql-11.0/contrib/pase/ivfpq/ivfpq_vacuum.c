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
#include "catalog/storage.h"
#include "commands/vacuum.h"
#include "miscadmin.h"
#include "postmaster/autovacuum.h"
#include "storage/bufmgr.h"
#include "storage/indexfsm.h"
#include "storage/lmgr.h"

#include "ivfpq.h"


// Bulk deletion of all index entries pointing to a set of heap tuples.
// The set of target tuples is specified via a callback routine that tells
// whether any given heap tuple (identified by ItemPointer) is being deleted.
// 
// Result: a palloc'd struct containing statistical info for VACUUM displays.
IndexBulkDeleteResult *
ivfpq_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
    IndexBulkDeleteCallback callback, void *callback_state) {
  Relation	        index = info->index;
  BlockNumber         cblkno, iblkno;
  IvfpqState	    state;
  Buffer		        metaBuffer, cbuffer, ibuffer;
  Page		        cpage, ipage;
  IvfpqMetaPageData *meta;
  PqCentroidTuple       *ctup;
  PqInvertedListTuple   *itup;
  GenericXLogState    *gxlogState;
  IvfpqPageOpaque   iopaque;
  OffsetNumber ioffset, imaxOffset;

  elog(INFO, "ivfpq_bulkdelete begin");

  if (stats == NULL)
    stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));

  InitIvfpqState(&state, index);
  metaBuffer = ReadBuffer(index, IVFPQ_METAPAGE_BLKNO);
  LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
  meta = IvfpqPageGetMeta(BufferGetPage(metaBuffer));
  cblkno = meta->centroid_head_blkno;
  for (;cblkno < meta->centroid_head_blkno + meta->centroid_page_count; ++cblkno) {
    cbuffer = ReadBufferExtended(index, MAIN_FORKNUM, cblkno, RBM_NORMAL,
        info->strategy);
    LockBuffer(cbuffer, BUFFER_LOCK_SHARE);
    cpage = BufferGetPage(cbuffer); 
    if (!PageIsNew(cpage) && !IvfpqPageIsDeleted(cpage)) {
      OffsetNumber offset,
                   maxOffset = IvfpqPageGetMaxOffset(cpage);
      for (offset = 1; offset <= maxOffset; ++offset) {
        ctup = PqCentroidPageGetTuple(&state, cpage, offset);
        if (ctup->head_ivl_blkno == 0)
          continue;
        iblkno = ctup->head_ivl_blkno;
        for (;;) {
          // iterator to the inverted list end
          if (iblkno == 0)
            break;
          ibuffer = ReadBuffer(index, iblkno);
          LockBuffer(ibuffer, BUFFER_LOCK_EXCLUSIVE);
          gxlogState = GenericXLogStart(index);  
          ipage = GenericXLogRegisterBuffer(gxlogState, ibuffer, 0);
          if (PageIsNew(ipage) || IvfpqPageIsDeleted(ipage)) {
            UnlockReleaseBuffer(ibuffer);
            GenericXLogAbort(gxlogState);
            iopaque = IvfpqPageGetOpaque(ipage);
            iblkno = iopaque->next; 
            continue;
          }
          imaxOffset = IvfpqPageGetMaxOffset(ipage);
          for (ioffset = 1; ioffset <= imaxOffset; ++ioffset) {
            itup = PqInvertedListPageGetTuple(
                &state, ipage, ioffset);
            if (callback(&itup->heap_ptr, callback_state)) {
              itup->is_deleted = 1;
              stats->tuples_removed += 1; 
            }
          }
          iopaque = IvfpqPageGetOpaque(ipage);
          iblkno = iopaque->next; 
          UnlockReleaseBuffer(ibuffer);
          GenericXLogFinish(gxlogState);
        }
      }
    }
    UnlockReleaseBuffer(cbuffer);
  }
  UnlockReleaseBuffer(metaBuffer);
  return stats;
}

// Post-VACUUM cleanup.
//
// Result: a palloc'd struct containing statistical info for VACUUM displays.
IndexBulkDeleteResult *
ivfpq_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats) {
  // Iterate over the pages: insert deleted pages into FSM and collect
  // statistics.
  Relation	 index = info->index;
  BlockNumber  npages, cblkno, iblkno;
  IvfpqState state;
  Buffer		 metaBuffer, cbuffer, ibuffer;
  Page		 cpage, ipage;
  IvfpqMetaPageData *meta;
  PqCentroidTuple *ctup;
  IvfpqPageOpaque opaque;

  if (info->analyze_only)
    return stats;

  if (stats == NULL)
    stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));

  npages = RelationGetNumberOfBlocks(index);
  stats->num_pages = npages;
  stats->pages_free = 0;
  stats->num_index_tuples = 0;

  InitIvfpqState(&state, index);
  metaBuffer = ReadBuffer(index, IVFPQ_METAPAGE_BLKNO);
  LockBuffer(metaBuffer, BUFFER_LOCK_SHARE);
  meta = IvfpqPageGetMeta(BufferGetPage(metaBuffer));
  cblkno = meta->centroid_head_blkno;
  for (;cblkno < meta->centroid_head_blkno + meta->centroid_page_count;
      ++cblkno) {
    cbuffer = ReadBufferExtended(index, MAIN_FORKNUM, cblkno, RBM_NORMAL,
        info->strategy);
    LockBuffer(cbuffer, BUFFER_LOCK_SHARE);
    cpage = BufferGetPage(cbuffer); 
    if (!PageIsNew(cpage) && !IvfpqPageIsDeleted(cpage)) {
      OffsetNumber offset,
                   maxOffset = IvfpqPageGetMaxOffset(cpage);
      for (offset = 1; offset <= maxOffset; ++offset) {
        ctup = PqCentroidPageGetTuple(&state, cpage, offset);
        if (ctup->head_ivl_blkno == 0)
          continue;
        iblkno = ctup->head_ivl_blkno;
        for (;;) {
          // iterator to the inverted list end
          if (iblkno == 0)
            break;
          ibuffer = ReadBufferExtended(index, MAIN_FORKNUM, iblkno,
              RBM_NORMAL, info->strategy);
          LockBuffer(ibuffer, BUFFER_LOCK_SHARE);
          ipage = BufferGetPage(ibuffer);
          opaque = IvfpqPageGetOpaque(ipage);
          if (PageIsNew(ipage) || IvfpqPageIsDeleted(ipage))
          {
            RecordFreeIndexPage(index, iblkno);
            stats->pages_free++;
          }
          else
          {
            stats->num_index_tuples += IvfpqPageGetMaxOffset(ipage);
          }
          UnlockReleaseBuffer(ibuffer);
          iblkno = opaque->next;
        }
      }
    }
    UnlockReleaseBuffer(cbuffer);
  }
  UnlockReleaseBuffer(metaBuffer);
  IndexFreeSpaceMapVacuum(info->index);
  return stats;
}
