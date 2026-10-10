// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/* Read-only Linux direct AIO for exact BF16 rows in a safetensors checkpoint.
 *
 * ZFS completes O_DIRECT AIO in the submitting thread, so a deeper queue buys
 * nothing (measured on this box: 7.4k IOPS flat from iodepth 1 to 128) while the
 * number of submitting threads scales to 63.7k IOPS at 16. PLE_AIO_THREADS
 * therefore splits one rows_read() across independent AIO contexts.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/aio_abi.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MAX_THREADS 64u

struct reader;

struct job {
  struct reader* r;
  aio_context_t ctx;
  unsigned char* buf;
  struct iocb* cbs;
  struct iocb** ptrs;
  struct io_event* events;
  unsigned seq;
  const int* fds;
  const uint64_t* offsets;
  unsigned char* out;
  unsigned start, count, row_bytes, depth;
  int error;
};

struct reader {
  unsigned depth, threads, ntid;
  unsigned generation, done_count;
  int shutdown;
  pthread_mutex_t mu;
  pthread_cond_t work;
  pthread_cond_t finished;
  pthread_t tid[MAX_THREADS];
  struct job jobs[MAX_THREADS];
};

static void job_free(struct job* j) {
  if (j->ctx) syscall(__NR_io_destroy, j->ctx);
  j->ctx = 0;
  free(j->buf);   j->buf = NULL;
  free(j->cbs);   j->cbs = NULL;
  free(j->ptrs);  j->ptrs = NULL;
  free(j->events); j->events = NULL;
}

static int job_alloc(struct job* j, unsigned depth) {
  j->depth = depth;
  j->ctx = 0;
  if (posix_memalign((void**)&j->buf, 4096, (size_t)depth * 8192)) return -1;
  j->cbs = calloc(depth, sizeof(*j->cbs));
  j->ptrs = calloc(depth, sizeof(*j->ptrs));
  j->events = calloc(depth, sizeof(*j->events));
  if (!j->cbs || !j->ptrs || !j->events) return -1;
  if (syscall(__NR_io_setup, depth, &j->ctx) < 0) return -1;
  return 0;
}

static void job_run(struct job* j) {
  const uint64_t* offsets = j->offsets;
  const int* fds = j->fds;
  unsigned char* out = j->out;
  unsigned row_bytes = j->row_bytes, depth = j->depth;
  unsigned end = j->start + j->count;
  int error = 0;
  for (unsigned base = j->start; base < end && !error; base += depth) {
    unsigned batch = end - base < depth ? end - base : depth;
    unsigned submitted = 0, complete = 0;
    memset(j->cbs, 0, batch * sizeof(*j->cbs));
    for (unsigned i = 0; i < batch; ++i) {
      struct iocb* cb = &j->cbs[i];
      cb->aio_data = base + i;
      cb->aio_lio_opcode = IOCB_CMD_PREAD;
      cb->aio_fildes = fds[base + i];
      cb->aio_buf = (uint64_t)(uintptr_t)(j->buf + (size_t)i * 8192);
      cb->aio_offset = offsets[base + i] & ~(uint64_t)4095;
      cb->aio_nbytes = ((offsets[base + i] & 4095) + row_bytes + 4095) & ~4095;
      j->ptrs[i] = cb;
    }
    while (complete < batch) {
      if (submitted < batch) {
        long rc = syscall(__NR_io_submit, j->ctx, batch - submitted, j->ptrs + submitted);
        if (rc > 0)
          submitted += (unsigned)rc;
        else if (rc < 0 && errno == EINTR)
          continue;
        else if (rc <= 0 &&
                 (rc == 0 || errno != EAGAIN || submitted == complete)) {
          error = rc < 0 ? -errno : -EIO;
          goto failed;
        }
      }
      long n = syscall(__NR_io_getevents, j->ctx, 1, submitted - complete, j->events, NULL);
      if (n < 0) {
        if (errno == EINTR) continue;
        error = -errno;
        goto failed;
      }
      for (long i = 0; i < n; ++i) {
        struct io_event* e = &j->events[i];
        unsigned x = (unsigned)e->data;
        size_t delta = (size_t)(offsets[x] & 4095);
        if (e->res < (int64_t)(delta + row_bytes))
          error = e->res < 0 ? (int)e->res : -EIO;
        else
          memcpy(out + (size_t)x * row_bytes,
                 j->buf + (size_t)(x - base) * 8192 + delta, row_bytes);
      }
      complete += (unsigned)n;
    }
  }
  j->error = error;
  return;
failed:
  /* io_destroy waits for outstanding requests before buffers can be reused. */
  syscall(__NR_io_destroy, j->ctx);
  j->ctx = 0;
  j->error = error;
}

static void* worker_main(void* arg) {
  struct job* j = (struct job*)arg;
  struct reader* r = j->r;
  for (;;) {
    pthread_mutex_lock(&r->mu);
    while (!r->shutdown && j->seq == r->generation) pthread_cond_wait(&r->work, &r->mu);
    if (r->shutdown) {
      pthread_mutex_unlock(&r->mu);
      break;
    }
    j->seq = r->generation;
    pthread_mutex_unlock(&r->mu);
    job_run(j);
    pthread_mutex_lock(&r->mu);
    if (++r->done_count == r->ntid) pthread_cond_signal(&r->finished);
    pthread_mutex_unlock(&r->mu);
  }
  return NULL;
}

void rows_close(struct reader* r) {
  if (!r) return;
  pthread_mutex_lock(&r->mu);
  r->shutdown = 1;
  pthread_cond_broadcast(&r->work);
  pthread_mutex_unlock(&r->mu);
  for (unsigned i = 0; i < r->ntid; ++i) pthread_join(r->tid[i], NULL);
  for (unsigned i = 0; i < MAX_THREADS; ++i) job_free(&r->jobs[i]);
  pthread_mutex_destroy(&r->mu);
  pthread_cond_destroy(&r->work);
  pthread_cond_destroy(&r->finished);
  free(r);
}

struct reader* rows_open(unsigned depth) {
  if (!depth || depth > 4096) {
    errno = EINVAL;
    return NULL;
  }
  unsigned threads = 1;
  const char* env = getenv("PLE_AIO_THREADS");
  if (env && *env) {
    char* end = NULL;
    long v;
    errno = 0;
    v = strtol(env, &end, 10);
    if (!errno && end != env && v >= 1 && v <= (long)MAX_THREADS) threads = (unsigned)v;
  }
  struct reader* r = calloc(1, sizeof(*r));
  if (!r) {
    errno = ENOMEM;
    return NULL;
  }
  r->depth = depth;
  r->threads = threads;
  pthread_mutex_init(&r->mu, NULL);
  pthread_cond_init(&r->work, NULL);
  pthread_cond_init(&r->finished, NULL);
  for (unsigned i = 0; i < threads; ++i) {
    struct job* j = &r->jobs[i];
    j->r = r;
    if (job_alloc(j, depth)) break;
    if (pthread_create(&r->tid[i], NULL, worker_main, j) != 0) {
      job_free(j);
      break;
    }
    r->ntid = i + 1;
  }
  if (!r->ntid) {
    for (unsigned i = 0; i < MAX_THREADS; ++i) job_free(&r->jobs[i]);
    pthread_mutex_destroy(&r->mu);
    pthread_cond_destroy(&r->work);
    pthread_cond_destroy(&r->finished);
    free(r);
    errno = ENOMEM;
    return NULL;
  }
  r->threads = r->ntid;
  return r;
}

int rows_read(struct reader* r, const int* fds, const uint64_t* offsets,
              unsigned count, unsigned row_bytes, unsigned char* out) {
  if (!r || !r->ntid) return -EBADF;
  if (!row_bytes || row_bytes > 4096) return -EINVAL;
  if (!count) return 0;
  unsigned t = r->ntid;
  int error = 0;
  pthread_mutex_lock(&r->mu);
  for (unsigned i = 0; i < t; ++i) {
    struct job* j = &r->jobs[i];
    unsigned lo = (unsigned)((uint64_t)count * i / t);
    unsigned hi = (unsigned)((uint64_t)count * (i + 1) / t);
    j->fds = fds;
    j->offsets = offsets;
    j->out = out;
    j->start = lo;
    j->count = hi - lo;
    j->row_bytes = row_bytes;
    j->error = 0;
    if (!j->ctx) error = -EBADF;
  }
  if (error) {
    pthread_mutex_unlock(&r->mu);
    return error;
  }
  r->generation += 1;
  r->done_count = 0;
  pthread_cond_broadcast(&r->work);
  while (r->done_count != r->ntid) pthread_cond_wait(&r->finished, &r->mu);
  pthread_mutex_unlock(&r->mu);
  for (unsigned i = 0; i < t; ++i)
    if (r->jobs[i].error) error = r->jobs[i].error;
  return error;
}
