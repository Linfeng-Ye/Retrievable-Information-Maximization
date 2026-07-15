# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import warnings
import numpy as np
from pykdtree.kdtree import KDTree

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def _cupy_runtime_available():
    if not CUPY_AVAILABLE:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


CUPY_RUNTIME_AVAILABLE = _cupy_runtime_available()


def cupy_is_usable(gpu=0):
    if not CUPY_RUNTIME_AVAILABLE:
        return False
    try:
        if hasattr(gpu, "index"):
            gpu = 0 if gpu.index is None else gpu.index
        elif isinstance(gpu, str):
            gpu = int(gpu.split(":")[-1]) if ":" in gpu else 0
        with cp.cuda.Device(int(gpu)):
            cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


def _as_numpy(x):
    if x is None:
        return None
    if CUPY_AVAILABLE and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and callable(x.cpu):
        return x.cpu().numpy()
    return np.asarray(x)


class KMeans():
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def init_centers(self, X, n_clusters, sample_weight=None):
        replace = X.shape[0] < n_clusters
        if sample_weight is not None:
            replace = replace or np.count_nonzero(sample_weight) < n_clusters
            idx = np.random.choice(X.shape[0], size=n_clusters, replace=replace, p=sample_weight)
        else:
            idx = np.random.choice(X.shape[0], size=n_clusters, replace=replace)
        return X[idx]

    def init_centers_cp(self, X, n_clusters, sample_weight=None):
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed.")
        if sample_weight is not None:
            idx = cp.random.choice(X.shape[0], size=n_clusters, replace=True, p=sample_weight)
        else:
            idx = cp.random.choice(X.shape[0], size=n_clusters, replace=False)
        return X[idx]

    def compute_centers_loop_np(self, X, labels, sample_weight=None):
        X = X.astype(np.float64)
        centers = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
        if sample_weight is not None:
            sample_weight = sample_weight.astype(np.float64)
            X = X * sample_weight[:, None]
        for k in range(self.n_clusters):
            idx = labels == k
            if idx.sum() == 0:
                centers[k, :] = 0
            else:
                if sample_weight is None:
                    count = idx.sum()
                else:
                    count = sample_weight[idx].sum()
                centers[k, :] = X[idx, :].sum(axis=0) / count
        return centers

    def compute_centers_np(self, X, labels, sample_weight=None, return_count=False):
        labels = labels.astype(np.int64, copy=False)
        X = X.astype(np.float64, copy=False)
        if sample_weight is not None:
            sample_weight = sample_weight.astype(np.float64, copy=False)
            weighted = X * sample_weight[:, None]
            count = np.bincount(labels, weights=sample_weight, minlength=self.n_clusters).astype(np.float64)
        else:
            weighted = X
            count = np.bincount(labels, minlength=self.n_clusters).astype(np.float64)
        centers = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
        np.add.at(centers, labels, weighted)
        nonempty = count > 0
        centers[nonempty] /= count[nonempty, None]
        if return_count:
            return centers, count
        return centers

    def compute_centers_cupy(self, X, labels, sample_weight=None):
        '''
        X: [p d], cupy float
        labels: [p], cupy int
        sample_weight: [p], cupy float
        '''
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed.")
        ix = cp.argsort(labels)
        labels = labels[ix]
        X = X[ix]
        if sample_weight is not None:
            sample_weight = sample_weight[ix]
            X = X * sample_weight[:, None]
        
        d = cp.diff(labels, prepend=0)
        pos = cp.flatnonzero(d)
        pos = cp.asarray(np.repeat(pos.get(), d[pos].get()))
        pos = cp.append(cp.concatenate((cp.zeros_like(pos[0:1]), pos)), len(X))

        X = cp.concatenate((cp.zeros_like(X[0:1]), X), axis=0)
        X = cp.cumsum(X, axis=0)
        if sample_weight is not None:
            sample_weight = cp.concatenate((cp.zeros_like(sample_weight[0:1]), sample_weight), axis=0)
            sample_weight = cp.cumsum(sample_weight, axis=0)

        X = cp.diff(X[pos], axis=0)
        if sample_weight is None:
            count = cp.diff(pos)
        else:
            count = cp.diff(sample_weight[pos], axis=0)
        centers = X / count[:, None]

        return centers, count

    def fit(self, X, sample_weight=None, backend=0, gpu=0):
        if backend==0:  # numpy
            self.fit_np(_as_numpy(X), _as_numpy(sample_weight))
        elif backend==1:  # cupy
            if not cupy_is_usable(gpu):
                warnings.warn("CuPy is unavailable or CUDA is not usable; falling back to NumPy KMeans.", RuntimeWarning)
                self.fit_np(_as_numpy(X), _as_numpy(sample_weight))
                return
            try:
                with cp.cuda.Device(gpu):
                    self.fit_cupy(X, sample_weight)
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as exc:
                warnings.warn(f"CuPy KMeans failed ({exc}); falling back to NumPy KMeans.", RuntimeWarning)
                self.fit_np(_as_numpy(X), _as_numpy(sample_weight))
        else:
            raise NotImplementedError

    def fit_np(self, X, sample_weight=None, **kwargs):
        X = _as_numpy(X).astype(np.float32, copy=False)
        if sample_weight is not None:
            sample_weight = _as_numpy(sample_weight).astype(np.float64, copy=False)
            sw_sum = sample_weight.sum()
            sample_weight_normalized = sample_weight / sw_sum if sw_sum > 0 else None
        else:
            sample_weight_normalized = None
        self.centers = self.init_centers(X, self.n_clusters, sample_weight_normalized)
        for i in range(self.max_iter):
            centers_old = self.centers
            self.kdtree = KDTree(self.centers)
            _, self.labels = query_chunked(self.kdtree, X, k=1, sqr_dists=True, chunk_size=int(2e8), return_dist=False)
            self.centers, count = self.compute_centers_np(X, self.labels, sample_weight, return_count=True)
            self.centers[count == 0] = centers_old[count == 0]
            if np.all(centers_old == self.centers):
                break
        self.kdtree = KDTree(self.centers)
        _, self.labels = query_chunked(self.kdtree, X, k=1, sqr_dists=True, chunk_size=int(2e8), return_dist=False)

    def fit_cupy(self, X, sample_weight=None, **kwargs):
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed.")
        X_cp = cp.asarray(X, dtype=cp.float64)
        X = X.cpu().numpy().astype(np.float32, copy=False)
        if sample_weight is not None:
            sample_weight_cp = cp.asarray(sample_weight, dtype=cp.float64)
            sample_weight = sample_weight.cpu().numpy().astype(np.float32, copy=False)
            sample_weight_normalized = sample_weight / sample_weight.sum()
        else:
            sample_weight_cp = None
            sample_weight_normalized = None
        centers_cp = self.init_centers_cp(
            X_cp,
            self.n_clusters,
            cp.asarray(sample_weight_normalized) if sample_weight_normalized is not None else None,
        )
        self.centers = cp.asnumpy(centers_cp).astype(np.float32)
        for i in range(self.max_iter):
            centers_old_cp = centers_cp
            self.kdtree = KDTree(self.centers)
            _, self.labels = query_chunked(self.kdtree, X, k=1, sqr_dists=True, chunk_size=int(2e8), return_dist=False)
            centers_cp, count = reduce_within_clusters_chunked(X_cp, self.n_clusters, self.labels, 
                sample_weight_cp, chunk_size=int(3e6 / X_cp.shape[-1]))
            centers_cp[count==0] = centers_old_cp[count==0]
            self.centers = cp.asnumpy(centers_cp).astype(np.float32)
            # if cp.all(centers_old_cp == centers_cp):
            #     break
        self.kdtree = KDTree(self.centers)
        _, self.labels = query_chunked(self.kdtree, X, k=1, sqr_dists=True, chunk_size=int(2e8), return_dist=False)

    def predict(self, X, sample_weight=None):
        _, labels = query_chunked(self.kdtree, X.astype(np.float32, copy=False), k=1, sqr_dists=True, 
            chunk_size=int(2e8), return_dist=False)

        return labels


def query_chunked(kd_tree, x, k, sqr_dists, chunk_size=int(2e8), return_dist=False):
    if chunk_size is None: chunk_size = x.shape[0]
    if chunk_size >= x.shape[0]: return kd_tree.query(x, k=k, sqr_dists=sqr_dists)

    dist = np.zeros([x.shape[0], k], dtype=np.float32) if return_dist else None
    idx = np.zeros([x.shape[0], k], dtype=np.uint32)
    if k == 1:
        if return_dist: dist = dist[:, 0]
        idx = idx[:, 0]
    for i in range(0, x.shape[0], chunk_size):
        dist_i, idx[i:i+chunk_size] = kd_tree.query(x[i:i+chunk_size], k=k, sqr_dists=sqr_dists)
        if return_dist: dist[i:i+chunk_size] = dist_i
    return dist, idx


def reduce_within_clusters(X, n_clusters, labels, sample_weight=None, reduce_weight=True):
    '''
    X: [p ...], cupy float
    labels: [p], cupy int
    sample_weight: [p], cupy float
    '''
    if not CUPY_RUNTIME_AVAILABLE or type(X) is not cp.ndarray:
        return reduce_within_clusters_np(X, n_clusters, labels, sample_weight, reduce_weight=reduce_weight)
    if type(labels) is not cp.ndarray:
        labels = cp.asarray(labels)
    if sample_weight is not None and type(sample_weight) is not cp.ndarray:
        sample_weight = cp.asarray(sample_weight)

    X = X.astype(dtype=cp.float64, copy=False)
    if sample_weight is not None:
        sample_weight = sample_weight.astype(dtype=cp.float64, copy=False)

    ix = cp.argsort(labels)
    labels = labels[ix]
    X = X[ix]
    if sample_weight is not None:
        sample_weight = sample_weight[ix]
        X = X * sample_weight.reshape([X.shape[0], *([1]*(X.ndim-1))])
    
    d = cp.diff(labels, prepend=0)
    pos = cp.flatnonzero(d)
    pos = cp.asarray(np.repeat(pos.get(), d[pos].get()))
    pos = cp.append(cp.concatenate((cp.zeros_like(pos[0:1]), pos)), len(X))

    X = cp.concatenate((cp.zeros_like(X[0:1]), X), axis=0)
    X = cp.cumsum(X, axis=0)
    if sample_weight is not None:
        sample_weight = cp.concatenate((cp.zeros_like(sample_weight[0:1]), sample_weight), axis=0)
        sample_weight = cp.cumsum(sample_weight, axis=0)

    X = cp.diff(X[pos], axis=0)
    if sample_weight is None:
        count = cp.diff(pos)
    else:
        count = cp.diff(sample_weight[pos], axis=0)
    if reduce_weight:
        out = X / count.reshape([X.shape[0], *([1]*(X.ndim-1))])
    else:
        out = X

    if out.shape[0] < n_clusters:
        n_fill = n_clusters - out.shape[0]
        out = cp.concatenate([out, cp.zeros([n_fill, *out.shape[1:]])])
        count = cp.concatenate([count, cp.zeros([n_fill])])

    return out, count


def reduce_within_clusters_chunked(X, n_clusters, labels, sample_weight=None, chunk_size=None):
    if not CUPY_RUNTIME_AVAILABLE or type(X) is not cp.ndarray:
        return reduce_within_clusters_chunked_np(X, n_clusters, labels, sample_weight=sample_weight, chunk_size=chunk_size)
    if chunk_size is None: chunk_size = X.shape[0]
    for i in range(0, X.shape[0], chunk_size):
        out_i, count_i = reduce_within_clusters(X[i:i+chunk_size], n_clusters, labels[i:i+chunk_size], 
            sample_weight[i:i+chunk_size] if sample_weight is not None else None, reduce_weight=False)
        if i == 0:
            out = out_i
            count = count_i
        else:
            out += out_i
            count += count_i
    mask = count > 0
    out[mask] /= count[mask].reshape([int(mask.sum().get()), *([1]*(out.ndim-1))])
    return out, count


def reduce_within_clusters_np(X, n_clusters, labels, sample_weight=None, reduce_weight=True):
    X = _as_numpy(X).astype(np.float64, copy=False)
    labels = _as_numpy(labels).astype(np.int64, copy=False)
    sample_weight = _as_numpy(sample_weight)
    if sample_weight is not None:
        sample_weight = sample_weight.astype(np.float64, copy=False)
        weighted = X * sample_weight.reshape([X.shape[0], *([1] * (X.ndim - 1))])
        count = np.bincount(labels, weights=sample_weight, minlength=n_clusters).astype(np.float64)
    else:
        weighted = X
        count = np.bincount(labels, minlength=n_clusters).astype(np.float64)

    out = np.zeros([n_clusters, *X.shape[1:]], dtype=np.float64)
    np.add.at(out, labels, weighted)
    if reduce_weight:
        mask = count > 0
        out[mask] /= count[mask].reshape([mask.sum(), *([1] * (out.ndim - 1))])
    return out, count


def reduce_within_clusters_chunked_np(X, n_clusters, labels, sample_weight=None, chunk_size=None):
    X = _as_numpy(X)
    labels = _as_numpy(labels)
    sample_weight = _as_numpy(sample_weight)
    if chunk_size is None:
        chunk_size = X.shape[0]
    out = None
    count = None
    for i in range(0, X.shape[0], chunk_size):
        out_i, count_i = reduce_within_clusters_np(
            X[i:i + chunk_size],
            n_clusters,
            labels[i:i + chunk_size],
            sample_weight[i:i + chunk_size] if sample_weight is not None else None,
            reduce_weight=False,
        )
        if out is None:
            out = out_i
            count = count_i
        else:
            out += out_i
            count += count_i
    mask = count > 0
    out[mask] /= count[mask].reshape([mask.sum(), *([1] * (out.ndim - 1))])
    return out, count
