"""
Created on Feb 20, 2017

@author: jumabek
"""

import argparse
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from ..coco import COCO


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.0
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def write_anchors_to_file(centroids, X, anchor_file):
    f = open(anchor_file, "w")

    anchors = centroids.copy()
    widths = anchors[:, 0]
    heights = anchors[:, 1]
    sorted_indices = np.argsort(heights)
    print(anchors.shape)
    print("Anchors = ", anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write("%0.2f,%0.2f, " % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write("%0.2f,%0.2f\n" % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))

    print("avg_IOU(X, centroids): ", avg_IOU(X, centroids))
    f.write("%f\n" % (avg_IOU(X, centroids)))
    print()


def kmeans(X, centroids, eps, anchor_file, num_clusters):
    print("kmeans clustering, k = ", num_clusters)
    N = X.shape[0]
    iterations = 0
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        # print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, X, anchor_file)
            return

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()


def compute_anchors_old(
    ann_path: Path, output_dir: Path, num_clusters: Union[int, List[int]] = [1, 12]
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    annotation_dims = []
    # size = np.zeros((1, 1, 3))
    coco = COCO(ann_path)
    for ann in coco.dataset["annotations"]:
        annotation_dims.append(tuple(map(float, (ann["bbox"][2], ann["bbox"][3]))))
    annotation_dims = np.array(annotation_dims)
    eps = 0.005

    if isinstance(num_clusters, list):
        for _num_clusters in range(num_clusters[0], num_clusters[1]):
            anchor_file = output_dir / f"anchors-num_clusters_{_num_clusters}.txt"
            indices = [random.randrange(annotation_dims.shape[0]) for i in range(_num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims, centroids, eps, anchor_file, _num_clusters)
            print("centroids.shape", centroids.shape)
    else:
        anchor_file = output_dir / f"anchors-num_clusters_{num_clusters}.txt"
        indices = [random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims, centroids, eps, anchor_file, num_clusters)
        print("centroids.shape", centroids.shape)


def scale_dim(w, h, from_w=1280, from_h=720, dim_to=640):
    return (
        w * (dim_to / from_w),
        h * (dim_to / from_h),
        # (h * (dim_to / from_h)) / (w * (dim_to / from_w)),
    )


def compute_anchors(
    ann_path: Path,
    output_dir: Path,
    num_clusters: Union[int, List[int]] = [1, 12],
    stop_at_iou: float = None,
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    annotation_dims = []
    # size = np.zeros((1, 1, 3))
    coco = COCO(ann_path)
    for ann in coco.dataset["annotations"]:
        img = coco.imgs[ann["image_id"]]
        annotation_dims.append(
            scale_dim(ann["bbox"][2], ann["bbox"][3], from_w=img["width"], from_h=img["height"])
        )
    annotation_dims = np.array(annotation_dims)
    eps = 0.005

    results = {}
    if isinstance(num_clusters, list):
        for _num_clusters in range(num_clusters[0], num_clusters[1]):
            print("")
            print("kmeans clustering, k = ", _num_clusters)
            kmeans = KMeans(n_clusters=_num_clusters, random_state=0).fit(annotation_dims)
            clusters = kmeans.cluster_centers_[np.argsort(kmeans.cluster_centers_[:, 1])]
            iou = avg_IOU(annotation_dims, clusters)
            results[_num_clusters] = {
                "labels": kmeans.labels_,
                "clusters": clusters,
                "iou": iou,
            }
            print("Labels: ", kmeans.labels_)
            print("Clusters: ", clusters)
            print("h/w ratios: ", clusters[:, 1] / clusters[:, 0])
            print("Avg. IoU: ", iou)
            if stop_at_iou and iou >= stop_at_iou:
                print(f"Desired IoU {stop_at_iou} reached, ending loop early at k={_num_clusters}")
                break
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(annotation_dims)
        clusters = kmeans.cluster_centers_[np.argsort(kmeans.cluster_centers_[:, 1])]
        iou = avg_IOU(annotation_dims, clusters)
        results[num_clusters] = {
            "labels": kmeans.labels_,
            "clusters": clusters,
            "iou": iou,
        }
        print(kmeans.labels_)
        print(kmeans.cluster_centers_)
    return results
