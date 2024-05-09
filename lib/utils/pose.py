import os
import re
import math
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from typing import List, Tuple, Dict

tqdm.pandas()


def load_poses(data_dir, split='train', num_frames=64):
    data = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                label = re.search(r'p([0-9]+)_s([0-9]+)_a([0-9]+)_r([0-9]+)', file)
                if label:
                    label = int(label.group(1))
                    df = pd.read_csv(os.path.join(root, file))
                    df = df.drop(columns=['frame'])
                    df = df.dropna()
                    if len(df) < num_frames:
                        continue
                    df = df.iloc[:num_frames]
                    data.append(df.values)
                    labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def load_data(data_dir, split='train', num_frames=64):
    data, labels = load_poses(data_dir, split, num_frames)
    return data, labels


def project_pose_to_2d(pose_3d, K, R, t):
    pose_3d = pose_3d.reshape(-1, 3)
    pose_3d = pose_3d @ R.T + t
    pose_2d = pose_3d @ K.T
    pose_2d = pose_2d[:, :2] / pose_2d[:, 2:]
    return pose_2d

def project_pose_into_image(pose_2d: List[np.float32, np.float32], image_size) -> List[np.array]:
    """Project pose into image coordinates.
    
    Args:
        pose_2d: Pose in normalized coordinates.
        image_size: Image size in pixels.

    Returns:
        Pose in image coordinates.
    """
    pose_2d = pose_2d.copy()
    pose_2d[:, 0] = pose_2d[:, 0] * image_size[0] + image_size[0] / 2
    pose_2d[:, 1] = -pose_2d[:, 1] * image_size[1] + image_size[1] / 2
    return pose_2d

def project_pose_into_image_batch(pose_2d: List[np.float32, np.float32], image_size) -> List[np.array]:
    """Project pose into image coordinates.
    
    Args:
        pose_2d: Pose in normalized coordinates.
        image_size: Image size in pixels.

    Returns:
        Pose in image coordinates.
    """
    pose_2d = pose_2d.copy()
    pose_2d[:, :, 0] = pose_2d[:, :, 0] * image_size[0] + image_size[0] / 2
    pose_2d[:, :, 1] = -pose_2d[:, :, 1] * image_size[1] + image_size[1] / 2
    return pose_2d


def project_pose_into_heatmaps(pose_2d: List[np.float32, np.float32], image_size, heatmap_size) -> List[np.array]:
    """Project pose into heatmaps.
    
    Args:
        pose_2d: Pose in image coordinates.
        image_size: Image size in pixels.
        heatmap_size: Heatmap size.

    Returns:
        Pose in heatmaps.
    """
    pose_2d = pose_2d.copy()
    pose_2d[:, 0] = pose_2d[:, 0] / image_size[0] * heatmap_size[0]
    pose_2d[:, 1] = pose_2d[:, 1] / image_size[1] * heatmap_size[1]
    # Add pose_2d[:, 3] into as heatmap color channel and scale min max scale
    
    
    return pose_2d