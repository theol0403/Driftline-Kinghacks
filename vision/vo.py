from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .types import Pose2D


@dataclass
class VisualOdometryConfig:
    scale_m_per_px: float = 0.02
    max_matches: int = 80
    nfeatures: int = 1000


class VisualOdometry:
    def __init__(self, config: VisualOdometryConfig) -> None:
        self.scale = config.scale_m_per_px
        self.max_matches = config.max_matches
        self.orb = cv2.ORB_create(nfeatures=config.nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_gray: Optional[np.ndarray] = None
        self.last_kp = None
        self.last_des = None
        self.pose = Pose2D(0.0, 0.0, 0.0)

    def update(self, frame_bgr: np.ndarray) -> Optional[Pose2D]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if self.last_des is None or des is None or len(kp) < 6:
            self.last_gray = gray
            self.last_kp = kp
            self.last_des = des
            return None

        matches = self.matcher.match(self.last_des, des)
        if len(matches) < 8:
            self.last_gray = gray
            self.last_kp = kp
            self.last_des = des
            return None

        matches = sorted(matches, key=lambda m: m.distance)[: self.max_matches]
        pts_prev = np.float32([self.last_kp[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

        transform, _ = cv2.estimateAffinePartial2D(
            pts_prev, pts_curr, method=cv2.RANSAC
        )
        if transform is None:
            self.last_gray = gray
            self.last_kp = kp
            self.last_des = des
            return None

        dx_px = float(transform[0, 2])
        dy_px = float(transform[1, 2])
        dyaw = math.atan2(transform[1, 0], transform[0, 0])

        forward_m = -dy_px * self.scale
        lateral_m = dx_px * self.scale

        cos_yaw = math.cos(self.pose.yaw)
        sin_yaw = math.sin(self.pose.yaw)
        self.pose.x += forward_m * cos_yaw - lateral_m * sin_yaw
        self.pose.y += forward_m * sin_yaw + lateral_m * cos_yaw
        self.pose.yaw += dyaw

        self.last_gray = gray
        self.last_kp = kp
        self.last_des = des
        return Pose2D(self.pose.x, self.pose.y, self.pose.yaw)
