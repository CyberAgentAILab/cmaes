"""
_uhvi_archiving.py

This module implements the UHVI (uncrowded hypervolume improvement) evaluation.
Parts of this implementation are adapted from the `moarchiving` library.

Source: https://github.com/CMA-ES/moarchiving/blob/development/moarchiving/moarchiving.py

-------------------------------------------------------------------------------
ORIGINAL LICENSE FOR moarchiving:
The BSD 3-Clause License
Copyright (c) 2018-2025 Inria
Authors: Nikolaus Hansen, Nace Sever, Mila Nedić, Tea Tušar

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright and
   authors notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   and authors notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided with
   the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors nor the authors names may be used to endorse or promote
   products derived from this software without specific prior written
   permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
from math import sqrt
from bisect import bisect_left

from typing import List, Tuple, TypeAlias

Point2D: TypeAlias = Tuple[float, float]


class UHVIArchive2D:
    """
    Minimal 2D archive to evaluate Uncrowded Hypervolume Improvement (UHVI).

    This class maintains a lexicographically sorted Pareto front and computes
    the improvement a new point brings to the hypervolume.
    """

    def __init__(self, list_of_f_pairs: np.ndarray) -> None:
        """
        Initialize the archive and perform non-dominated sorting.

        Args:
            list_of_f_pairs: Initial set of objective pairs (e.g., [[f1, f2], ...]).
        """
        pts: List[Point2D] = [(float(p[0]), float(p[1])) for p in list_of_f_pairs]

        pts.sort()

        front: List[Point2D] = []
        best_f2 = float("inf")
        for f1, f2 in pts:
            if f2 < best_f2:
                front.append((f1, f2))
                best_f2 = f2

        self.front: List[Point2D] = front

    @staticmethod
    def _to_point(x: np.ndarray) -> Point2D:
        return (float(x[0]), float(x[1]))

    @staticmethod
    def _dominates(p1: Point2D, p2: Point2D) -> bool:
        """Check if p1 weakly Pareto-dominates p2."""
        return p1[0] <= p2[0] and p1[1] <= p2[1]

    @staticmethod
    def _in_domain(p: Point2D, r: Point2D) -> bool:
        """Check if the point is within the reference domain."""
        return p[0] < r[0] and p[1] < r[1]

    def _dominated_or_equal_by_front(self, p: Point2D) -> bool:
        if not self.front:
            return False
        i = bisect_left(self.front, p)
        if i > 0 and self._dominates(self.front[i - 1], p):
            return True
        if i < len(self.front) and self._dominates(self.front[i], p):
            return True
        return False

    def _distance_to_pareto_front(self, p: Point2D, r: Point2D) -> float:
        """
        Compute Euclidean distance to the empirical non-domination front.

        If the point is non-dominated and in-domain, distance is 0.0.
        """
        if self._in_domain(p, r) and not self._dominated_or_equal_by_front(p):
            return 0.0

        ref_d0 = max(0.0, p[0] - r[0])
        ref_d1 = max(0.0, p[1] - r[1])

        if not self.front:
            return sqrt(ref_d0 * ref_d0 + ref_d1 * ref_d1)

        # Distances to the two outer kink points (use one coordinate of the reference point).
        best_sq = min(
            max(0.0, p[0] - self.front[0][0]) ** 2 + ref_d1**2,
            ref_d0**2 + max(0.0, p[1] - self.front[-1][1]) ** 2,
        )

        if len(self.front) == 1:
            return sqrt(best_sq)

        start = bisect_left(self.front, p)
        # inner kink points: (front[idx].f1, front[idx-1].f2)
        for idx in range(start, 0, -1):
            if idx == len(self.front):
                continue
            right = self.front[idx]
            left = self.front[idx - 1]
            sq = max(0.0, p[0] - right[0]) ** 2 + max(0.0, p[1] - left[1]) ** 2
            if sq < best_sq:
                best_sq = sq
            if right[1] >= p[1] or idx == 1:
                break

        return sqrt(best_sq)

    @staticmethod
    def _hypervolume_2d(sub: List[Point2D], ref: Point2D) -> float:
        """
        Hypervolume of a 2D non-dominated sorted list (minimization) w.r.t. `ref`.
        """
        r0, r1 = ref
        hv = 0.0

        i = 0
        while i < len(sub) and not (sub[i][0] < r0 and sub[i][1] < r1):
            i += 1

        if i < len(sub):
            hv += (r0 - sub[i][0]) * (r1 - sub[i][1])
            i += 1

        while i < len(sub) and (sub[i][0] < r0 and sub[i][1] < r1):
            hv += (r0 - sub[i][0]) * (sub[i - 1][1] - sub[i][1])
            i += 1

        return hv

    def uhvi(self, s: np.ndarray, r: np.ndarray) -> float:
        """
        Compute the Uncrowded Hypervolume Improvement (UHVI).

        Args:
            s: The objective pair to evaluate.
            r: The reference point for hypervolume calculation.

        Returns:
            float: Positive value for HVI, negative for distance to the front.
        """
        sp = self._to_point(s)
        rp = self._to_point(r)

        # Exact match on the current front -> zero improvement.
        for p in self.front:
            if p == sp:
                return 0.0

        dist = self._distance_to_pareto_front(sp, rp)
        if dist > 0.0:
            return -dist

        # Non-dominated and in the reference domain -> hypervolume improvement (HVI).
        i0 = bisect_left(self.front, sp)

        i1 = i0
        while i1 < len(self.front) and sp[1] <= self.front[i1][1]:
            i1 += 1

        # Reference point for the local sub-front used in local HVI computation.
        r0 = self.front[i1][0] if i1 < len(self.front) else rp[0]
        r1 = self.front[i0 - 1][1] if i0 > 0 else rp[1]
        local_ref: Point2D = (r0, r1)

        sub = self.front[i0:i1]
        hv0 = self._hypervolume_2d(sub, local_ref)

        # Since s dominates the extracted sublist, the local hypervolume after insertion
        # is the rectangle area between s and the local reference point.
        hv1 = (r0 - sp[0]) * (r1 - sp[1])
        return hv1 - hv0
