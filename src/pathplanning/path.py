import numpy as np
import scipy.stats as stats

class Path(object):
    def __init__(self, points, copy=True):
        """A path object represents a path as a series of waypoints in projection co-ordinates.

        If *copy* is False, the points used to initialise the path will not be copied unless necessary.

        """

        self.points = np.array(points, copy=copy)
        assert self.points.shape[0] > 1
        assert self.points.shape[1] == 2
        self._update_dists()

    def copy(self):
        """Return a deep copy of the path."""

        return Path(self.points)

    def _update_dists(self):
        deltas = np.diff(self.points, axis=0)
        delta_dists = np.hstack(([0,], np.sqrt((deltas * deltas).sum(axis=1))))
        self.dists = np.cumsum(delta_dists)

    def length(self):
        return self.dists[-1]

    def segment(self, idx):
        """Return a tuple giving the start, end and length of the idx-th segment."""

        assert idx >= 0
        assert idx < len(self.points) - 1
        start = self.points[idx]
        end = self.points[idx+1]
        d = end - start
        return start, end, np.sqrt(np.dot(d,d))

    def split(self, distance):
        """Split the path at the specified distance. Return a tuple index of the new point and it's value."""

        assert distance < self.length()
        assert distance > 0

        right_idx = np.nonzero(self.dists > distance)[0][0]
        assert right_idx > 0
        assert right_idx < len(self.points)

        right_dist = self.dists[right_idx]
        left_dist = self.dists[right_idx-1]
        assert left_dist < distance
        assert right_dist > distance
        alpha = (distance - left_dist) / (right_dist - left_dist)

        left = self.points[right_idx-1]
        right = self.points[right_idx]
        new_point = alpha * (right - left) + left

        self.points = np.vstack((self.points[:right_idx], new_point, self.points[right_idx:]))
        self._update_dists()

        return right_idx, new_point

    def remove(self, idx):
        """Remove the point at the specified index. Returns the deleted point."""

        assert idx > 0
        assert idx < len(self.points) - 1
        del_p = self.points[idx]
        self.points = np.vstack((self.points[:idx], self.points[idx+1:]))
        self._update_dists()
        return del_p

    def update(self, idx, new_p):
        """Change the point at the specified index."""
        
        assert idx > 0
        assert idx < len(self.points) - 1
        self.points[idx] = new_p
        self._update_dists()
