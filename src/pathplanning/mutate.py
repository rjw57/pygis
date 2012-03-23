import numpy as np
import scipy.stats as stats

def update_move(orig_path):
    """Choose some distance along the path and then the closest point to that
    distance. Distances are chosen so that the start and end points are not
    considered.

    Join a line segment between the points on either side of the point to be
    moved and displace it perpendicular to this line by an amount drawn from a
    normal distribution with standard deviation proportional to the segment
    length.

    """
    global bravery

    log_forward = 0.0
    log_backward = 0.0
    path = orig_path.copy()
    if len(path.points) < 3:
        return path, -np.inf, -np.inf

    mid_dists = 0.5 * (path.dists[:-1] + path.dists[1:])
    distance = np.random.uniform(mid_dists[0], mid_dists[-1])
    closest_idx = np.argmin(np.abs(path.dists - distance))

    factor = \
            np.log(mid_dists[closest_idx] - mid_dists[closest_idx-1]) - \
            np.log(mid_dists[-1] - mid_dists[0])
    log_forward += factor
    log_backward += factor

    left = path.points[closest_idx-1]
    right = path.points[closest_idx+1]
    to_change = path.points[closest_idx]

    segment = right - left
    segment_len = np.sqrt(np.dot(segment, segment))
    normal = np.array((segment[1], -segment[0])) / segment_len
    sigma = segment_len * bravery
    displacement = np.random.randn() * sigma

    factor = stats.norm.logpdf(displacement, 0, sigma)
    log_forward += factor
    log_backward += factor

    path.update(closest_idx, displacement + to_change)

    return path, log_forward, log_backward

bravery = 0.2

def split_move(orig_path):
    """Choose some distance along the path and split it at that point. Move the
    new point perpendicular to the split by an amount drawn from a normal
    distribution with standard deviation proportional to the original segment
    length.

    """
    global bravery

    log_forward = 0.0
    log_backward = 0.0
    path = orig_path.copy()

    distance = np.random.uniform(0.0, path.length())
    new_idx, new_p = path.split(distance)
    # log_forward += -np.log(path.length())

    split_start = path.points[new_idx-1]
    split_end = path.points[new_idx+1]
    segment = (split_end - split_start).ravel()
    segment_len = np.sqrt(np.vdot(segment, segment))

    normal = np.array((segment[1], -segment[0])) / segment_len
    sigma = segment_len * bravery
    displacement = np.random.randn() * sigma
    log_forward += stats.norm.logpdf(displacement, 0, sigma)
    path.update(new_idx, new_p + displacement * normal)

    log_backward += -np.log(len(path.points)-2)

    return path, log_forward, log_backward

def remove_move(orig_path):
    """Choose some index along the path and remove the point at that index.
    
    We use indices rather than distance to favour removing little cusps.

    """
    global bravery

    log_forward = 0.0
    log_backward = 0.0
    path = orig_path.copy()
    if len(path.points) < 3:
        return path, -np.inf, -np.inf

    rm_idx = np.random.randint(1, len(path.points)-1)
    log_forward += -np.log(len(path.points)-2)

    assert rm_idx > 0
    assert rm_idx < len(path.points) - 1

    left = path.points[rm_idx-1]
    right = path.points[rm_idx+1]
    removed = path.remove(rm_idx)

    # log_backward += -np.log(path.length())

    segment = right - left
    segment_len = np.sqrt(np.dot(segment, segment))
    direction = segment / segment_len

    delta = removed - left
    alpha = np.dot(delta, direction) / segment_len
    centre = alpha * segment + left

    displacement_vec = removed - centre
    displacement = np.sqrt(np.dot(displacement_vec, displacement_vec))
    sigma = segment_len * bravery
    log_backward += stats.norm.logpdf(displacement, 0, sigma)

    return path, log_forward, log_backward

def mutate_path(p):
    """Propose a mutation to path *p* and return a tuple containing mutated
    path, the log likelihood of the forward move which generated the proposal
    and the log likelihood of proposing the original path from the mutated
    path.

    """

    move = np.random.randint(0,3)
    if move == 0:
        return split_move(p)
    elif move == 1:
        return remove_move(p)
    elif move == 2:
        return update_move(p)
    else:
        assert False
