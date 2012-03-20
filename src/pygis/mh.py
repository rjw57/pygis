from pylab import *
import pygis.path

def vlen(vec):
    """Return the Euclidean length of the vector vec."""
    return sqrt(vdot(vec, vec))

def line_cost(start, end, grad):
    """Given some gradient raster, calculate the cost of the line start -> end.
    
    """

    start = array(start)
    end = array(end)
    delta = end - start
    delta_len = vlen(delta)
    direction = delta / delta_len

    step = array(grad.pixel_linear_shape).min()
    line_points, dists = pygis.path.points_on_line(start, end, step)
    return integrate_cost(line_points, grad)

    line_points = array(line_points)
    dists = array(dists)

    mid_points = 0.5 * (line_points[:-1] + line_points[1:])
    seg_lengths = grad.linear_scale * (dists[1:] - dists[:-1])

    samples = grad.lanczos_sample(mid_points)
    if len(line_points) == 2:
        samples = [samples,]

    cost = 0.0
    for dh, gradient in zip(seg_lengths, samples):
        # get vert. distance in metres
        dv = vdot(gradient, direction) * dh

        # get linear distance
        d = vlen(array(dh, dv))

        # calculate the subjective gradient
        subj_gradient = vdot(gradient, direction)

        # increase cost
        cost += abs(subj_gradient) * d

    return cost

def integrate_cost(points, grad):
    """Given some gradient raster, calculate the cost of the lines segments
    joining the points assuming that the size of each line segment is small
    w.r.t pixel size.
    
    """

    points = np.array(points)
    assert len(points.shape) > 1 and points.shape[0] > 1

    midpoints = 0.5*(points[:-1] + points[1:])
    deltas = points[1:] - points[:-1]
    delta_lengths = np.sqrt((deltas * deltas).sum(axis=1)).transpose()
    directions = np.array(deltas, copy=True)
    directions[:,0] /= delta_lengths
    directions[:,1] /= delta_lengths

    samples = grad.lanczos_sample(midpoints)
    if len(points) == 2:
        samples = [samples,]

    cost = 0.0
    for direction, dh, gradient in zip(directions, delta_lengths, samples):
        # get vert. distance in metres
        dv = vdot(gradient, direction) * dh

        # get linear distance
        d = vlen(array(dh, dv))

        # calculate the subjective gradient
        subj_gradient = vdot(gradient, direction)

        # increase cost
        cost += abs(subj_gradient) * d

    return cost

def path_cost(points, grad):
    """Given a list of points specifying a path and a gradient raster, return the total cost of the path.

    """

    step = array(grad.pixel_linear_shape).min()
    int_points = []
    for start, end in zip(points[:-1], points[1:]):
        np, _ = pygis.path.points_on_line(start, end, step)
        int_points.extend(np[:-1])
    int_points.append(points[-1])
    return integrate_cost(int_points, grad)

#    cost = 0.0
#    for start, end in zip(points[:-1], points[1:]):
#        cost += line_cost(start, end, grad)
#    return cost

def index_after_distance(points, dist):
    """Given an input path and a distance, return the index of the first point
    on the path beyond or at that distance.

    """

    idx = 0
    length = 0.0
    for start, end in zip(points[:-1], points[1:]):
        length += vlen(end-start)
        idx += 1
        if length >= dist:
            return idx

    assert False

def cum_path_length(points):
    """Given a list of points, return the length of the piece-wise linear path after reaching each point.

    """

    cum_lengths = [0.0,]
    length = 0.0
    # for each segment...
    for start, end in zip(points[:-1], points[1:]):
        length += vlen(end-start)
        cum_lengths.append(length)
    return cum_lengths

def propose_path(path, braveness, pix_size):
    """Given an input path, propose a new path and return (new_path,
    forward_log_lik, backward_log_lik).

    """

    log_forward = 0
    log_inv = 0
    new_path = list(path)

    # choose what sort of step to do
    move = randint(3)
    if move == 0:
        # add move, inverse move is del

        # sample some index
        cum_lengths = cum_path_length(new_path)
        add_dist = uniform() * cum_lengths[-1]
        add_end = find(cum_lengths > add_dist)[0]
        add_start = add_end - 1

        # how likely is it we sample this segment
        start_dist = cum_lengths[add_start]
        end_dist = cum_lengths[add_end]
        log_forward += log(end_dist - start_dist) - log(cum_lengths[-1])

        # how long along this line segment do we add?
        add_alpha = (add_dist - start_dist) / (end_dist - start_dist)
        assert add_alpha >= 0.0
        assert add_alpha <= 1.0

        # create the new point
        seg_len = vlen(new_path[add_end] - new_path[add_start])
        sigma = braveness*seg_len*0.5
        dp = randn(2) * sigma
        log_forward += log(normpdf(dp,0,sigma)).sum() 
        new_point = dp + new_path[add_start] + add_alpha * (new_path[add_end] - new_path[add_start])

        # insert it
        new_path = new_path[:add_end] + [new_point,] + new_path[add_end:]

        # inverse step is choosing the right distance
        cum_lengths = cum_path_length(new_path)
        log_inv += log(cum_lengths[add_end+1] - cum_lengths[add_end]) - log(cum_lengths[-2])

    elif move == 1:
        # delete move, inverse move is add

        # sample some index
        cum_lengths = cum_path_length(new_path)
        del_dist = uniform() * cum_lengths[-2]
        del_end = find(cum_lengths > del_dist)[0]
        del_start = del_end - 1

        # how likely is it we sample this segment
        start_dist = cum_lengths[del_start]
        end_dist = cum_lengths[del_end]
        log_forward += log(end_dist - start_dist) - log(cum_lengths[-2])

        # assert this wouldn't remove the last point:
        assert del_end < len(new_path)-1

        del_p = new_path[del_end]
        del new_path[del_end]

        # get the new start and end points
        p1 = new_path[del_end-1]
        p2 = new_path[del_end]
        seg_len = vlen(p2 - p1)

        # and how likely is it to add this point (dubious)
        delta1 = p2 - p1
        delta2 = del_p - p1
        alpha = vdot(delta1, delta2)
        alpha /= vlen(delta1) * vlen(delta2)

        sigma = braveness*seg_len*0.5
        marginal = 0.0
        for alpha in arange(0,1.1,0.1):
            dp = del_p - (p1 + alpha*delta1)
            marginal += normpdf(dp,0,sigma).prod()
        marginal /= 10.0

        cum_lengths = cum_path_length(new_path)
        log_inv += log(marginal)
        log_inv += log(cum_lengths[del_end] - cum_lengths[del_start]) - log(cum_lengths[-1])
    elif move == 2:
        # move a point, lik. of forward = lik. of backward
        if len(new_path) > 2:
            p_idx = randint(1, len(new_path)-1)
            seg_len = vlen(new_path[p_idx+1] - new_path[p_idx-1])
            new_path[p_idx] = new_path[p_idx] + randn(2) * 0.5 * seg_len * braveness
    else:
        assert false

    return new_path, log_forward, log_inv

def sample(path, cost, grad):
    # propose path
    pix_size = array(grad.pixel_proj_shape).min()
    new_path, log_forward, log_inv = propose_path(path, 0.2, pix_size)
    new_cost = path_cost(new_path, grad)

    log_alpha = log_inv - log_forward

    l = 5.0
    log_p_new = log(l) - l * new_cost
    log_p_old = log(l) - l * cost
    log_alpha += log_p_new - log_p_old

    #log_alpha -= len(new_path) # prior on path waypoint count
    #log_alpha -= path_length(new_path) # prior on distance

    alpha = exp(log_alpha)
    if uniform() < alpha:
        return new_path, new_cost
    else:
        return path, cost

# vim:filetype=python:sts=4:et:ts=4:sw=4
