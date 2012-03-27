from pygis.raster import *

def line_dist(start, end, r):
    # get an image of x and y proj. coords
    pxs = np.arange(0, r.data.shape[1])
    pys = np.arange(0, r.data.shape[0])
    Xp, Yp = np.meshgrid(pxs, pys)

    pixels = np.vstack((Xp.ravel(), Yp.ravel())).transpose()
    points = r.pixel_to_proj(pixels)
    X = points[:,0].reshape(Xp.shape)
    Y = points[:,1].reshape(Yp.shape)

    # see http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    # if line is X = A + t * N (where N.N = 1)
    # dist of P = length((A-P) - ((A-P) . N)N)
    #           = length(B - (B.N)N), B = A-P
    
    N = np.array(end) - np.array(start)
    separation = np.sqrt(np.dot(N, N))
    N /= separation
    
    Bx = start[0] - X
    By = start[1] - Y
    
    BdotN = Bx * N[0] + By * N[1]
    
    Bx -= BdotN * N[0]
    By -= BdotN * N[1]

    # perpendicular distance
    perp_dist = np.sqrt(Bx*Bx + By*By) * r.linear_scale
    
    # end-point distances
    start_dx = X - start[0]
    start_dy = Y - start[1]
    start_dist = np.sqrt(start_dx*start_dx + start_dy*start_dy)
     
    end_dx = X - end[0]
    end_dy = Y - end[1]
    end_dist = np.sqrt(end_dx*end_dx + end_dy*end_dy)
   
    # where is the perpendicular distance valid?
    dists = perp_dist
    np.putmask(dists, BdotN > 0.0, start_dist)
    np.putmask(dists, BdotN < -separation, end_dist)

    return similar_raster(dists, r)

def path_dist(path, r):
    dist = np.ones(r.data.shape) * np.inf
    for start, end in zip(path[:-1], path[1:]):
        dist = np.minimum(dist, line_dist(start, end, r).data)
    return similar_raster(dist, r)
