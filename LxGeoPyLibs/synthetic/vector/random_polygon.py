import numpy as np
import random


def generateConvex(n):
    '''
    Generate convex shappes according to Pavel Valtr's 1995 alogrithm. Ported from
    Sander Verdonschot's Java version, found here:
    https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
    '''
    # initialise random coordinates
    X_rand, Y_rand = np.sort(np.random.random(n)), np.sort(np.random.random(n))
    X_new, Y_new = np.zeros(n), np.zeros(n)
    # divide the interior points into two chains
    last_true = last_false = 0
    for i in range(1, n):
        if i != n - 1:
            if random.getrandbits(1):
                X_new[i] = X_rand[i] - X_rand[last_true]
                Y_new[i] = Y_rand[i] - Y_rand[last_true]
                last_true = i
            else:
                X_new[i] = X_rand[last_false] - X_rand[i]
                Y_new[i] = Y_rand[last_false] - Y_rand[i]
                last_false = i
        else:
            X_new[0] = X_rand[i] - X_rand[last_true]
            Y_new[0] = Y_rand[i] - Y_rand[last_true]
            X_new[i] = X_rand[last_false] - X_rand[i]
            Y_new[i] = Y_rand[last_false] - Y_rand[i]
    # randomly combine x and y and sort by polar angle
    np.random.shuffle(Y_new)
    vertices = np.stack((X_new, Y_new), axis=-1)
    vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]
    # arrange points end to end to form a polygon
    vertices = np.cumsum(vertices, axis=0)
    # center around the origin
    x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])
    vertices[:, 0] += ((x_max - np.min(vertices[:, 0])) / 2) - x_max
    vertices[:, 1] += ((y_max - np.min(vertices[:, 1])) / 2) - y_max
    return vertices

##################
# https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

import math, random
from typing import List, Tuple


def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")
    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)
    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]
    return points


def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle
    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))
