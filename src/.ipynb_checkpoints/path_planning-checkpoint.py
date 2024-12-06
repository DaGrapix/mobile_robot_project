from typing import List
import cv2
import numpy as np

import networkx as nx

def create_obstacle_map(extended_polygons, image_shape):
    obstacle_map = np.ones(image_shape[:2])
    for polygon in extended_polygons:
        cv2.fillPoly(obstacle_map, [polygon], 0)
    return obstacle_map

def in_image(point, image):
    return 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]

def is_visible(p1, p2, obstacle_map):
    """
    This function checks if we can reach p2 from p1 without colliding with the obstacle map.
    We first compute the equation of the line segment between p1 and p2.
    Then we check at each pixel along the line segment if it is part of an obstacle given the obstacle map.
    """
    # check if p1 and p2 are inside of the map
    if not in_image(p1, obstacle_map) or not in_image(p2, obstacle_map):
        return False
    
    if p1[0] > p2[0]:
        p3 = p2
        p2 = p1
        p1 = p3
    
    x = np.linspace(p1[0].astype(int) + 1, p2[0].astype(int) - 1, p2[0].astype(int) - p1[0].astype(int) - 1).astype(int)

    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    intercept = p1[1]

    f = lambda x: slope * (x - p1[0]) + intercept
    y = f(x).astype(int)

    for i in range(len(x)):
        if obstacle_map[y[i], x[i]] == 0:
            return False
    return True

def create_adjacency_matrix_pts(points, obstacle_map):
    """
    Creates an adjacency matrix for the graph formed by the points.
    If the points are visible to each other, then the edge value is the euclidean distance between the points.
    Otherwise, their is no edge between the points.
    """
    n = points.shape[0]
    adjacency_matrix = np.zeros((n, n))

    l = np.arange(n)

    for n1 in range(len(points)):
        for n2 in range(n1+1, len(points)):
            p1 = points[n1]
            p2 = points[n2]
            if is_visible(p1, p2, obstacle_map):
                edge_value = np.linalg.norm(p1 - p2)
                adjacency_matrix[n1, n2] = edge_value
                adjacency_matrix[n2, n1] = edge_value
    return adjacency_matrix

def create_adjacency_matrix(polygons, obstacle_map):
    """
    Creates an adjacency matrix for the graph formed by the points in the polygons. Note that here, we may use this function with the
    polygons as well as the start position, and key points.
    The adjacency matrix is created by connecting the points which are visible.
    """
    points = [[*polygon.squeeze(1)] for polygon in polygons]
    points = np.concatenate(points)

    adjacency_matrix = create_adjacency_matrix_pts(points, obstacle_map)
    
    # adding edges of polygons to the adjacency matrix
    n_buffer = 0
    for polygon in polygons:
        polygon = polygon.squeeze(1)
        n = polygon.shape[0]
        for i in range(n):
            if not in_image(polygon[i], obstacle_map) or not in_image(polygon[(i + 1) % n], obstacle_map):
                continue
            adjacency_matrix[i + n_buffer, (i + 1) % n + n_buffer] = np.linalg.norm(polygon[i] - polygon[(i + 1) % n])
            adjacency_matrix[(i + 1) % n + n_buffer, i + n_buffer] = np.linalg.norm(polygon[i] - polygon[(i + 1) % n])
        n_buffer += n
    return adjacency_matrix, points


import networkx as nx


def compute_paths(start_position, key_points, polygons, obstacle_map):
    """
    Computes the shortest path between each pair of points in the graph formed by the start position, key points and polygons
    """
    n_targets = len(key_points)

    poly_list = polygons.copy()
    poly_list.append(start_position[None, None, :])
    poly_list.extend([kp[None, None, :] for kp in key_points])

    adjacency_matrix, points = create_adjacency_matrix(poly_list, obstacle_map)
    graph = nx.from_numpy_array(adjacency_matrix)
    n_poly_points = len(points) - n_targets - 1

    distances_dict = {}
    path_dict = {}

    # inspired from https://medium.com/@davidlfliang/intro-python-algorithms-traveling-salesman-problem-ffa61f0bd47b
    for i in range(n_poly_points, len(points)):
        for j in range(i+1, len(points)):
            source = i
            target = j
            path = nx.shortest_path(graph, source=source, target=target, weight='weight')
            
            path_cost = 0
            for k in range(len(path) - 1):
                path_cost += adjacency_matrix[path[k]][path[k+1]]
            distances_dict[(i, j)] = path_cost
            path_dict[(i, j)] = path
    
    return distances_dict, path_dict, points, adjacency_matrix

import itertools

def calculate_cost(route, distances):
    """Calculates the cost of a route given a dictionary of costs between each pair of points"""
    # code inspired from https://medium.com/@davidlfliang/intro-python-algorithms-traveling-salesman-problem-ffa61f0bd47b
    total_cost = 0
    n = len(route)
    for i in range(n-1):
        current_city = route[i]
        next_city = route[i + 1]
        # Look up the distance in both directions
        if (current_city, next_city) in distances:
            total_cost += distances[(current_city, next_city)]
        else:
            total_cost += distances[(next_city, current_city)]
    return total_cost

def path_planning(start_position, key_points, polygons, img_shape=(700, 1100)):
    """
    Brute force algorithm to solve the TSP problem.
    This algorithm works in two parts. We first start by computing the shortest path between each pair of points in {start position, key points}
    that make use of the full adjacency graph comprising of the {start position, key points, polygon points}.
    This gives us a dictionary of shortest paths and their associated costs between each pair of nodes we need to visit.
    Once this is computed, our goal is to find the shortest path that visits all the key points from the start position.
    We can loop over all possible permutations of the key points, compute the cost of each associated path, as the sum of the costs of each pair of points in the path.
    The final cost must also include the cost of going from the start position to the first key point.
    We then select the path with the minimum cost.
    """
    # inspired from https://medium.com/@davidlfliang/intro-python-algorithms-traveling-salesman-problem-ffa61f0bd47b
    obstacle_map = create_obstacle_map(polygons, img_shape)
    distances_dict, path_dict, points, adjacency_matrix = compute_paths(start_position, key_points, polygons, obstacle_map)
    n_targets = len(key_points)
    n_poly_points = len(points) - n_targets - 1
    
    # exclude the start position from the landmarks
    landmarks = list(range(n_poly_points + 1, len(points)))
    robot_id = n_poly_points

    all_permutations = list(itertools.permutations(landmarks))
    min_cost = float('inf')
    optimal_route = None

    # Iterate over all permutations and calculate costs
    for perm in all_permutations:
        cost = calculate_cost(perm, distances_dict)
        cost += distances_dict[(robot_id, perm[0])]
        if cost < min_cost:
            min_cost = cost
            optimal_route = perm

    optimal_route = [robot_id] + list(optimal_route)
    global_path_ids = [robot_id]
    optimal_route_positions = [points[robot_id]]

    for i in range(len(optimal_route) - 1):
        if (optimal_route[i], optimal_route[i+1]) in path_dict:
            path = path_dict[(optimal_route[i], optimal_route[i+1])]
        else:
            path = path_dict[(optimal_route[i+1], optimal_route[i])][::-1]
        path_extension = path[1:]
        path_positions = [points[id] for id in path_extension]
        optimal_route_positions.extend(path_positions)
        global_path_ids.extend(path_extension)

    return optimal_route, optimal_route_positions, points, adjacency_matrix