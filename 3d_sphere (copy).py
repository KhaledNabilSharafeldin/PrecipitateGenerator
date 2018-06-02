import os
import numpy as np
import random as r
import ast

INPUT_FILE_NAME = '13g606060.geom'
OUTPUT_FILE_NAME = '13g606060_OUTptrn.geom'

pptValue = 265
pptCenter = [30, 30, 30]
radius = 5
minRadius = 2
maxRadius = 5


def check_all_numbers(list):
    for element in list:
        if not element.isnumeric():
            return False
    return True


def extract_size(arr):
    return [int(arr[2]), int(arr[4]), int(arr[6])]


space = []
size = []
original_lines = []
with open(INPUT_FILE_NAME) as fp:
    line = "run"
    current_y = 0
    current_z = 0
    skip_saving = True
    while line:
        line_mod = line.rstrip()
        line_mod = line_mod.replace(' ', '\t')
        arr = line_mod.split('\t')
        if arr[0] == 'grid':
            size = extract_size(arr)
            space = np.zeros((size[0], size[1], size[2]))
        else:
            if check_all_numbers(arr):
                idx = 0
                skip_saving = True
                for elem in arr:
                    space[idx, current_y, current_z] = int(elem)
                    idx = idx + 1
                current_y = current_y + 1
                if current_y == size[1]:
                    current_z = current_z + 1
                    current_y = 0
        if (not skip_saving):
            original_lines.append(line)
        line = fp.readline()
        skip_saving = False


# print(space)

def list_to_string(list):
    return ''.join(map(str, list))


def string_to_list(word):
    return ast.literal_eval(word)


def is_safe(move, max):
    return move >= 0 and move < max


# global volFraction = 0
# visits all the nodes of a graph (connected component) using BFS
def bfs_connected_component(start, end, value):
    # keep track of all visited nodes
    explored = []
    # keep track of nodes to be checked
    queue = [start]
    space[start[0], start[1], start[2]] = value
    volFraction = 1
    levels = {}  # this dict keeps track of levels
    levels[list_to_string(start)] = 0  # depth of start node is 0

    visited = [start]  # to avoid inserting the same node twice into the queue
    x_moves = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1]
    y_moves = [0, 0, 0, 1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 1, -1, 1, -1]
    z_moves = [0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0]
    # keep looping until there are nodes still to be checked
    while queue:
        # pop shallowest node (first node) from queue
        node_location = queue.pop(0)
        explored.append(node_location)
        for i in range(27):
            x_move = x_moves[i] + node_location[0]
            y_move = y_moves[i] + node_location[1]
            z_move = z_moves[i] + node_location[2]
            if (not is_safe(x_move, size[0])) or (not is_safe(y_move, size[1])) or (not is_safe(z_move, size[2])):
                continue

            neighbour_level = levels[list_to_string(node_location)] + 1
            neighbour = [x_move, y_move, z_move]
            dist = [start[0] - neighbour[0], start[1] - neighbour[1], start[2] - neighbour[2]]
            Arr = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]
            if neighbour not in visited and neighbour_level <= end and Arr <= end * end:
                queue.append(neighbour)
                visited.append(neighbour)
                space[x_move, y_move, z_move] = value
                volFraction += 1
                levels[list_to_string(neighbour)] = neighbour_level
                # print(neighbour, ">>", levels[neighbour])

    # print(levels)

    return volFraction


def looper(pptCenterList, rad):
    volFractionTotal = 0
    conformingValue = []
    for i in range(len(pptCenterList)):
        conformingValue.append(space[pptCenterList[i]])
        volFractionTotal += bfs_connected_component(pptCenterList[i], rad[i], pptValue + i)
    return volFractionTotal, conformingValue


def isOverlap(pt1, r1, pts, rs):
    for i in range(len(pts)):
        pt2 = pts[i]
        r2 = rs[i]
        dist = (pt2[0] - pt1[0]) * (pt2[0] - pt1[0]) + (pt2[1] - pt1[1]) * (pt2[1] - pt1[1]) + (pt2[2] - pt1[2]) * (
                    pt2[2] - pt1[2])
        if dist < (r1 + r2) * (r1 + r2):
            return False
    return True


def rand_ppt(n):
    rand_ppt_center_list = []
    rad = rand_radius(n)
    for i in range(n):
        newPT = [r.randrange(0 + rad[i], size[0] - rad[i], 1), r.randrange(0 + rad[i], size[1] - rad[i], 1),
                 r.randrange(0 + rad[i], size[2] - rad[i], 1)]
        if i > 0:
            while (not (isOverlap(newPT, rad[i], rand_ppt_center_list, rad))):
                newPT = [r.randrange(0 + rad[i], size[0] - rad[i], 1), r.randrange(0 + rad[i], size[1] - rad[i], 1),
                         r.randrange(0 + rad[i], size[2] - rad[i], 1)]

        rand_ppt_center_list.append(newPT)
    return rand_ppt_center_list, rad


def rand_radius(n):
    rand_radius_list = []
    for i in range(n):
        rand_radius_list.append(r.randrange(minRadius, maxRadius))
    return rand_radius_list


def linear_pattern_z(n, x, y, spacing = 0):
    temp = int(spacing)
    if not spacing:
        spacing = size[2] / n
    linear_pattern_list = []
    for i in range(n):
        linear_pattern_list.append([x, y, int(spacing / 2 + spacing * i)+temp])
    return linear_pattern_list

def linear_pattern_y(n, x, z, spacing = 0):
    temp = int(spacing)
    if not spacing:
        spacing = size[1] / n
    linear_pattern_list = []
    for i in range(n):
        linear_pattern_list.append([x, int(spacing / 2 + spacing * i)+temp, z])
    return linear_pattern_list

def linear_pattern_x(n, y, z, spacing = 0):
    temp = int(spacing)
    if not spacing:
        spacing = size[0] / n
    linear_pattern_list = []
    for i in range(n):
        linear_pattern_list.append([int(spacing / 2 + spacing * i)+temp, y, z])
    return linear_pattern_list

def linear_pattern3D(x, y, z, R):
    spacingX = size[0] / x
    spacingY = size[1] / y
    spacingZ = size[2] / z
    linear_pattern_list = linear_pattern_x(x, int(spacingY/2), int(spacingZ/2))
    for i in range(x):
        linear_pattern_list += linear_pattern_y(y - 1, linear_pattern_list[i][0], linear_pattern_list[i][2], spacingY)

    for i in range(x - 1):
        for j in range(y):
            linear_pattern_list += linear_pattern_z(z - 1, linear_pattern_list[x+i*y+j][0], linear_pattern_list[x+i*y+j][1], spacingZ)

    for i in range(x):
        linear_pattern_list += linear_pattern_z(z - 1, linear_pattern_list[i][0],
                                                linear_pattern_list[i][1], spacingZ)

    return linear_pattern_list, const_radius(len(linear_pattern_list), R)

def const_radius(n, R):
    rand_radius_list = []
    for i in range(n):
        rand_radius_list.append(R) #r.randrange(minRadius, maxRadius)
    return rand_radius_list

# print(randPPT(3))
# ans = bfs_connected_component(pptCenter, radius, pptValue)  # returns ['A', 'B', 'C', 'E', 'D', 'F', 'G']
# print(ans)

#pts, radii = rand_ppt(15) #use for random generating

#radii = const_radius(16,3)

#pts = linear_pattern_x(5,3,3) + linear_pattern_x(10, 8, 8) #use for patterns
pts, radii = linear_pattern3D(5, 5, 5, 4)

ans, ans2 = looper(pts, radii)
print(ans)
print((ans / (len(space[0]) * len(space[1]) * len(space[2]))) * 100)
#print(ans2)
with open(OUTPUT_FILE_NAME, 'w') as the_file:
    for line in original_lines:
        the_file.write(line)
    for z in range(size[2]):
        for y in range(size[1]):
            line = ''
            for x in range(size[0]):
                line = line + str(int(space[x][y][z]))
                if x != size[0] - 1:
                    line = line + ' '
            line = line + '\n'
            the_file.write(line)
