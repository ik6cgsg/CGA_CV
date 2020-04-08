import numpy as np
import cv2
import argparse
import sys
from math import sin, cos, radians
from shapely.geometry import Polygon


def get_line(line):
    p1, p2 = line[0], line[1]
    a = (p1[1] - p2[1])
    b = (p2[0] - p1[0])
    c = (p1[0] * p2[1] - p2[0] * p1[1])
    return a, b, -c


def full_intersection(l1, l2):
    line1 = get_line(l1)
    line2 = get_line(l2)
    d = line1[0] * line2[1] - line1[1] * line2[0]
    dx = line1[2] * line2[1] - line1[1] * line2[2]
    dy = line1[0] * line2[2] - line1[2] * line2[0]
    if abs(d) > 1e-9:
        x = dx / d
        y = dy / d
        return [x, y]
    else:
        return None


def find_closed_contour(res, arr, num, lines_length):
    max_dist = 0
    ind = -1
    for i, el in enumerate(arr[num]):
        new_point = el[1]
        if new_point not in res:
            if len(res) == 0:
                res.append(el[1])
                find_closed_contour(res, arr, el[0], lines_length)
                return res
            last_added = res[len(res) - 1]
            dist = np.linalg.norm([new_point[0] - last_added[0], new_point[1] - last_added[1]])
            if dist > max_dist:
                max_dist = dist
                ind = i
    if ind == -1 or max_dist < (lines_length[num] / 2):
        return res
    el = arr[num][ind]
    res.append(el[1])
    find_closed_contour(res, arr, el[0], lines_length)
    return res


def rotate_polyline(points, pivot, degrees):
    """ Rotate polygon the given angle about its center. """
    theta = radians(degrees)  # Convert angle to radians
    cosang, sinang = cos(theta), sin(theta)

    cx = pivot.x
    cy = pivot.y

    new_points = []
    for p in points:
        x = p[0]
        y = p[1]
        tx, ty = x-cx, y-cy
        new_x = (tx*cosang + ty*sinang) + cx
        new_y = (-tx*sinang + ty*cosang) + cy
        new_points.append([new_x, new_y])

    return np.array(new_points)


def register_launch_arguments():
    parser = argparse.ArgumentParser(description='Serve the application')
    parser.add_argument('-i', '--image', help='path to source image file', default='./image.png')
    parser.add_argument('-s', '--structure', help='path to input txt file', default='./input.txt')
    parser.add_argument('-o', '--output', help='path to output txt file', default=None)
    return parser.parse_args()


def parse_input_txt(path):
    with open(path, "r") as f:
        content = f.read().split("\n")
    polylines = []
    num_primitives = int(content[0])
    for i in range(num_primitives):
        polyline_pts = []
        coordinates_arr = content[i + 1].split(",")
        assert (len(coordinates_arr) % 2 == 0)
        for j in range(len(coordinates_arr) // 2):
            x = int(coordinates_arr[j * 2])
            y = int(coordinates_arr[j * 2 + 1])
            polyline_pts.append([x, y])
        polylines.append(np.array(polyline_pts))

    return polylines


def find_appropriate_contours(img):
    cnt, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # only with children
    cnt = np.array(cnt)[hier[0][:, 2] != -1]
    return cnt


def line_point_dist(line, point):
    return [np.linalg.norm([line[0][0] - point[0], line[0][1] - point[1]]),
               np.linalg.norm([line[1][0] - point[0], line[1][1] - point[1]])]


def append_intersection(graph, inter_point, line1, ind1, line2, ind2):
    d1 = line_point_dist(line1, inter_point)
    d2 = line_point_dist(line2, inter_point)
    graph[ind1].append([ind2, inter_point, d1])
    graph[ind2].append([ind1, inter_point, d2])
    return graph


def find_lines_intersection(lines):
    intersection_graph = [[] for i in range(len(lines))]
    for i, l_1 in enumerate(lines):
        x1, y1, x2, y2 = l_1[0]
        line_1 = [[x1, y1], [x2, y2]]
        for j, l_2 in enumerate(lines[i + 1:]):
            x1, y1, x2, y2 = l_2[0]
            line_2 = [[x1, y1], [x2, y2]]
            inter_point = full_intersection(line_1, line_2)
            if inter_point is None:
                continue
            intersection_graph = append_intersection(intersection_graph, inter_point, line_1, i, line_2, j + i + 1)
    return intersection_graph


def find_shape_points(img_shape, contour):
    res = np.zeros(img_shape, np.uint8)
    cv2.drawContours(res, [contour], 0, 255, 1)
    lines = cv2.HoughLinesP(res, 0.5, np.pi / 360, 10, minLineLength=5, maxLineGap=2)

    if len(lines) < 3:
        return []

    lines_length = []
    for l_1 in lines:
        x1, y1, x2, y2 = l_1[0]
        lines_length.append(np.linalg.norm([x2 - x1, y2 - y1]))

    # show result
    # hough_res = np.zeros(img_shape, np.uint8)
    # for l_1 in lines:
    #     x1, y1, x2, y2 = l_1[0]
    #     cv2.line(hough_res, (x1, y1), (x2, y2), 150, 1)
    #     cv2.imshow("hough_res", hough_res)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # cv2.imshow("hough_res", hough_res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    intersection_graph = find_lines_intersection(lines)

    # remove inappropriate intersections
    for i, line_inter in enumerate(intersection_graph):
        if len(line_inter) > 2:
            sorted_p1 = list(np.array(sorted(line_inter, key=lambda x: x[2][0])))
            sorted_p2 = list(np.array(sorted(line_inter, key=lambda x: x[2][1])))
            intersection_graph[i] = [sorted_p1[0], sorted_p2[0]]

    shape_points = []
    shape_points = find_closed_contour(shape_points, intersection_graph, 0, lines_length)
    if len(shape_points) < 3:
        return []

    # show result
    # tmp = np.zeros(img_shape, np.uint8)
    # cv2.drawContours(tmp, [np.array(shape_points).astype(int)], 0, 255, -1)
    # cv2.imshow("final shape", tmp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return shape_points


def find_angle(transformed_polyline, img_cnt):
    angle_res = None
    max_iou = 0
    transformed_polygon = Polygon(transformed_polyline)

    for angle in range(0, 360):
        rotated_polyline = rotate_polyline(transformed_polyline, transformed_polygon.centroid, angle)
        img_rotated = np.zeros_like(img_cnt)
        cv2.drawContours(img_rotated, [np.array(rotated_polyline).astype(int)], 0, 255, -1)

        iou = np.count_nonzero(np.logical_and(img_rotated, img_cnt)) / np.count_nonzero(
            np.logical_or(img_rotated, img_cnt))

        if iou > max_iou:
            max_iou = iou
            angle_res = angle

    # show result
    # print(max_iou)

    if max_iou < 0.9:
        return None
    return angle_res


def find_polyline_with_transform(polylines, img_cnt, centroid, area):
    for count, polyline in enumerate(polylines):
        # transform scale
        scale = np.sqrt(area / Polygon(polyline).area)
        transformed_polyline = polyline * scale

        # transform shifts
        transformed_polygon = Polygon(transformed_polyline)
        dx = centroid[0] - transformed_polygon.centroid.x
        dy = centroid[1] - transformed_polygon.centroid.y
        transformed_polyline[:, 0] += dx
        transformed_polyline[:, 1] += dy

        # find angle (opposite rotation direction)
        angle = find_angle(transformed_polyline, img_cnt)
        if angle is None:
            continue
        angle = 360 - angle

        # find new shifts
        res_polyline = polyline * scale
        tmp = res_polyline.copy()
        for i in [0, 1]:
            res_polyline[:, i] = np.cos(radians(angle)) * tmp[:, i] \
                                 - ((-1) ** i) * np.sin(radians(angle)) * tmp[:, 1 - i]
        res_polygon = Polygon(res_polyline)
        dx = centroid[0] - res_polygon.centroid.x
        dy = centroid[1] - res_polygon.centroid.y

        # show results
        # res_polyline[:, 0] += dx
        # res_polyline[:, 1] += dy
        #
        # test_img = np.zeros_like(img_cnt)
        # cv2.drawContours(test_img, [np.array(res_polyline).astype(int)], 0, 255, -1)
        # cv2.imshow("contour", img_cnt)
        # cv2.imshow("transformed polyline", test_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return np.array([count, scale, angle, dx, dy]).astype(int)
    return None


def print_result(result_params_arr):
    print(str(len(result_params_arr)))
    for res_params in result_params_arr:
        tmp_str = ""
        for param in res_params[:-1]:
            tmp_str += (str(param) + ',')
        tmp_str += str(res_params[-1])
        print(tmp_str)


if __name__ == '__main__':
    args = register_launch_arguments()
    if args.output is not None:
        f_out = open(args.output, "w")
        sys.stdout = f_out

    polylines = parse_input_txt(args.structure)
    img_src = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    result_params_arr = []

    cnt = find_appropriate_contours(img_src)

    for c in cnt:
        shape_cnt = find_shape_points(img_src.shape, c)
        if len(shape_cnt) == 0:
            continue

        img_cnt = np.zeros_like(img_src)
        cv2.drawContours(img_cnt, [np.array(shape_cnt).astype(int)], 0, 255, -1)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_cnt, 8, cv2.CV_32S)
        area = stats[1][cv2.CC_STAT_AREA]
        centroid = centroids[1]

        transform = find_polyline_with_transform(polylines, img_cnt, centroid, area)
        if transform is not None:
            result_params_arr.append(transform)

    print_result(result_params_arr)
