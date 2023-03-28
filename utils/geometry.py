import numpy as np

import os


Segment = tuple[float, float, float, float]


def get_angle(segment: Segment) -> float:
    x1, y1, x2, y2 = segment
    return np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi


def filter_by_angle(
    segments: list[Segment], angle: float, threshold: float = 15
) -> list[Segment]:
    return [s for s in segments if abs(abs(get_angle(s)) - angle) < threshold]


def do_segments_intersect(segment1: Segment, segment2: Segment):
    x1, y1, x2, y2 = segment1
    x3, y3, x4, y4 = segment2

    denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denominator == 0:
        return False

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

    if ua >= 0 and ua <= 1 and ub >= 0 and ub <= 1:
        return True
    return False


def group_intersecting_segments(segments: list[Segment]) -> list[list[Segment]]:
    groups = []
    segments_set = set(segments)
    while segments_set:
        segment = segments_set.pop()
        group = [segment]
        for other_segment in segments_set.copy():
            if do_segments_intersect(segment, other_segment):
                segments_set.remove(other_segment)
                group.append(other_segment)
        groups.append(group)
    return groups


def extend_within_bounds(
    segments: Segment | list[Segment], bounds: tuple[float, float, float, float]
):
    if type(segments) != list:
        segments = [segments]

    xmin, ymin, xmax, ymax = bounds
    extended_segments = []

    for segment in segments:
        x1, y1, x2, y2 = segment

        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float("inf")
        intercept = y1 - slope * x1

        if slope != float("inf"):
            x1 = x1 - slope * 1_000_000
            y1 = y1 - slope * 1_000_000

        if slope != float("inf"):

            x2 = x2 + slope * 1_000_000
            y2 = y2 + slope * 1_000_000

            # extend the segment to meet the bounds
            if x1 <= xmin:
                x1 = xmin
                y1 = slope * x1 + intercept
            elif x1 >= xmax:
                x1 = xmax
                y1 = slope * x1 + intercept

            if x2 <= xmin:
                x2 = xmin
                y2 = slope * x2 + intercept
            elif x2 >= xmax:
                x2 = xmax
                y2 = slope * x2 + intercept

            if y1 <= ymin:
                y1 = ymin
                x1 = (y1 - intercept) / slope
            elif y1 >= ymax:
                y1 = ymax
                x1 = (y1 - intercept) / slope

            if y2 <= ymin:
                y2 = ymin
                x2 = (y2 - intercept) / slope
            elif y2 >= ymax:
                y2 = ymax
                x2 = (y2 - intercept) / slope
        else:
            y1 = ymin
            y2 = ymax

        # ensure the segment is within bounds
        x1 = min(max(x1, xmin), xmax)
        x2 = min(max(x2, xmin), xmax)
        y1 = min(max(y1, ymin), ymax)
        y2 = min(max(y2, ymin), ymax)

        extended_segments.append((x1, y1, x2, y2))
    return extended_segments


def vertically_close(s1: Segment, s2: Segment, threshold: int = 20) -> bool:
    return abs(s1[0] - s2[0]) < threshold and abs(s1[2] - s2[2]) < threshold


def group_vertically(segments: list[Segment]) -> list[list[Segment]]:
    groups = []
    segments_set = set(segments)
    while segments_set:
        segment = segments_set.pop()
        group = [segment]
        for other_segment in segments_set.copy():
            if vertically_close(segment, other_segment):
                segments_set.remove(other_segment)
                group.append(other_segment)
        groups.append(group)
    return groups


import argparse
from enum import Enum
import io

from google.cloud import vision
from PIL import Image, ImageDraw


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def draw_boxes(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        draw.polygon(
            [
                bound[0]["x"],
                bound[0]["y"],
                bound[1]["x"],
                bound[1]["y"],
                bound[2]["x"],
                bound[2]["y"],
                bound[3]["x"],
                bound[3]["y"],
            ],
            None,
            color,
        )
    return image


def get_document_bounds(image_file, feature):
    """Returns document bounds given an image."""
    client = vision.ImageAnnotatorClient()

    bounds = []

    with io.open(image_file, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    document = response.full_text_annotation

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    if feature == FeatureType.WORD:
                        bounds.append(word.bounding_box)

    # The list `bounds` contains the coordinates of the bounding boxes.
    return bounds


def bounds_to_dict(bounds):
    dico = []
    for bound in bounds:
        dico.append([{"x": v.x, "y": v.y} for v in bound.vertices])
    return dico


def get_bbox(filein):
    from django.core.files import File

    image = Image.open(filein)
    bounds = get_document_bounds(filein, FeatureType.WORD)

    bounds = bounds_to_dict(bounds)

    draw_boxes(image, bounds, "red")

    import io

    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG")

    image_file = File(image_buffer)

    return image_file, bounds


import io
import os

from unidecode import unidecode


def has_alnum(s):
    for i in s:
        if i.isalnum():
            return True
    return False


def process_row(row):
    try:
        num = " ".join(r.description for r in row[0] if has_alnum(r.description))

        num = num.upper()

        mistake_maps = {
            "L": "1",
            "I": "1",
            "H": "4",
            "S": "5",
            "O": "0",
            "(": "1",
            ")": "1",
        }

        for key, value in mistake_maps.items():
            num = num.replace(key, value)

        assert len(num) == 3

    except Exception:
        num = None
    try:
        nome = " ".join(r.description for r in row[1] if has_alnum(r.description))
    except Exception:
        nome = ""

    try:
        p1 = float(
            "".join(r.description for r in row[2] if has_alnum(r.description)).replace(
                ",", "."
            )
        )
        while p1 > 40:
            p1 /= 10
    except Exception:
        p1 = None
    try:
        p2 = float(
            "".join(r.description for r in row[3] if has_alnum(r.description)).replace(
                ",", "."
            )
        )
        while p2 > 40:
            p2 /= 10
    except Exception:
        p2 = None
    return (num, nome, str(p1).replace(".", ","), str(p2).replace(".", ","))


def filter_words(annotations: list) -> list:
    return [
        annotation for annotation in annotations if "\n" not in annotation.description
    ]


def center_y(annotation) -> float:
    return sum([v.y for v in annotation.bounding_poly.vertices]) / len(
        annotation.bounding_poly.vertices
    )


def center_x(annotation) -> float:
    return sum([v.x for v in annotation.bounding_poly.vertices]) / len(
        annotation.bounding_poly.vertices
    )


def sort_vertically(annotations: list) -> list:
    return sorted(annotations, key=lambda annotation: center_y(annotation))


def min_y(a):
    return min([v.y for v in a.bounding_poly.vertices])


def max_y(a):
    return max([v.y for v in a.bounding_poly.vertices])


from Levenshtein import distance

prefixes = [
    "TE ",
    "AM ",
    "PRI ",
    "DESC ",
    "LI ",
]


def remove_prefixes(s):
    for prefix in prefixes:
        s = s.replace(prefix, "")
    return s


def str_process(s):
    try:
        return remove_prefixes(unidecode(s.upper()))
    except Exception:
        return ""


def closest_string(s, strings):
    min_s = min(strings, key=lambda x: distance(str_process(s), str_process(x)))
    return min_s, distance(str_process(s), str_process(min_s))


def closest_vaca(vaca, vacas):
    num, nome = vaca
    if nome.strip() == "":
        return ("", "")
    nums, nomes = zip(*vacas)
    distances = map(
        lambda x: (x, distance(str_process(nome), str_process(x[1]))), vacas
    )
    distances = list(sorted(distances, key=lambda d: d[1]))
    min_dist = distances[0][1]
    best_matches = [d[0] for d in distances if d[1] == min_dist]
    best_match = min(best_matches, key=lambda x: distance(str(num), str(x[0])))
    return best_match


# Imports the Google Cloud client library
from google.cloud import vision


def get_dataframe(filepath):
    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # The name of the image file to annotate
    file_name = os.path.abspath(filepath)

    # Loads the image into memory
    with io.open(file_name, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    from collections import defaultdict
    import numpy as np

    response = client.document_text_detection(image=image)

    annotations = filter_words(response.text_annotations)
    annotations = sort_vertically(annotations)

    # create a list of bounding box vertices
    bounding_boxes = [
        np.array([[v.x, v.y] for v in b.bounding_poly.vertices]) for b in annotations
    ]

    heights = [
        abs(
            a.bounding_poly.vertices[0].y
            + a.bounding_poly.vertices[1].y
            - a.bounding_poly.vertices[2].y
            - a.bounding_poly.vertices[3].y
        )
        / 2
        for a in annotations
    ]

    # define a threshold for grouping boxes horizontally
    threshold = round(sum(heights) / len(heights) / 1.2)

    # create a dictionary to store the groups of bounding boxes
    groups = defaultdict(list)

    # iterate through the bounding boxes and group them horizontally
    for i, box in enumerate(bounding_boxes):
        y_coordinates = [point[1] for point in box]
        y_min = min(y_coordinates)
        y_max = max(y_coordinates)
        center = (y_min + y_max) / 2
        found_group = False
        for key in groups.keys():
            if abs(center - key) <= threshold:
                groups[round(key)].append(annotations[i])
                found_group = True
                break
        if not found_group:
            groups[round(center)].append(annotations[i])

    for key in groups.keys():
        groups[key] = sorted(groups[key], key=lambda a: center_x(a))

    rows = []

    for key, group in groups.items():
        row = " ".join([a.description for a in group])
        rows.append(row)

    min_s, _ = closest_string("nª nome manhã tarde obs", rows)

    from copy import copy

    new_rows = []

    for i, (key, group) in enumerate(copy(groups).items()):
        if i < rows.index(min_s) + 1:
            del groups[key]
            continue
        row = " ".join(
            [a.description for a in sorted(group, key=lambda a: center_x(a))]
        )
        new_rows.append(row)
        # print(f'key[{key}]: {row}')

    from sklearn.cluster import KMeans
    import numpy as np

    new_annotations = []

    values = list(groups.values())

    for i in range(len(values)):
        for thing in values[i]:
            new_annotations.append(thing)

    x_mins = [[min([v.x for v in a.bounding_poly.vertices])] for a in new_annotations]

    # Generate sample data
    data = np.array(x_mins)

    # Define the number of clusters
    k = 4

    # Create a KMeans model
    kmeans = KMeans(n_clusters=k)

    # Fit the model to the data
    kmeans.fit(data)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    regions = []

    from PIL import Image

    # get image
    img = Image.open(filepath)

    start = 0
    end = img.width

    for center in sorted(cluster_centers):
        regions.append((start, center[0]))
        start = center[0]

    regions.append((start, end))

    del regions[0]

    from collections import defaultdict
    import numpy as np

    new_annotations = list(reversed(sort_vertically(new_annotations)))

    # create a list of bounding box vertices
    bounding_boxes = [
        np.array([[v.x, v.y] for v in b.bounding_poly.vertices])
        for b in new_annotations
    ]

    widths = [
        abs(
            a.bounding_poly.vertices[0].x
            + a.bounding_poly.vertices[-1].x
            - a.bounding_poly.vertices[2].x
            - a.bounding_poly.vertices[1].x
        )
        / 2
        for a in new_annotations
    ]

    # define a threshold for grouping boxes horizontally
    threshold = round(sum(widths) / len(widths) / 1)

    # create a dictionary to store the groups of bounding boxes
    new_groups = defaultdict(list)

    for i, region in enumerate(regions):
        new_groups[i] = []

    def get_region(center):
        for i, region in enumerate(regions):
            if region[0] < center < region[1]:
                return i

    # iterate through the bounding boxes and group them horizontally
    for i, box in enumerate(bounding_boxes):
        x_coordinates = [point[0] for point in box]
        x_min = min(x_coordinates)
        x_max = max(x_coordinates)
        center = (x_min + x_max) / 2
        found_group = False
        new_groups[get_region(center)].append(new_annotations[i])

    cols = []

    for key, group in new_groups.items():
        col = " ".join(
            [a.description for a in sorted(group, key=lambda a: (center_y(a)))]
        )
        cols.append(col)
        # print(f'key[{key}]: {col}')

    def find_in_groups(obj, groups):
        for i, group in enumerate(groups.values()):
            if obj in group:
                return i

    table = np.empty((len(groups), len(new_groups)), dtype=object)

    for annotation in annotations:
        row = find_in_groups(annotation, groups)
        col = find_in_groups(annotation, new_groups)
        if (row, col) != (None, None):
            if table[row][col] is None:
                table[row][col] = []
            table[row][col] += [annotation]
            table[row][col] = sorted(table[row][col], key=lambda a: center_x(a))

    import csv
    from unidecode import unidecode

    ns = []
    nomes = []
    vacas = []

    with open("rebanho.csv", "r") as file:
        csvreader = csv.reader(file)
        for row in list(csvreader)[1:]:
            pass  # print(row[0], row[1], unidecode(row[1]))
            vacas.append(row)
            ns.append(row[0])
            nomes.append(row[1])

    import pandas as pd

    df = pd.DataFrame(columns=["Nº", "NOME", "AUTO Nº", "AUTO NOME", "MANHÃ", "TARDE"])
    for i, row in enumerate(table):

        (num, nome, p1, p2) = process_row(row)

        auto_num, auto_nome = closest_vaca((num, nome), vacas)

        df.loc[i, :] = (num, nome, auto_num, auto_nome, p1, p2)

    import pandas as pd

    def disimilar(s1, s2):

        s1 = s1.map(lambda s: str_process(s))

        s2 = s2.map(lambda s: str_process(s))

        return s1 != s2

    def match(x):
        name = x.name.replace("AUTO ", "")
        return (disimilar(x, df[name])).map(
            {True: "background-color: yellow; color: black", False: ""}
        )

    def select_col(x):
        red = "background-color: red"
        yellow = "background-color: yellow"
        c2 = ""
        # compare columns
        mask_manha = x["MANHÃ"].map(lambda p: p is None or p > 40 or p <= 0)
        mask_tarde = x["TARDE"].map(lambda p: p is None or p > 40 or p <= 0)

        mask_nums = disimilar(x["Nº"], x["AUTO Nº"])
        mask_nomes = disimilar(x["NOME"], x["AUTO NOME"])

        # DataFrame with same index and columns names as original filled empty strings
        df1 = pd.DataFrame(c2, index=x.index, columns=x.columns)
        # modify values of df1 column by boolean mask
        df1.loc[mask_manha, "MANHÃ"] = yellow
        df1.loc[mask_tarde, "TARDE"] = yellow
        df1.loc[mask_nums, "AUTO Nº"] = yellow
        df1.loc[mask_nomes, "AUTO NOME"] = yellow
        return df1

    return df
    # return df.style.apply(select_col, axis=None)


def fix_headers(df):
    import pandas as pd

    new_df = pd.DataFrame(
        columns="Nome;Nome;Ord. 1;Ord. 2;Ord. 3;Tot.;Data;Responsável;DEL;Dias sec. prev.;Grupo no controle;Observação;".split(
            ";"
        )
    )
    new_df["Nome"] = df["AUTO Nº"] + len(df) * [" "] + df["AUTO NOME"]
    new_df["Ord. 1"] = df["MANHÃ"]
    new_df["Ord. 2"] = df["TARDE"]
    return new_df


def order_vertices_clockwise(polygon_data):
    # extract the vertices from the polygon data
    vertices = polygon_data["regions"][0][:4]

    # find the upper left vertex
    upper_left = min(vertices, key=lambda vertex: vertex["y"] + vertex["x"])
    upper_right = min(vertices, key=lambda vertex: vertex["y"] - vertex["x"])
    lower_right = max(vertices, key=lambda vertex: vertex["y"] + vertex["x"])
    lower_left = max(vertices, key=lambda vertex: vertex["y"] - vertex["x"])

    sorted_vertices = [upper_left, upper_right, lower_right, lower_left]

    return [[p["x"], p["y"]] for p in sorted_vertices]


def get_dewarped_auto(full_path):
    from django.core.files import File
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    image = cv2.imread(full_path)

    from django.core.files import File
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the image
    image = cv2.imread(full_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Canny Edge Detection to find edges
    edges = cv2.Canny(gray, 50, 150)

    # Use Hough Transform to detect lines
    for threshold in range(400, 100, -5):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold, maxLineGap=100)
        if not lines is None and len(lines) > 20:
            break

    # Filter out the small lines
    # lines = [line for line in lines if np.sqrt((line[0][0] - line[0][2]) ** 2 + (line[0][1] - line[0][3]) ** 2) > 50]

    # Create a copy of the image to draw the lines on
    line_image = np.copy(image)

    # Get the endpoints of all the lines
    endpoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints.append([[x1, y1], [x2, y2]])

    # Differentiate between horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    for endpoint in endpoints:
        x1, y1 = endpoint[0]
        x2, y2 = endpoint[1]
        if abs(x2 - x1) > abs(y2 - y1):
            horizontal_lines.append(endpoint)
        else:
            vertical_lines.append(endpoint)

    horizontal_lines_orig = np.copy(horizontal_lines)
    # Consider only extreme lines
    highest_horizontal_line = max(horizontal_lines, key=lambda x: x[0][1])
    lowest_horizontal_line = min(horizontal_lines, key=lambda x: x[0][1])
    leftmost_vertical_line = min(vertical_lines, key=lambda x: x[0][0])
    rightmost_vertical_line = max(vertical_lines, key=lambda x: x[0][0])

    horizontal_lines = [highest_horizontal_line, lowest_horizontal_line]
    vertical_lines = [leftmost_vertical_line, rightmost_vertical_line]

    # Draw the horizontal lines in red
    for endpoint in horizontal_lines:
        x1, y1 = endpoint[0]
        x2, y2 = endpoint[1]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw the vertical lines in green
    for endpoint in vertical_lines:
        x1, y1 = endpoint[0]
        x2, y2 = endpoint[1]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    intersections = []
    # Get intersection points and draw them in blue
    for horizontal_line in horizontal_lines:
        for vertical_line in vertical_lines:
            x1, y1 = horizontal_line[0]
            x2, y2 = horizontal_line[1]
            x3, y3 = vertical_line[0]
            x4, y4 = vertical_line[1]
            x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            )
            y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            )
            intersections.append([x, y])
            cv2.circle(line_image, (int(x), int(y)), 2, (255, 0, 0), 2)

    # Correct order for the transform
    intersections = [
        intersections[0],
        intersections[1],
        intersections[3],
        intersections[2],
    ]
    # mirro the image
    intersections = [
        intersections[0],
        intersections[3],
        intersections[2],
        intersections[1],
    ]
    # rotate the image
    intersections = [
        intersections[1],
        intersections[2],
        intersections[3],
        intersections[0],
    ]
    # # rotate the image
    # intersections = [intersections[1], intersections[2], intersections[3], intersections[0]]
    # # rotate again
    # intersections = [intersections[1], intersections[2], intersections[3], intersections[0]]

    # Perspective transform from intersections to a rectangle
    src = np.array(intersections, dtype="float32")

    # Define the size of the transformed image
    y1, x1, _ = np.shape(image)

    # Define the four corners of the parallelogram after the transform
    dst = np.array([[0, 0], [x1, 0], [x1, y1], [0, y1]], dtype="float32")

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Put image limits in an array
    image_limits = np.array(
        [
            [0, 0],
            [image.shape[1], 0],
            [image.shape[1], image.shape[0]],
            [0, image.shape[0]],
        ],
        dtype="float32",
    )

    # Transform those limits using the perspective transform matrix
    transformed_limits = cv2.perspectiveTransform(image_limits.reshape(-1, 1, 2), M)

    # Get bounding box of the transformed limits as points in an array
    x_min = int(min(transformed_limits[:, 0, 0]))
    x_max = int(max(transformed_limits[:, 0, 0]))
    y_min = int(min(transformed_limits[:, 0, 1]))
    y_max = int(max(transformed_limits[:, 0, 1]))

    bounding_box = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype="float32",
    )

    # Use the inverse perspective transform to get the bounding box in the original image
    bounding_box_orig = cv2.perspectiveTransform(
        bounding_box.reshape(-1, 1, 2), np.linalg.inv(M)
    )

    # Get new perspective transform matrix using your bounding box orig
    src = np.array(bounding_box_orig, dtype="float32")
    dst = np.array([[0, 0], [x1, 0], [x1, y1], [0, y1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)

    # Apply the perspective transform to the image
    dewarped = cv2.warpPerspective(image, M, (x1, y1))

    dewarped = cv2.cvtColor(dewarped, cv2.COLOR_BGR2RGB)
    dewarped = Image.fromarray(dewarped)

    import io

    image_buffer = io.BytesIO()
    dewarped.save(image_buffer, format="JPEG")

    image_file = File(image_buffer)

    return image_file


def get_dewarped_poly_with_lines(full_path, poly):
    from django.core.files import File
    import cv2
    import numpy as np

    from utils.dataframe import get_vertical_lines, add_lines

    image = cv2.imread(full_path)

    y1, x1, _ = np.shape(image)
    dst = np.array([[0, 0], [x1, 0], [x1, y1], [0, y1]], dtype="float32")

    src = order_vertices_clockwise(poly)
    src = np.array(src, dtype="float32")

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Apply the perspective transform to the image
    dewarped = cv2.warpPerspective(image, M, (x1, y1))

    cropped = cv2.cvtColor(dewarped, cv2.COLOR_BGR2RGB)

    vertical_lines = get_vertical_lines(cropped)

    cropped = add_lines(cropped, vertical_lines)

    cropped = Image.fromarray(cropped)

    import io

    image_buffer = io.BytesIO()
    cropped.save(image_buffer, format="JPEG")

    image_file = File(image_buffer)

    return image_file


def get_dewarped_poly(full_path, poly):
    from django.core.files import File
    import cv2
    import numpy as np

    image = cv2.imread(full_path)

    y1, x1, _ = np.shape(image)
    dst = np.array([[0, 0], [x1, 0], [x1, y1], [0, y1]], dtype="float32")

    src = order_vertices_clockwise(poly)
    src = np.array(src, dtype="float32")

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Apply the perspective transform to the image
    dewarped = cv2.warpPerspective(image, M, (x1, y1))

    cropped = cv2.cvtColor(dewarped, cv2.COLOR_BGR2RGB)

    cropped = Image.fromarray(cropped)

    import io

    image_buffer = io.BytesIO()
    cropped.save(image_buffer, format="JPEG")

    image_file = File(image_buffer)

    return image_file


def get_dewarped(full_path, poly=None):
    if poly:
        return get_dewarped_poly(full_path=full_path, poly=poly)
    return get_dewarped_auto(full_path=full_path)


def get_dewarped_with_lines():
    return get_dewarped_poly(full_path=full_pat)


def get_fixed_dataframe(full_path):
    return fix_headers(get_dataframe(full_path))


def get_contour(filein):
    import cv2
    from django.core.files import File

    from django.core.files import File

    img = np.array(Image.open(filein))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, binary) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    img = Image.fromarray(closed)

    import io

    image_buffer = io.BytesIO()
    img.save(image_buffer, format="JPEG")

    image_file = File(image_buffer)

    return image_file
