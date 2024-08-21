import io
from collections.abc import Generator
from enum import Enum

import numpy as np
import pandas as pd
from google.cloud import vision
from Levenshtein import distance
from PIL import Image, ImageDraw
from unidecode import unidecode

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
    if type(segments) is list:
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


def has_alnum(s):
    for i in s:
        if i.isalnum():
            return True
    return False


def _process_row(row):
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
    return (num, nome, p1, p2)


def process_row(row):
    try:
        num = " ".join(r.description for r in row[0] if has_alnum(r.description))
        num = num.upper()

    except Exception:
        num = None
    try:
        nome = " ".join(r.description for r in row[1] if has_alnum(r.description))
    except Exception:
        nome = ""

    try:
        p1 = "".join(r.description for r in row[2] if has_alnum(r.description))
    except Exception:
        p1 = None
    try:
        p2 = "".join(r.description for r in row[3] if has_alnum(r.description))
    except Exception:
        p2 = None

    return (num, nome, p1, p2)


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


def get_closest_vaca_tuple(vaca, vacas) -> tuple[str, str, str]:
    num, nome = vaca
    distances = map(
        lambda v: (
            v,
            distance(
                str(num) + " " + str_process(nome), str(v[0]) + " " + str_process(v[1])
            ),
        ),
        vacas,
    )
    distances = list(sorted(distances, key=lambda d: d[1]))
    return distances[0][0]


def fix_peso(peso):
    ascii_to_number = {
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "!": "1",
        '"': "1",
        "#": "4",
        "$": "5",
        "%": "7",
        "&": "8",
        "'": ".",
        "(": "1",
        ")": "1",
        "*": "",
        "+": "",
        ",": ".",
        "-": ".",
        ".": ".",
        "/": "1",
        ":": "",
        ";": "",
        "<": "",
        "=": "",
        ">": "",
        "?": "2",
        "@": "",
        "A": "4",
        "B": "3",
        "C": "0",
        "D": "0",
        "E": "3",
        "F": "",
        "G": "6",
        "H": "",
        "I": "1",
        "J": "",
        "K": "",
        "L": "1",
        "M": "",
        "N": "",
        "O": "0",
        "P": "9",
        "Q": "0",
        "R": "",
        "S": "5",
        "T": "7",
        "U": "",
        "V": "",
        "W": "",
        "X": "",
        "Y": "",
        "Z": "2",
    }
    auto_peso = "".join(ascii_to_number.get(c, "") for c in peso.upper())

    try:
        return float(auto_peso)
    except Exception:
        return None


def disimilar(s1, s2):
    s1 = s1.map(lambda s: str_process(s))

    s2 = s2.map(lambda s: str_process(s))

    return s1 != s2


def select_col(x):
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


def get_table(filepath):
    # Imports the Google Cloud client library
    from google.cloud import vision

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open(filepath, "rb") as image_file:
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

    import numpy as np
    from sklearn.cluster import KMeans

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

    return table


def get_top_four(lista):
    sorted_list = sorted(lista, reverse=True)
    top_four = sorted_list[:4]
    return [i for i in range(len(lista)) if lista[i] in top_four]


def get_mean(row, size):
    mean_dist = []
    for r in row:
        v = r.bounding_poly.vertices
        x = sum(v.x for v in sorted(v, key=lambda v: v.x)[:size]) / size
        mean_dist.append(x)
    return mean_dist


def process_table(table) -> Generator[list[str], None, None]:
    new_table = [[x for c in row if c for x in c if x] for row in table]

    left_dist = []
    for row in new_table:
        left_dist += get_mean(row, size=2)

    (n, bins) = np.histogram(left_dist, bins=20)

    limits = [bins[i] for i in get_top_four(n)]

    for row in new_table:
        mean_dists = get_mean(row, size=4)
        res = [""] * 4
        indexes = [
            np.argmax(
                [int(mean < limits[1])]
                + [int(limits[i - 1] < mean < limits[i]) for i in range(2, len(limits))]
                + [int(mean > limits[-1])]
            )
            for mean in mean_dists
        ]
        for index, r in zip(indexes, row):
            res[index % len(res)] += r.description + " "
        for i in range(len(res)):
            res[i] = res[i].strip()
        yield res


def get_vertical_lines(img):
    from functools import reduce

    import cv2
    import numpy as np

    from utils import geometry

    def add_lines(l1, l2):
        a = np.array([l1, l2])
        return tuple(np.average(a, axis=0))

    Segment = tuple[float, float, float, float]

    def vertically_close(s1: Segment, s2: Segment, threshold: int = 5) -> bool:
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

    height, width, _ = img.shape

    bounds = (0, 0, width, height)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_size = (1, 45)

    # Perform vertical blurring using the GaussianBlur function
    blurred_img = cv2.GaussianBlur(gray, kernel_size, sigmaX=0, sigmaY=0)

    kernel = (9, 9)

    edges = cv2.Canny(blurred_img, 200, 200, apertureSize=3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=50)
    edges = cv2.dilate(edges, kernel, iterations=50)

    length_threshold = round(0.3 * height)

    lines = [
        line[0]
        for line in cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 720,
            threshold=length_threshold,
            minLineLength=100,
            maxLineGap=10,
        )
    ]
    lines = geometry.filter_by_angle(lines, 90, 30)

    extended_lines = geometry.extend_within_bounds(lines, bounds=bounds)

    groups = group_vertically(extended_lines)

    vertical_lines = [reduce(add_lines, group) for group in groups]

    return vertical_lines


def add_lines(img, lines):
    from copy import copy

    import cv2

    img_cp = copy(img)

    for line in lines:
        x1, y1, x2, y2 = [round(x) for x in line]
        cv2.line(img_cp, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return img_cp
