import numpy as np


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
