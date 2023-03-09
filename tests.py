import pytest

SEGMENT_ANGLE_PAIRS = [
    ((0, 0, 1, 0), 0),
    ((0, 0, 1, 1), 45),
    ((0, 0, 0, 1), 90),
]


@pytest.mark.parametrize("segment, angle", SEGMENT_ANGLE_PAIRS)
def test_get_angle(segment, angle):
    from utils.geometry import get_angle

    assert get_angle((segment)) == angle


def test_filter_by_angle():
    from utils.geometry import filter_by_angle

    segments = [
        (0, 0, 1, 0),
        (0.1, 0, 1.1, 0),
        (0.1, 0, 0.9, 0),
        (0, 0, 1, 1),
        (0, 0, 1.1, 1),
        (0, 0, 0.9, 1),
        (0, 0, 0, 1),
        (0.1, 0, 0, 1.1),
        (0.1, 0, 0, 0.9),
    ]

    expected = [
        (0, 0, 0, 1),
        (0.1, 0, 0, 1.1),
        (0.1, 0, 0, 0.9),
    ]

    actual = filter_by_angle(segments, 90)

    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])


intersecting_segments_1 = [(0, 0, 2, 2), (1, 1, 3, 3), (2, 0, 0, 2)]
intersecting_segments_2 = [(0, 0, 2, 0), (1, 0, 1, 2), (0, 1, 2, 1)]
intersecting_segments_3 = [
    (0, 0, 0, 3),
    (0.3, 2, 2, 2),
    (1, 1.5, 2.3, 4),
    (-0.1, 1, 1.8, 2.3),
]


non_intersecting_segments_1 = [
    (0, 0, 0, 3),
    (0.3, 2, 2, 2),
    (2, 2.1, 2, 4),
    (1.8, 3, 0, 4),
]

INTERSECTING_SEGMENTS = [
    intersecting_segments_1,
    intersecting_segments_2,
    intersecting_segments_3,
]


NON_INTERSECTING_SEGMENTS = [
    non_intersecting_segments_1,
]


@pytest.mark.parametrize(
    "segments",
    INTERSECTING_SEGMENTS,
)
def test_intersecting_segments(segments):
    from utils.geometry import group_intersecting_segments

    groups = group_intersecting_segments(segments)

    assert len(groups) == 1
    assert len(groups[0]) == len(segments)


@pytest.mark.parametrize(
    "segments",
    NON_INTERSECTING_SEGMENTS,
)
def test_non_intersecting_segments(segments):
    from utils.geometry import group_intersecting_segments

    groups = group_intersecting_segments(segments)

    assert len(groups) == len(segments)
    assert len(groups[0]) == 1


SEGMENT_BOUNDS_PAIRS = [
    ((1, 1, 2, 2), (0, 0, 4, 4), (0, 0, 4, 4)),
]


@pytest.mark.parametrize("segment, bounds, expected", SEGMENT_BOUNDS_PAIRS)
def test_extend_segment_within_bounds(segment, bounds, expected):
    from utils.geometry import extend_within_bounds

    extended = extend_within_bounds(segment, bounds)[0]

    assert extended == expected
