import pytest

from samgo.instructions import name, steps, write
from tests.test_technic import car


def test_every_part_appears_once_bottom_up_with_axles_then_wheels_last():
    parts = car()
    plan = steps(parts)
    flat = [p for step in plan for p in step]
    assert len(flat) == len(parts) and all(any(p is q for q in flat) for p in parts)
    solid_z = [p["z"] for step in plan[:-2] for p in step]
    assert solid_z == sorted(solid_z)
    assert {p["type"] for p in plan[-2]} == {"axle"} and {p["type"] for p in plan[-1]} == {"wheel"}


def test_part_names_read_like_a_parts_list():
    assert name({"type": "plate", "w": 8, "l": 2, "color": "light_gray"}) == "2x8 plate, light gray"
    assert name({"type": "axle", "length": 6}) == "Technic axle 6"


def current_car():
    parts = car()
    for p in parts:
        if p['type'] == 'wheel':
            p.pop('width')
            p['size'] = '30.4x14'
        elif p['type'] == 'axle':
            p['y'], p['length'] = -2, 8
    parts += [{'type': 'bush', 'axis': 'y', 'x': x, 'y': y, 'z': 0}
              for x in (1, 7) for y in (-2, 5)]
    return parts


def test_current_parts_are_all_present_once_in_instructions():
    parts = current_car()
    parts.append({'type': 'half_bush', 'axis': 'y', 'x': 1, 'y': -2.5, 'z': 0})
    flat = [p for group in steps(parts) for p in group]
    assert sorted(map(id, flat)) == sorted(map(id, parts))


@pytest.mark.parametrize('size', ['30.4x14', '43.2x22', '56x28', '62.4x20', '68.8x36', '81.6x38'])
def test_names_use_actual_wheel_size(size):
    assert name({'type': 'wheel', 'size': size}) == f"wheel {size.replace('x', ' x ')} with tire"


def test_bushes_and_stop_axles_have_accurate_names():
    assert name({'type': 'bush'}) == 'Technic bush (3713)'
    assert name({'type': 'half_bush'}) == 'Technic half bush (32123a)'
    assert name({'type': 'axle', 'length': 4, 'stop': True}) == 'Technic stop axle 4'


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_current_car_renders_complete_instructions(tmp_path, axis):
    parts = current_car()
    if axis == 'x':
        parts = [{**p, 'x': p['y'], 'y': p['x'],
                  **({'w': p['l'], 'l': p['w']} if 'w' in p else {}),
                  **({'axis': 'x'} if 'axis' in p else {})} for p in parts]
    path = write(parts, tmp_path)
    text = path.read_text()
    assert '4 × Technic bush (3713)' in text
    assert '4 × wheel 30.4 x 14 with tire' in text
    assert (tmp_path / 'finished.png').stat().st_size > 1000
    assert len(list(tmp_path.glob('step_*.png'))) == len(steps(parts))


def test_unknown_part_is_rejected_before_writing_any_output(tmp_path):
    parts = car() + [{'type': 'mystery_part'}]
    with pytest.raises(ValueError, match='unsupported'):
        write(parts, tmp_path / 'instructions')
    assert not (tmp_path / 'instructions').exists()


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_large_offset_wheel_fits_the_render_bounds(axis):
    from samgo.instructions import _bounds

    p = {'type': 'wheel', 'axis': axis, 'x': 10, 'y': 20, 'z': 10, 'size': '81.6x38'}
    bounds = _bounds([p])
    # Catalog envelope: diameter 81.6 mm; 5 studs along the axle; center height 37.6 mm.
    expected = ((80, 120), (119.2, 200.8), (-3.2, 78.4)) if axis == 'x' else (
        (39.2, 120.8), (160, 200), (-3.2, 78.4))
    for (low, high), (want_low, want_high) in zip(bounds, expected):
        assert low < want_low and high > want_high
