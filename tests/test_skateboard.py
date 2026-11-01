from samgo.skateboard import STEER, build_skateboard, steer


def ok(v):
    return v if isinstance(v, bool) else v[0]


def test_default_board_passes_every_check():
    r = build_skateboard()
    assert all(ok(v) for v in r["checks"].values()), r["checks"]
    for k in ("structure", "connected", "stands", "wheels_spin", "rolls"):
        assert k in r["checks"]
    assert len(r["parts"]) > 40                                   # the 6 x 30 board with kick tails


def test_other_sizes_pass_too():
    for length, width in ((16, 4), (24, 6)):
        assert all(ok(v) for v in build_skateboard(length=length, width=width)["checks"].values())


def test_rigid_trucks_do_not_claim_to_steer():
    r = build_skateboard()
    assert any("steering" in u for u in r["unsupported"])
    assert not steer(r["parts"], STEER)[0]


def test_a_lifted_wheel_is_caught():
    r = build_skateboard()
    parts = [dict(p) for p in r["parts"]]
    i = next(i for i, p in enumerate(parts) if p["type"] == "wheel")
    parts[i]["z"] += 1
    from samgo.technic import verify_function
    assert not verify_function(parts).passed
