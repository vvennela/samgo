from samgo.build_kit import census, wanted_xml


def test_census_counts_parts_by_colour_and_writes_a_bricklink_list():
    ldr = "0 x\n1 72 0 0 0 1 0 0 0 1 0 0 0 1 3024.dat\n1 72 8 0 0 1 0 0 0 1 0 0 0 1 3024.dat\n1 0 0 0 0 1 0 0 0 1 0 0 0 1 3710.dat\n"
    bom = census(ldr)
    assert bom == {("3024", 72): 2, ("3710", 0): 1}
    xml = wanted_xml(bom)
    assert "<ITEMID>3024</ITEMID><COLOR>85</COLOR><MINQTY>2</MINQTY>" in xml and "<COLOR>11</COLOR>" in xml
