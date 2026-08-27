"""
Reads an object's TRUE dimensions directly from its SDF file at runtime.
This is what makes the grasp generalize to ANY object, not just one
hardcoded block -- no assumptions, just reading the real geometry.
"""
import xml.etree.ElementTree as ET


def get_object_dimensions(sdf_path):
    """
    Returns a dict describing the object's real shape and size, read
    directly from its SDF geometry tag. Supports box, cylinder, sphere --
    the standard primitive shapes used in grasping research benchmarks.
    """
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    geom = root.find(".//collision/geometry")
    if geom is None:
        raise ValueError(f"No <collision><geometry> found in {sdf_path}")

    box = geom.find("box")
    if box is not None:
        size = [float(v) for v in box.find("size").text.split()]
        return {"type": "box", "width": size[0], "depth": size[1], "height": size[2]}

    cylinder = geom.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.find("radius").text)
        length = float(cylinder.find("length").text)
        return {"type": "cylinder", "radius": radius, "height": length}

    sphere = geom.find("sphere")
    if sphere is not None:
        radius = float(sphere.find("radius").text)
        return {"type": "sphere", "radius": radius}

    raise ValueError(f"Unsupported geometry type in {sdf_path}")


if __name__ == "__main__":
    import sys
    dims = get_object_dimensions(sys.argv[1])
    print(dims)
