import numpy as np
import utm


def latlong_to_utm(latlong):
    """
    :param latlong: latlong or list of latlongs
    :return: utm (easting, northing)
    """
    latlong = np.array(latlong)
    if len(latlong.shape) > 1:
        return np.array([latlong_to_utm(p) for p in latlong])

    easting, northing, _, _ = utm.from_latlon(*latlong)
    return np.array([easting, northing])


def utm_to_latlong(u, zone_number=10, zone_letter='S'):
    u = np.array(u)
    if len(u.shape) > 1:
        return np.array([utm_to_latlong(u_i, zone_number=zone_number, zone_letter=zone_letter) for u_i in u])


    easting, northing = u
    return utm.to_latlon(easting, northing, zone_number=zone_number, zone_letter=zone_letter)


def compass_bearing_to_cartesian_angle(compass_bearing):
    return -compass_bearing + 0.5 * np.pi


def bearing(deg_latlong0, deg_latlong1):
    """
    https://www.movable-type.co.uk/scripts/latlong.html

    :param deg_latlong0:
    :param deg_latlong1:
    :return: bearing (radians)
    """
    lat0, long0 = np.deg2rad(deg_latlong0)
    lat1, long1 = np.deg2rad(deg_latlong1)

    y = np.sin(long1 - long0) * np.cos(lat1)
    x = np.cos(lat0) * np.sin(lat1) - np.sin(lat0) * np.cos(lat1) * np.cos(long1 - long0)

    bearing = np.arctan2(y, x)
    return bearing

def gt_pose_bearing(pos_latlong0, pos_latlong1):
    """
    GT version of above

    :param pos_latlong0:
    :param pos_latlong0:
    :return: bearing (radians)
    """

    x, y = np.array(pos_latlong1) - np.array(pos_latlong0)
    bearing = np.arctan2(y, x)
    return bearing