def convert_to_lat(lat_str):
    lat_parts = lat_str.split(':')
    lat_deg, lat_min = lat_parts[0].split(' ')
    lat_sec = float(lat_parts[1])
    lat_deg = float(lat_deg)
    lat_min = float(lat_min)
    lat_decimal = lat_deg + (lat_min / 60.0) + (lat_sec / 3600.0)
    return lat_decimal if lat_deg >= 0 else -lat_decimal

def convert_to_long(long_str):
    long_parts = long_str.split(':')
    long_deg, long_min = long_parts[0].split(' ')
    long_sec = -float(long_parts[1])
    long_deg = -float(long_deg)
    long_min = -float(long_min)
    lat_decimal = long_deg + (long_min / 60.0) + (long_sec / 3600.0)
    return lat_decimal
