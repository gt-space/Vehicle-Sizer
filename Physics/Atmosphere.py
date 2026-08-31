from ambiance import Atmosphere

def get_atmospheric_pressure(h):
    """Returns standard atmospheric pressure at altitude"""
    return Atmosphere(h).pressure