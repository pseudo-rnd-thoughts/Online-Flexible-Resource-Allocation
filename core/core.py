
class ResourceAllocationError(Exception):
    pass


def assert_resource_allocation(cond, message=''):
    if not cond:
        raise ResourceAllocationError(message)


def round_float(value: float) -> float:
    return round(value, 4)
