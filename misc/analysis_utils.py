def extra_communication_time_lower_bound(comp_time, comm_time):
    """communication is completly parallel to computation """
    if comp_time >= comm_time:
        return 0
    else:
        return comm_time - comp_time


def extra_communication_time_upper_bound(comp_time, comm_time):
    """communication is completly not parallel to computation """
    return comm_time


def upper_utilization_bound(comp_time, comm_time):
    """communication is completly parallel to computation """
    comm_time = extra_communication_time_lower_bound(comp_time, comm_time)
    return comp_time / (comm_time + comp_time)


def lower_utilization_bound(comp_time, comm_time):
    """communication is completly not parallel to computation """
    comm_time = extra_communication_time_upper_bound(comp_time, comm_time)
    return comp_time / (comm_time + comp_time)


def apply_ratio(upper, lower, ratio):
    return (upper * (1 - ratio)) + (lower * ratio)