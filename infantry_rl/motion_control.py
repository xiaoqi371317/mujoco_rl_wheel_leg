"""Manual velocity/height composition for the moving-maneuver policy."""
def resolve_command(speed=0.,yaw=0.,high=False,spin=False):
    if spin:return 0.,3.,.235
    return -max(-.5,min(2.,float(speed))),max(-1.5,min(1.5,float(yaw))),(.28 if high else .235)
