"""User-facing speed convention for the fast policy; independent of GUI/GPU."""
def resolve_command(speed=0., yaw=0., high=False, spin=False):
    speed=max(-.5,min(1.3,float(speed)))
    yaw=max(-1.5,min(1.5,float(yaw)))
    if spin:return 0.,3.,.235
    height=.28 if high and abs(speed)<.01 and abs(yaw)<.01 else .235
    return -speed,yaw,height


def map_legacy_keyboard(vx,wz,height):
    speed=1.3 if vx>0 else (-.5 if vx<0 else 0.)
    return resolve_command(speed,wz,high=height>.25,spin=abs(wz)>1.5)
