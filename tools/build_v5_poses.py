"""Generate feasible closed-linkage crouches; never reset joints independently."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from infantry_rl.model_selection import select_model
from infantry_rl.mjlab_task import DRIVES, WHEELS


def main():
    path = select_model('fix')
    m = mujoco.MjModel.from_xml_path(str(path)); d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    home = d.qpos.copy()
    ids = [j for j in range(1, m.njnt) if m.joint(j).name not in WHEELS]
    adr = m.jnt_qposadr[ids]
    lo = [m.jnt_range[j, 0]+.005 if m.jnt_limited[j] else -3.1 for j in ids]
    hi = [m.jnt_range[j, 1]-.005 if m.jnt_limited[j] else 3.1 for j in ids]
    rows = []
    for h in [.24, .22, .20, .18, .16, .14, .26, .28, .30, .32]:
        d.qpos[:] = home; d.qpos[2] = h
        def residual(x):
            d.qpos[adr] = x; mujoco.mj_forward(m,d)
            loop = np.concatenate([d.site_xpos[a]-d.site_xpos[b] for a,b in zip(m.eq_obj1id,m.eq_obj2id)])
            wheel = np.concatenate([[d.xpos[m.body(n).id,0]-d.subtree_com[1,0], d.xpos[m.body(n).id,2]-.0602] for n in ('l_wheel_Link','r_wheel_Link')])
            return np.r_[20*loop,10*wheel]
        sol=least_squares(residual,home[adr],bounds=(lo,hi),max_nfev=1200,gtol=1e-11,xtol=1e-11,ftol=1e-11)
        err=float(abs(residual(sol.x)).max())
        contacts=[float(d.contact[i].dist) for i in range(d.ncon)]
        print(h,err,min(contacts,default=0),flush=True)
        if err>1e-5 or min(contacts,default=0)<-.001:continue
        rows.append(dict(height=h,qpos=d.qpos.tolist(),drive={n:float(d.qpos[m.jnt_qposadr[m.joint(n).id]]) for n in DRIVES},closure_weighted_error=err))
    out=Path('infantry_rl/v5_poses.json')
    out.write_text(json.dumps(dict(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),joint_names=[m.joint(j).name for j in range(1,m.njnt)],poses=rows),indent=2))
    assert any(r['height']==.16 for r in rows)
    assert any(r['height']==.30 for r in rows)

if __name__=='__main__':main()
