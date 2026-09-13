"""CPU geometry-only verification: actual ray heights, no policy training."""
import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab.terrains import TerrainEntity
from infantry_rl.rough_terrain import terrain_cfg

def main():
    cfg=terrain_cfg();cfg.num_envs=12
    terrain=TerrainEntity(cfg,device='cpu');m=terrain.spec.compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    heights=[];origin_errors=[];rng=np.random.default_rng(9)
    for origin in terrain.terrain_origins.reshape(-1,3).cpu().numpy():
        for _ in range(100):
            xy=origin[:2]+rng.uniform([-10,-2],[10,2]);point=np.r_[xy,1.]
            hit=mujoco.mj_ray(m,d,point,np.array([0.,0.,-1.]),np.array([0,0,0,0,0,1],dtype=np.uint8),True,-1,None)
            assert hit>=0;heights.append(1-hit)
        hit=mujoco.mj_ray(m,d,np.r_[origin[:2],1.],np.array([0.,0.,-1.]),None,True,-1,None)
        origin_errors.append(abs(1-hit-origin[2]))
    report=dict(hfields=m.nhfield,geoms=m.ngeom,sampled_height_min_m=min(heights),sampled_height_max_m=max(heights),
        spawn_height_error_m=float(max(origin_errors)),sample_count=len(heights))
    assert max(origin_errors)<1e-6 and min(heights)>=-.015001 and max(heights)<=.015001
    path=Path('artifacts/infantry_audit/rough_geometry.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
