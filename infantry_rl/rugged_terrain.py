"""Stronger, shorter-scale bumps/pits with a level spawn pad and explicit metric bounds."""
from dataclasses import dataclass
import uuid
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter,zoom
from mjlab.terrains.terrain_generator import SubTerrainCfg,TerrainOutput,TerrainGeometry,TerrainGeneratorCfg
from mjlab.terrains import TerrainEntityCfg

@dataclass(kw_only=True)
class RuggedTerrainCfg(SubTerrainCfg):
    kind: str='noise'
    def function(self,difficulty,spec,rng):
        body=spec.body('terrain');sx,sy=self.size
        if self.kind=='flat':
            geom=body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,size=(sx/2,sy/2,.05),pos=(sx/2,sy/2,-.05),
                group=5,friction=(1.,.005,.0001),rgba=(.48,.52,.56,1))
            return TerrainOutput(origin=np.array([sx/2,sy/2,0.]),geometries=[TerrainGeometry(geom=geom)])
        amplitude=.015+.045*difficulty
        nx,ny=round(sx/.04)+1,round(sy/.04)+1
        x=np.linspace(-sx/2,sx/2,nx);y=np.linspace(-sy/2,sy/2,ny)
        xx,yy=np.meshgrid(x,y)
        if self.kind=='noise':
            coarse=rng.uniform(-1,1,(round(sy/.18)+1,round(sx/.18)+1))
            z=zoom(coarse,(ny/coarse.shape[0],nx/coarse.shape[1]),order=1)
            z=gaussian_filter(z,sigma=.55)
            z/=max(abs(z).max(),1e-8)
            # Broaden pits and ridges while keeping height continuous and bounded.
            z=np.tanh(1.8*z)/np.tanh(1.8)
        else:
            phase=rng.uniform(-np.pi,np.pi,3)
            z=(np.sin(xx*2*np.pi/.50+phase[0])*np.cos(yy*2*np.pi/.60+phase[1])
                +.3*np.sin(xx*2*np.pi/.85+yy*3+phase[2]))/1.3
        # Exact zero pad for closed-chain resets, with a smooth transition to roughness.
        pad=np.clip((np.maximum(abs(xx),abs(yy))-.55)/.35,0,1)
        pad=pad*pad*(3-2*pad)
        edge=np.clip(np.minimum(sx/2-abs(xx),sy/2-abs(yy))/.3,0,1)
        z=np.clip(z*pad*edge,-1,1)*amplitude
        key=uuid.uuid4().hex
        zmin,zmax=float(z.min()),float(z.max())
        field=spec.add_hfield(name='rugged_'+key,size=(sx/2,sy/2,zmax-zmin,.03),
            nrow=ny,ncol=nx,userdata=((z-zmin)/(zmax-zmin)).astype(np.float32).ravel().tolist())
        geom=body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD,hfieldname=field.name,
            pos=(sx/2,sy/2,zmin),group=5,friction=(1.,.005,.0001),rgba=(.45,.50,.40,1))
        return TerrainOutput(origin=np.array([sx/2,sy/2,0.]),geometries=[TerrainGeometry(geom=geom,hfield=field)])

def terrain_cfg(seed=43):
    return TerrainEntityCfg(terrain_type='generator',max_init_terrain_level=0,textures=(),materials=(),
        terrain_generator=TerrainGeneratorCfg(seed=seed,curriculum=True,size=(28.,6.),num_rows=3,num_cols=3,
            color_scheme='none',sub_terrains={
                'flat':RuggedTerrainCfg(kind='flat',proportion=.30),
                'bumps_pits':RuggedTerrainCfg(kind='noise',proportion=.50),
                'ripples':RuggedTerrainCfg(kind='wave',proportion=.20)}))
