"""Import only the original V4 environment, keeping the trained fix robot intact."""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg,TerrainOutput,TerrainGeometry,TerrainGeneratorCfg

COURSES={'v4':'mjmodel_lqr_fix.xml','v4-steps':'mjmodel.xml'}


@dataclass(kw_only=True)
class V4CourseCfg(SubTerrainCfg):
    course: str='v4'
    training_friction: bool=True

    def function(self,difficulty,spec,rng):
        path=Path(__file__).resolve().parents[1]/'infantry_V4/meshes'/COURSES[self.course]
        root=ET.parse(path).getroot();body=spec.body('terrain');sx,sy=self.size
        fields={};geometries=[]
        for node in root.findall('worldbody/geom'):
            kind=node.get('type');name=node.get('name')
            kwargs=dict(name='course_'+name,group=5)
            for attr in ['pos','size','euler','quat','rgba','friction']:
                if node.get(attr):kwargs[attr]=[float(x) for x in node.get(attr).split()]
            for attr in ['contype','conaffinity','condim']:
                if node.get(attr):kwargs[attr]=int(node.get(attr))
            if self.training_friction:kwargs.update(friction=(1.,.005,.0001),condim=3)
            if 'euler' in kwargs:
                angles=np.array(kwargs.pop('euler'));quat=np.zeros(4)
                if root.find('compiler').get('angle','degree')=='degree':angles=np.deg2rad(angles)
                mujoco.mju_euler2Quat(quat,angles,'xyz');kwargs['quat']=quat
            pos=kwargs.get('pos',[0.,0.,0.]);pos[0]+=sx/2;pos[1]+=sy/2;kwargs['pos']=pos
            field=None
            if kind=='plane':
                # A finite floor covers this test arena; obstacle positions/heights are original.
                kwargs.update(type=mujoco.mjtGeom.mjGEOM_BOX,size=(sx/2,sy/2,.05),pos=(sx/2,sy/2,-.05),rgba=(.5,.55,.6,1.))
            elif kind=='box':kwargs['type']=mujoco.mjtGeom.mjGEOM_BOX
            elif kind=='hfield':
                ref=node.get('hfield')
                if ref not in fields:
                    asset=root.find(f"asset/hfield[@name='{ref}']")
                    fields[ref]=spec.add_hfield(name='course_'+ref,file=str(path.parent/asset.get('file')),
                        size=[float(x) for x in asset.get('size').split()])
                field=fields[ref];kwargs.update(type=mujoco.mjtGeom.mjGEOM_HFIELD,hfieldname=field.name)
            else:raise ValueError(f'Unsupported course geometry: {kind}')
            geom=body.add_geom(**kwargs)
            geometries.append(TerrainGeometry(geom=geom,hfield=field))
        return TerrainOutput(origin=np.array([sx/2,sy/2,0.]),geometries=geometries)


def terrain_cfg(course='v4',training_friction=True):
    return TerrainEntityCfg(terrain_type='generator',textures=(),materials=(),max_init_terrain_level=0,
        terrain_generator=TerrainGeneratorCfg(seed=43,curriculum=True,size=(40.,24.),num_rows=1,
            color_scheme='none',sub_terrains={'course':V4CourseCfg(course=course,training_friction=training_friction)}))
