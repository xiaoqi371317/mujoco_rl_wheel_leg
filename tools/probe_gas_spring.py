"""Create a separate constant-force gas-spring trial and inspect static loads.

No policy training or source-model modification. Force values are hypotheses,
not measured hardware specifications. Supply both tendon controls explicitly.
"""
import json,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

def main():
    source=Path('infantry_rl/robot/xmls/training_scene_fix.xml')
    out=source.with_name('training_scene_fix_gas_trial.xml')
    original=source.read_bytes();root=ET.fromstring(original)
    tendon=ET.SubElement(root,'tendon');actuator=root.find('actuator')
    for side in ['left','right']:
        spatial=ET.SubElement(tendon,'spatial',name=side+'_gas_tendon',width='0.001')
        for link in ['lf1','lf0']:ET.SubElement(spatial,'site',site=side+'_'+link+'_qitanhuang')
        ET.SubElement(actuator,'motor',name=side+'_gas_motor',tendon=side+'_gas_tendon',gear='1',ctrllimited='true',ctrlrange='0 300',forcelimited='true',forcerange='0 300')
    # New actuator slots default to zero: loading the trial alone adds no assistance.
    for key in root.findall('./keyframe/key'):
        if key.get('ctrl'):key.set('ctrl',key.get('ctrl')+' 0 0')
    ET.indent(root);data=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    if out.exists() and out.read_bytes()!=data:raise FileExistsError(out)
    out.write_bytes(data)
    m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m)
    bank=json.loads(Path('infantry_rl/v5_poses.json').read_text())
    assert bank['model_sha256']==hashlib.sha256(original).hexdigest()
    report={'model':str(out),'source_unchanged':True,'hardware_force_calibrated':False,'model_type':'constant-force tendon approximation; no calibrated stroke/friction','samples':[]}
    for pose in bank['poses']:
        mujoco.mj_resetData(m,d);d.qpos[:]=pose['qpos'];mujoco.mj_forward(m,d)
        lengths=d.ten_length.copy()
        assert abs(lengths[0]-lengths[1])<1e-9
        for force in [0.,100.,200.,300.]:
            d.ctrl[:]=0;d.ctrl[-2:]=force;mujoco.mj_fwdActuation(m,d)
            # Isolate spring generalized loads from zero-command leg position servos.
            loads=d.qfrc_actuator.copy();d.ctrl[-2:]=0;mujoco.mj_fwdActuation(m,d);loads-=d.qfrc_actuator
            torques={name:float(loads[m.jnt_dofadr[m.joint(name).id]]) for name in bank['joint_names']}
            pairs=[('lf0_Joint','rf0_Joint'),('lf1_Joint','rf1_Joint'),('l20_Joint','r20_Joint')]
            assert max(abs(torques[l]+torques[r]) for l,r in pairs)<1e-7
            report['samples'].append(dict(pose_height_m=pose['height'],force_per_side_N=force,tendon_lengths_m=lengths.tolist(),spring_joint_torques_Nm=torques))
    assert source.read_bytes()==original
    dest=Path('artifacts/infantry_audit/gas_spring_static_probe.json');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2))
    print('GAS_TRIAL',out,'static configurations',len(report['samples']),flush=True)
    for row in report['samples']:
        if row['force_per_side_N']==300:print(row['pose_height_m'],row['tendon_lengths_m'],row['spring_joint_torques_Nm']['lf1_Joint'],flush=True)

if __name__=='__main__':main()
