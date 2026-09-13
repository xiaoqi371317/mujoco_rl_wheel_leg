"""Report supplied body inertias in a common frame; never alter physics."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from infantry_rl.mjlab_task import MODEL

def audit():
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
    base=m.body('base_Link_del').id
    R=d.xmat[base].reshape(3,3)
    reflection=np.diag([1.,-1.,1.])
    def props(name):
        i=m.body(name).id
        axes=R.T@d.ximat[i].reshape(3,3)
        return m.body_mass[i],R.T@(d.xipos[i]-d.xpos[base]),axes@np.diag(m.body_inertia[i])@axes.T
    mass,com,I=props('base_Link_del')
    eig,axes=np.linalg.eigh(I)
    report={'base':{'mass_kg':float(mass),'com_m':com.tolist(),'inertia_kg_m2':I.tolist(),
        'principal_moments':eig.tolist(), 'closest_principal_axis_to_lateral_deg':float(np.degrees(np.arccos(np.max(np.abs(axes[1]))))),
        'sagittal_reflection_relative_tensor_error':float(np.linalg.norm(I-reflection@I@reflection)/np.linalg.norm(I))},'pairs':{}}
    for left,right in [('lf0_Link','rf0_Link'),('lf1_Link','rf1_Link'),('l20_Link','r20_Link'),('l21_Link','r21_Link'),('l22_Link','r22_Link'),('l23_Link','r23_Link'),('l_wheel_Link','r_wheel_Link')]:
        ml,cl,il=props(left);mr,cr,ir=props(right)
        report['pairs'][left+'/'+right]={'mass_left_kg':float(ml),'mass_right_kg':float(mr),
            'mirrored_com_difference_mm':float(np.linalg.norm(cl-reflection@cr)*1000),
            'mirrored_tensor_relative_error':float(np.linalg.norm(il-reflection@ir@reflection)/max(np.linalg.norm(il),np.linalg.norm(ir)))}
    source=Path('infantry_V4/meshes/mjmodel_lqr.xml')
    original=ET.parse(source).find('.//body[@name="base_Link_del"]/inertial')
    current=ET.parse(MODEL).find('.//body[@name="base_Link_del"]/inertial')
    report['base_inertial_matches_original']=original.attrib==current.attrib
    return report

if __name__=='__main__':
    report=audit();dest=Path('artifacts/infantry_audit/inertia_symmetry.json')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
