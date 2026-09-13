"""Build independent sagittally symmetric inertial/kinematic model variants.

The right linkage is the reference, not a claim of CAD ground truth. Meshes are
retained. No existing model or learned-policy input is overwritten.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [('lf0_Link','rf0_Link'),('lf1_Link','rf1_Link'),
         ('l20_Link','r20_Link'),('l21_Link','r21_Link'),
         ('l22_Link','r22_Link'),('l23_Link','r23_Link'),
         ('l_wheel_Link','r_wheel_Link')]
S = np.diag([1.,-1.,1.])

def numbers(value): return np.fromstring(value,sep=' ')
def fmt(value): return ' '.join(f'{x:.15g}' for x in np.asarray(value).flat)
def rotation(node):
    out=np.empty(9)
    mujoco.mju_quat2Mat(out,numbers(node.get('quat','1 0 0 0'))/np.linalg.norm(numbers(node.get('quat','1 0 0 0'))))
    return out.reshape(3,3)

def tensor(node):
    if 'fullinertia' in node.attrib:
        xx,yy,zz,xy,xz,yz=numbers(node.get('fullinertia'))
        return np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
    r=rotation(node)
    return r@np.diag(numbers(node.get('diaginertia')))@r.T

def write_inertia(node,mass,com,I):
    node.attrib.clear()
    node.set('mass',fmt([mass]));node.set('pos',fmt(com))
    node.set('fullinertia',fmt([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))

def build(source):
    root=ET.parse(source).getroot()
    root.set('model',root.get('model')+'_fix')
    # Linux filenames are case-sensitive; preserve the actual source asset.
    for mesh in root.findall('./asset/mesh'):
        if mesh.get('file')=='base_Link_del.STL':mesh.set('file','base_link_del.STL')
    base=root.find('.//body[@name="base_Link_del"]/inertial')
    I=tensor(base);com=numbers(base.get('pos'));com[1]=0.
    write_inertia(base,float(base.get('mass')),com,(I+S@I@S)/2.)
    for left,right in PAIRS:
        lb=root.find(f'.//body[@name="{left}"]');rb=root.find(f'.//body[@name="{right}"]')
        li,ri=lb.find('inertial'),rb.find('inertial')
        mr=float(ri.get('mass')); mass=(float(li.get('mass'))+mr)/2.
        ir=tensor(ri)*mass/mr; cr=numbers(ri.get('pos'))
        # All paired link local frames use the same local reflection y -> -y.
        write_inertia(ri,mass,cr,ir)
        write_inertia(li,mass,S@cr,S@ir@S)
        assert np.allclose(numbers(lb.get('pos','0 0 0')),S@numbers(rb.get('pos','0 0 0')),atol=1e-12)
        assert np.allclose(rotation(lb),S@rotation(rb)@S,atol=1e-12)
    # Sites are geometric points, unlike axial vectors such as hinge axes.
    for site in root.findall('.//site'):
        name=site.get('name','')
        counterpart=name.replace('Left_','Right_',1).replace('left_','right_',1)
        if counterpart!=name:
            other=root.find(f'.//site[@name="{counterpart}"]')
            if other is not None:
                site.set('pos',fmt(S@numbers(other.get('pos','0 0 0'))))
                if other.get('size') is not None:site.set('size',other.get('size'))
    # The original pin sites are in planes separated by 10 mm. Reuse the
    # previously documented axial correction in the standalone source variant.
    for site in root.findall('.//site'):
        if '_front_site' in site.get('name',''):
            p=numbers(site.get('pos'));p[2]=.01;site.set('pos',fmt(p))
    if source.name=='mjmodel_lqr.xml':
        # Offer a closed-chain home pose instead of the inconsistent zero pose.
        home=ET.parse(ROOT/'infantry_rl/robot/xmls/training_scene.xml').find('./keyframe/key')
        ET.SubElement(ET.SubElement(root,'keyframe'),'key',name='stand',qpos=home.get('qpos'))
        ET.SubElement(root,'option',timestep='0.002',integrator='implicitfast',
                      solver='Newton',iterations='60',ls_iterations='30',tolerance='1e-8')
        for eq in root.findall('./equality/connect'):eq.set('solref','0.006 1')
    destination=source.with_name(source.stem+'_fix.xml')
    ET.indent(root,space='  ')
    ET.ElementTree(root).write(destination,encoding='utf-8',xml_declaration=True)
    return destination

def validate(path,source):
    m=mujoco.MjModel.from_xml_path(str(path)); d=mujoco.MjData(m)
    original_root=ET.parse(source).getroot()
    original_mass=sum(float(n.get('mass')) for n in original_root.findall('.//inertial'))
    assert abs(m.body_mass.sum()-original_mass)<1e-10
    principal=m.body_inertia[1:]
    assert (principal>0).all()
    assert (2*principal.max(axis=1)<=principal.sum(axis=1)+1e-12).all()
    pairs=[(m.body(l).id,m.body(r).id) for l,r in PAIRS]
    body_id=m.body('base_Link_del').id
    ib=original_root.find('.//body[@name="base_Link_del"]/inertial')
    fixed_base=ET.parse(path).find('.//body[@name="base_Link_del"]/inertial')
    assert np.allclose(tensor(fixed_base),S@tensor(fixed_base)@S,atol=1e-14)
    assert numbers(fixed_base.get('pos'))[1]==0.
    max_com=max_inertia=max_axis=max_body=0.
    rng=np.random.default_rng(42)
    # Unconstrained mirror configurations verify transforms, not equilibrium.
    for trial in range(21):
        mujoco.mj_resetData(m,d)
        for li,ri in pairs:
            lj,rj=int(m.body_jntadr[li]),int(m.body_jntadr[ri])
            assert m.body_jntnum[li]==m.body_jntnum[ri]==1
            q=0. if trial==0 else rng.uniform(-.3,.3)
            d.qpos[m.jnt_qposadr[rj]]=q;d.qpos[m.jnt_qposadr[lj]]=-q
            assert np.allclose(m.jnt_axis[lj],S@m.jnt_axis[rj])
            if m.jnt_limited[rj]:assert np.allclose(m.jnt_range[lj],-m.jnt_range[rj][::-1])
        mujoco.mj_forward(m,d)
        R=d.xmat[body_id].reshape(3,3);origin=d.xpos[body_id]
        for li,ri in pairs:
            cl=R.T@(d.xipos[li]-origin);cr=R.T@(d.xipos[ri]-origin)
            al=R.T@d.ximat[li].reshape(3,3);ar=R.T@d.ximat[ri].reshape(3,3)
            il=al@np.diag(m.body_inertia[li])@al.T;ir=ar@np.diag(m.body_inertia[ri])@ar.T
            max_com=max(max_com,float(np.max(np.abs(cl-S@cr))))
            max_inertia=max(max_inertia,float(np.max(np.abs(il-S@ir@S))))
            max_body=max(max_body,float(np.max(np.abs(R.T@(d.xpos[li]-origin)-S@R.T@(d.xpos[ri]-origin)))))
            lj,rj=m.body_jntadr[li],m.body_jntadr[ri]
            # Joint velocity changes sign under the mirrored joint mapping.
            max_axis=max(max_axis,float(np.max(np.abs(R.T@d.xaxis[lj]-S@R.T@d.xaxis[rj]))))
    assert max_com<1e-9 and max_inertia<1e-8 and max_axis<1e-9 and max_body<1e-9
    out=dict(file=str(path.relative_to(ROOT)),total_mass_kg=float(m.body_mass.sum()),
        chassis_mirror_symmetric=True,
        max_mirrored_com_error_m=max_com,max_mirrored_inertia_error_kg_m2=max_inertia,
        max_mirrored_body_error_m=max_body,max_mirrored_axis_error=max_axis,
        symmetric_configurations_tested=21,positive_definite=True,triangle_inequality=True)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
        closure=max(np.max(np.abs(d.site_xpos[a]-d.site_xpos[b])) for a,b in zip(m.eq_obj1id,m.eq_obj2id))
        assert closure<1e-6
        out['home_closed_chain_error_m']=float(closure)
        # Short smoke simulation checks numerical/constraint health, not balance.
        for _ in range(50):mujoco.mj_step(m,d)
        assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        out['smoke_steps']=50
    return out

def main():
    sources=[ROOT/'infantry_V4/meshes/mjmodel_lqr.xml',ROOT/'infantry_rl/robot/xmls/training_scene.xml']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    reports=[validate(build(p),p) for p in sources]
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report={'scope':'Symmetric inertias, joint kinematics and closure sites; original STL meshes retained.',
        'reference':'Right-side linkage; pair-average mass; chassis projection about y=0.',
        'originals_unchanged':True,'models':reports}
    out=ROOT/'artifacts/infantry_audit/symmetry_fix_validation.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
