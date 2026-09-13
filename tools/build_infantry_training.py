"""Build the training scene from the untouched imported LQR MJCF.

The imported planar linkages have closure sites separated along their hinge
axes by 10 mm. Move front sites along those axes to the rear pin planes; keep
body geometry/inertia intact. These simulation assumptions need CAD validation.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "infantry_rl/robot/xmls/training_scene.xml"
DRIVES = ("lf0_Joint", "l20_Joint", "rf0_Joint", "r20_Joint")
WHEELS = ("l_wheel_Joint", "r_wheel_Joint")


def build():
    root = ET.parse(ROOT / "infantry_V4/meshes/mjmodel_lqr.xml").getroot()
    root.set("model", "infantry_balance_v1")
    root.find("compiler").set("meshdir", "../../../infantry_V4/meshes")
    root.find("compiler").set("autolimits", "true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast",
                  solver="Newton", iterations="60", tolerance="1e-8")
    asset = root.find("asset")
    for node in list(asset):
        if node.tag == "hfield":
            asset.remove(node)
    world = root.find("worldbody")
    for node in list(world):
        if node.tag != "body" and node.get("name") != "floor":
            world.remove(node)
    floor = world.find("geom")
    floor.set("friction", "1.0 0.005 0.0001")
    body = world.find("body")
    body.set("pos", "0 0 0.24")
    body.set("quat", "1 0 0 0")
    for site in body.iter("site"):
        if "_front_site" in site.get("name", ""):
            pos = site.get("pos").split()
            pos[2] = "0.01"
            site.set("pos", " ".join(pos))
    for joint in body.iter("joint"):
        name = joint.get("name")
        joint.attrib.pop("ref", None)
        joint.attrib.pop("frictionloss", None)
        joint.set("armature", "0.005" if name in WHEELS else "0.01")
        joint.set("damping", "0.02" if name in WHEELS else "0.2")
        joint.set("actuatorfrcrange", "-30 30")
    for b in body.iter("body"):
        if b.get("name") in ("l_wheel_Link", "r_wheel_Link"):
            for g in b.findall("geom"):
                g.set("contype", "0")
                g.set("conaffinity", "0")
            ET.SubElement(b, "geom", name=b.get("name") + "_collision", type="cylinder",
                          size="0.06 0.03075", pos="0 0 0.00765", rgba="0.13 0.14 0.16 1",
                          friction="1.0 0.005 0.0001", condim="3")
    for tag in ("actuator", "tendon", "sensor", "keyframe"):
        for node in list(root.findall(tag)):
            root.remove(node)
    act = ET.SubElement(root, "actuator")
    for name in DRIVES:
        ET.SubElement(act, "position", name="act_" + name, joint=name, kp="240", kv="8",
                      forcerange="-30 30")
    for name in WHEELS:
        ET.SubElement(act, "motor", name="act_" + name, joint=name,
                      gear="1", ctrlrange="-12 12", forcerange="-12 12")
    for eq in root.find("equality"):
        eq.set("solref", "0.006 1")
        eq.set("solimp", "0.99 0.99 0.01 0.5 2")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true")
    ET.SubElement(root, "statistic", center="0 0 0.25", extent="1.2")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="utf-8", xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(OUT))
    d = mujoco.MjData(m)
    joints = [j for j in range(1, m.njnt) if m.joint(j).name not in WHEELS]
    qadr = m.jnt_qposadr[joints]
    wheel_ids = [m.body(n).id for n in ("l_wheel_Link", "r_wheel_Link")]
    lower = np.full(len(joints), -3.1)
    upper = np.full(len(joints), 3.1)
    for i, j in enumerate(joints):
        if m.jnt_limited[j]:
            lower[i] = max(lower[i], m.jnt_range[j, 0] + .02)
            upper[i] = min(upper[i], m.jnt_range[j, 1] - .02)
    initial = np.array([-.7, .05, .7, -.05, .6, -.6, .5, -.6, -.6, .6, -.5, .6])

    def residual(x):
        d.qpos[qadr] = x
        mujoco.mj_forward(m, d)
        closure = np.concatenate([d.site_xpos[a] - d.site_xpos[b]
                                  for a, b in zip(m.eq_obj1id, m.eq_obj2id)])
        # Wheels underneath subtree CoM, 2 mm above ground to avoid penetration.
        target = np.concatenate([[d.xpos[w, 0] - d.subtree_com[1, 0], d.xpos[w, 2] - .062]
                                 for w in wheel_ids])
        return np.r_[closure * 20, target * 10]

    sol = least_squares(residual, initial, bounds=(lower, upper), max_nfev=2000,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    residual(sol.x)
    assert np.max(np.abs(residual(sol.x))) < 1e-4, sol.message
    ctrl = [d.qpos[m.jnt_qposadr[m.joint(n).id]] for n in DRIVES] + [0., 0.]
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="stand",
                  qpos=" ".join(f"{x:.10f}" for x in d.qpos),
                  ctrl=" ".join(f"{x:.10f}" for x in ctrl))
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="utf-8", xml_declaration=True)
    report = dict(source="infantry_V4/meshes/mjmodel_lqr.xml", training_model=str(OUT),
                  mass_kg=float(m.body_mass.sum()), home_qpos=d.qpos.tolist(), home_ctrl=ctrl,
                  solve_max_weighted_residual=float(np.max(np.abs(residual(sol.x)))),
                  assumptions=["Front closure sites shifted 10 mm along pin axis",
                               "Wheel cylinders fitted to imported STL bounds; axial center 7.65 mm",
                               "No tendon force model: no identified pneumatic spring parameters",
                               "Leg force limit 30 Nm; wheel policy limit 12 Nm",
                               "All original non-wheel collision meshes remain enabled"])
    OUT.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    build()
