from pathlib import Path
from xml.sax.saxutils import escape


def _link(name, size, mass=0.05, color="0.72 0.74 0.78 1", origin="0 0 0"):
    sx,sy,sz=size
    ix=max(1e-5, mass*(sy*sy+sz*sz)/12)
    iy=max(1e-5, mass*(sx*sx+sz*sz)/12)
    iz=max(1e-5, mass*(sx*sx+sy*sy)/12)
    return f'''\n  <link name="{escape(name)}">\n    <inertial><origin xyz="{origin}"/><mass value="{mass}"/><inertia ixx="{ix}" ixy="0" ixz="0" iyy="{iy}" iyz="0" izz="{iz}"/></inertial>\n    <visual><origin xyz="{origin}"/><geometry><box size="{sx} {sy} {sz}"/></geometry><material name="m"><color rgba="{color}"/></material></visual>\n    <collision><origin xyz="{origin}"/><geometry><box size="{sx} {sy} {sz}"/></geometry></collision>\n  </link>'''


def _joint(name, parent, child, xyz, axis, lower, upper, effort=3.0, velocity=4.0):
    return f'''\n  <joint name="{escape(name)}" type="revolute">\n    <parent link="{escape(parent)}"/><child link="{escape(child)}"/>\n    <origin xyz="{xyz}" rpy="0 0 0"/><axis xyz="{axis}"/>\n    <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="{velocity}"/>\n    <dynamics damping="0.2" friction="0.05"/>\n  </joint>'''


def build_urdf(path="generated/prosthetic_arm.urdf"):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    parts=['<?xml version="1.0"?>','<robot name="prosthetic_arm">']
    parts.append(_link('base',(0.12,0.12,0.08),0.0,color='0.25 0.28 0.32 1'))
    parts.append(_link('upper_arm',(0.08,0.08,0.28),0.25,origin='0 0 0.14'))
    parts.append(_joint('shoulder_yaw','base','upper_arm','0 0 0.04','0 0 1','-1.396','1.396',6,2))
    parts.append(_link('forearm',(0.07,0.07,0.26),0.22,origin='0 0 0.13'))
    parts.append(_joint('elbow_flex','upper_arm','forearm','0 0 0.28','0 1 0','0','2.356',6,2))
    parts.append(_link('wrist_roll_link',(0.065,0.065,0.07),0.08,origin='0 0 0.035'))
    parts.append(_joint('forearm_roll','forearm','wrist_roll_link','0 0 0.26','0 0 1','-1.571','1.571',4,3))
    parts.append(_link('wrist_pitch_link',(0.07,0.07,0.06),0.07,origin='0 0 0.03'))
    parts.append(_joint('wrist_pitch','wrist_roll_link','wrist_pitch_link','0 0 0.07','0 1 0','-0.873','1.047',4,3))
    parts.append(_link('palm',(0.11,0.035,0.12),0.15,origin='0 0 0.06',color='0.62 0.66 0.72 1'))
    parts.append(_joint('wrist_deviation','wrist_pitch_link','palm','0 0 0.06','1 0 0','-0.611','0.524',4,3))

    # Four fingers in parallel. Finger links extend in +Z from the distal palm edge.
    xs={'index':-0.042,'middle':-0.014,'ring':0.014,'little':0.042}
    lengths={'index':(0.052,0.038,0.028),'middle':(0.058,0.041,0.029),'ring':(0.054,0.039,0.028),'little':(0.047,0.034,0.024)}
    for finger,x in xs.items():
        l1,l2,l3=lengths[finger]
        a=f'{finger}_prox'; b=f'{finger}_mid'; c=f'{finger}_dist'
        parts.append(_link(a,(0.018,0.018,l1),0.018,origin=f'0 0 {l1/2}'))
        parts.append(_joint(f'{finger}_mcp','palm',a,f'{x} 0 0.12','1 0 0','0','1.35',1.5,5))
        parts.append(_link(b,(0.016,0.016,l2),0.014,origin=f'0 0 {l2/2}'))
        parts.append(_joint(f'{finger}_pip',a,b,f'0 0 {l1}','1 0 0','0','1.55',1.2,5))
        parts.append(_link(c,(0.014,0.014,l3),0.010,origin=f'0 0 {l3/2}'))
        parts.append(_joint(f'{finger}_dip',b,c,f'0 0 {l2}','1 0 0','0','1.25',1.0,5))

    # Thumb chain attached to side of palm.
    parts.append(_link('thumb_meta',(0.021,0.021,0.045),0.018,origin='0 0 0.0225',color='0.62 0.66 0.72 1'))
    parts.append(_joint('thumb_opposition','palm','thumb_meta','-0.063 0 0.055','0 1 0','-0.35','1.10',1.5,4))
    parts.append(_link('thumb_prox',(0.019,0.019,0.038),0.014,origin='0 0 0.019'))
    parts.append(_joint('thumb_mcp','thumb_meta','thumb_prox','0 0 0.045','1 0 0','0','1.25',1.2,4))
    parts.append(_link('thumb_dist',(0.017,0.017,0.031),0.010,origin='0 0 0.0155'))
    parts.append(_joint('thumb_ip','thumb_prox','thumb_dist','0 0 0.038','1 0 0','0','1.25',1.0,4))
    parts.append('</robot>')
    path.write_text('\n'.join(parts),encoding='utf-8')
    return path

if __name__=='__main__':
    print(build_urdf())
