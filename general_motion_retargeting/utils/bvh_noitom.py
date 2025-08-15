import numpy as np
import bvhio
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils

def load_noitom_file(bvh_file):
    root = bvhio.readAsHierarchy(bvh_file)
    bvh = bvhio.readAsBvh(bvh_file)
    print(f"Root: {root.Name}, Frame count: {bvh.FrameCount}, Frame time: {bvh.FrameTime}")
    bones = []
    parents = np.array([], dtype=int)
    
    joint_names_index = {}
    for joint, index, depth in root.layout():
        bones.append(joint.Name)
        joint_names_index[joint.Name] = index
        # print(f"Joint: {joint.Name}, Index: {index}, Depth: {depth} Parent: {joint.Parent.Name if joint.Parent else 'None'}")
        parents = np.append(parents, joint_names_index[joint.Parent.Name] if joint.Parent else -1)
    
    pos = np.zeros([bvh.FrameCount, len(bones), 3])
    quats = np.zeros([bvh.FrameCount, len(bones), 4])
    for i in range(bvh.FrameCount):
        root.loadPose(i)
        frame_pos = np.array([]).reshape((0, 3))
        frame_quats = np.array([]).reshape((0, 4))
        for joint, index, depth in root.layout():
            # print(f'{joint.PositionWorld} {joint.UpWorld} {joint.Name}')
            frame_pos = np.append(frame_pos, np.array([joint.PositionWorld]), axis=0)
            frame_quats = np.append(frame_quats, np.array([joint.RotationWorld]), axis=0)
            
            # r = R.from_euler('xyz', np.array([joint.UpWorld]), degrees=True)
            # quaternion = r.as_quat()
            # frame_quats = np.append(frame_quats, quaternion, axis=0)
            
        pos[i] = frame_pos
        quats[i] = frame_quats

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    print(pos.shape[0], len(bones))
    for frame in range(pos.shape[0]):
        result = {}
        for i, bone in enumerate(bones):
            orientation = utils.quat_mul(rotation_quat, quats[frame, i])
            position = pos[frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = (position, orientation)
        
        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftFoot"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightFoot"][1])
            
        frames.append(result)
    
    return frames, 1.75  # cm to m

def test_noitom():
    frames, human_height = load_noitom_file("/home/lh/Downloads/take011_chr01.bvh")
    # write to txt file
    with open("noitom_bvh_data.txt", "w") as f:
        f.write(f"Human height: {human_height}\n")
        f.write(str(frames))



if __name__ == "__main__":
    test_noitom()
