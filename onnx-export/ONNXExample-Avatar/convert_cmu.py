import numpy as np
import json
import os
from tqdm import tqdm  # 如果没有安装 tqdm，可以去掉这行和下面的 tqdm 包装
import argparse



def save_amass_to_json(npz_path, output_dir, max_frame_num, frame_stride):
    # 1. 加载数据
    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} not found.")
        return

    amass = np.load(npz_path)
    
    # 提取数据
    trans = amass['trans']   # (N, 3)
    betas = amass['betas']   # (16,) 通常 AMASS 给 16 个，但标准 SMPL 常用前 10 个
    poses = amass['poses']   # (N, 156)

    # 2. 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {poses.shape[0]} frames...")

    # 3. 静态参数 (来自你的示例 JSON)
    # AMASS 数据集本身不包含相机内参，这里硬编码为你提供的示例值
    static_camera_params = {
        "focal": [1177.7945556640625, 1177.7945556640625],
        "princpt": [680.0, 384.0],
        "img_size_wh": [1360, 768],
        "pad_ratio": 0.0
    }

    # 4. 逐帧处理
    num_frames = poses.shape[0]

    count = 0
    
    # 使用 tqdm 显示进度条，如果不需要可以改成: range(num_frames)
    for i in tqdm(range(0, max_frame_num * frame_stride, frame_stride)):

        if i >= poses.shape[0]:
            break
        
        # --- 切片与重塑逻辑 (对应 156 维的结构) ---
        
        # 1. Root Pose (0-3) -> [3]
        curr_root = poses[i, :3]
        
        # 2. Body Pose (3-66) -> 63维 -> 重塑为 [21, 3]
        curr_body = poses[i, 3:66].reshape(-1, 3)
        
        # 3. Hand Poses
        # Left Hand (66-111) -> 45维 -> 重塑为 [15, 3]
        curr_lhand = poses[i, 66:111].reshape(-1, 3)
        # Right Hand (111-156) -> 45维 -> 重塑为 [15, 3]
        curr_rhand = poses[i, 111:156].reshape(-1, 3)
        
        # 4. 缺失的面部数据 (填充 0)
        # 因为输入只有 156 维，无法恢复原始的下巴和眼球动作
        jaw_pose = [0.0, 0.0, 0.0]
        leye_pose = [0.0, 0.0, 0.0]
        reye_pose = [0.0, 0.0, 0.0]

        # --- 构建字典 ---
        frame_data = {
            # Betas 通常每一帧都一样，取前10维以匹配你的示例
            "betas": betas[:10].tolist(), 
            "root_pose": curr_root.tolist(),
            "body_pose": curr_body.tolist(),
            "jaw_pose": jaw_pose,
            "leye_pose": leye_pose,
            "reye_pose": reye_pose,
            "lhand_pose": curr_lhand.tolist(),
            "rhand_pose": curr_rhand.tolist(),
            "trans": trans[i].tolist(),
            # 合并相机参数
            **static_camera_params
        }

        # --- 保存 JSON ---
        # 文件名格式: frame_0000.json
        file_name = f"{count:05d}.json"
        save_path = os.path.join(output_dir, file_name)
        
        with open(save_path, 'w') as f:
            json.dump(frame_data, f) # 如果想为了调试看清楚结构，可以加 indent=4

        count += 1

    print(f"Done! Saved {count} JSON files to '{output_dir}'")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--max_frame_num", type=int, required=True)
    parser.add_argument("--frame_stride", type=int, required=True)
    args = parser.parse_args()

    # --- 运行代码 ---
    # 修改为你的实际路径
    npz_file_path = args.npz_path
    output_directory = args.output_directory
    max_frame_num = args.max_frame_num
    frame_stride = args.frame_stride

    # 确保你在运行前修改了 npz_file_path，或者保证该文件存在
    # 为了演示，这里注释掉实际运行，你需要取消注释来运行它
    save_amass_to_json(npz_file_path, output_directory, max_frame_num, frame_stride)