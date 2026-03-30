"""
将 recon HDF5 数据集转换为 FlowNav/NoMaD 格式

目标结构:
    <output_dir>/recon/
        <traj_name>/
            0.jpg
            1.jpg
            ...
            traj_data.pkl   ← {"position": (T,2), "yaw": (T,)}

    <split_dir>/recon/
        train/traj_names.txt
        test/traj_names.txt

用法:
    python convert_recon.py \
        --input  /home/zhanyt/Downloads/recon_dataset.tar.gz/recon_release \
        --output /home/zhanyt/nomad_dataset \
        --splits /home/zhanyt/data_splits \
        --train_ratio 0.9
"""

import os
import io
import glob
import pickle
import argparse
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm


def decode_image(raw_bytes):
    """将 hdf5 中的 jpeg bytes 解码为 PIL Image"""
    try:
        return Image.open(io.BytesIO(bytes(raw_bytes))).convert('RGB')
    except Exception as e:
        print(f"  ⚠️  图像解码失败: {e}")
        return None


def convert_hdf5(hdf5_path, output_dir):
    """
    将单个 hdf5 文件转换为 NoMaD 格式目录
    返回轨迹名（成功）或 None（失败）
    """
    basename = os.path.splitext(os.path.basename(hdf5_path))[0]
    traj_dir = os.path.join(output_dir, basename)

    # 已转换则跳过
    pkl_path = os.path.join(traj_dir, 'traj_data.pkl')
    if os.path.exists(pkl_path):
        return basename

    try:
        with h5py.File(hdf5_path, 'r') as f:
            n = f['jackal/position'].shape[0]

            # 读取位置和偏航角
            position_3d = f['jackal/position'][:]   # (T, 3)
            yaw         = f['jackal/yaw'][:]         # (T,)
            position_xy = position_3d[:, :2]         # (T, 2) 只取 xy

            # 读取图像
            images_raw = [f['images/rgb_left'][i] for i in range(n)]

        # 创建输出目录
        os.makedirs(traj_dir, exist_ok=True)

        # 保存图像
        valid_count = 0
        for i, raw in enumerate(images_raw):
            img = decode_image(raw)
            if img is not None:
                img.save(os.path.join(traj_dir, f'{i}.jpg'), quality=95)
                valid_count += 1

        if valid_count == 0:
            print(f"  ⚠️  {basename}: 没有有效图像，跳过")
            return None

        # 保存 traj_data.pkl
        traj_data = {
            'position': position_xy.astype(np.float32),  # (T, 2)
            'yaw':      yaw.astype(np.float32),           # (T,)
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(traj_data, f)

        return basename

    except Exception as e:
        print(f"  ❌ 转换失败 {basename}: {e}")
        return None


def write_split(traj_names, split_dir, dataset_name, train_ratio):
    """生成 train/test 的 traj_names.txt"""
    np.random.shuffle(traj_names)
    n_train = int(len(traj_names) * train_ratio)
    train_names = traj_names[:n_train]
    test_names  = traj_names[n_train:]

    for split, names in [('train', train_names), ('test', test_names)]:
        split_path = os.path.join(split_dir, dataset_name, split)
        os.makedirs(split_path, exist_ok=True)
        txt_path = os.path.join(split_path, 'traj_names.txt')
        with open(txt_path, 'w') as f:
            for name in names:
                f.write(name + '\n')
        print(f"  ✅ {split}: {len(names)} 条轨迹 → {txt_path}")

    return train_names, test_names


def main():
    parser = argparse.ArgumentParser(description='recon HDF5 → FlowNav/NoMaD 格式转换')
    parser.add_argument('--input',       type=str, required=True,
                        help='recon_release 文件夹（包含 .hdf5 文件）')
    parser.add_argument('--output',      type=str, required=True,
                        help='输出数据集根目录，例如 ~/nomad_dataset')
    parser.add_argument('--splits',      type=str, required=True,
                        help='split 文件输出目录，例如 ~/data_splits')
    parser.add_argument('--dataset_name', type=str, default='recon',
                        help='数据集名称（默认 recon）')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='训练集比例（默认 0.9）')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 找所有 hdf5 文件
    hdf5_files = sorted(glob.glob(os.path.join(args.input, '**/*.hdf5'), recursive=True))
    hdf5_files += sorted(glob.glob(os.path.join(args.input, '*.hdf5')))
    hdf5_files = sorted(set(hdf5_files))

    if not hdf5_files:
        print(f"❌ 在 {args.input} 下未找到 .hdf5 文件")
        return

    print(f"🔍 找到 {len(hdf5_files)} 个 hdf5 文件")

    # 输出目录
    output_dir = os.path.join(args.output, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # 转换
    traj_names = []
    for hdf5_path in tqdm(hdf5_files, desc='转换进度'):
        name = convert_hdf5(hdf5_path, output_dir)
        if name is not None:
            traj_names.append(name)

    print(f"\n✅ 成功转换 {len(traj_names)} / {len(hdf5_files)} 条轨迹")
    print(f"   数据保存在: {output_dir}")

    # 生成 train/test split
    print(f"\n📋 生成数据划分 (train {args.train_ratio:.0%} / test {1-args.train_ratio:.0%})")
    write_split(traj_names, args.splits, args.dataset_name, args.train_ratio)

    # 打印 yaml 配置片段
    print(f"""
─────────────────────────────────────────
在 flownav.yaml 的 datasets 部分填入:

datasets:
  {args.dataset_name}:
    data_folder: {output_dir}
    train: {os.path.join(args.splits, args.dataset_name, 'train')}
    test:  {os.path.join(args.splits, args.dataset_name, 'test')}
    end_slack: 3
    goals_per_obs: 1
    negative_mining: True
    waypoint_spacing: 1
─────────────────────────────────────────
""")


if __name__ == '__main__':
    main()