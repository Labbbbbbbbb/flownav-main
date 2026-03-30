"""
Recon HDF5 数据集可视化工具 by claude
用法: python visualize_recon.py --hdf5 /path/to/file.hdf5
      python visualize_recon.py --folder /path/to/recon_release/
"""

import argparse
import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import io


def decode_image(img_bytes):
    """将 hdf5 中存储的 bytes 解码为 numpy array"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img)
    except Exception:
        return None


def visualize_trajectory(hdf5_path, max_steps=None, save_dir=None):
    """可视化单条轨迹"""
    print(f"\n📂 加载: {os.path.basename(hdf5_path)}")
    
    with h5py.File(hdf5_path, 'r') as f:
        n_steps = f['jackal/position'].shape[0]
        if max_steps:
            n_steps = min(n_steps, max_steps)

        print(f"   轨迹长度: {n_steps} 步")

        # 读取数据
        positions   = f['jackal/position'][:n_steps]        # (N, 3)
        lin_vel     = f['jackal/linear_velocity'][:n_steps] # (N,)
        ang_vel     = f['jackal/angular_velocity'][:n_steps]# (N,)
        yaw         = f['jackal/yaw'][:n_steps]             # (N,)
        cmd_lin     = f['commands/linear_velocity'][:n_steps]
        cmd_ang     = f['commands/angular_velocity'][:n_steps]
        collision   = f['collision/any'][:n_steps]          # (N,)
        gps_latlong = f['gps/latlong'][:n_steps]            # (N, 2)

        # 读图像（只取部分帧避免太慢）
        sample_indices = np.linspace(0, n_steps - 1, min(8, n_steps), dtype=int)
        images_left = []
        for i in sample_indices:
            raw = f['images/rgb_left'][i]
            img = decode_image(bytes(raw))
            if img is not None:
                images_left.append(img)

    # ─── 绘图 ───
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Recon Trajectory: {os.path.basename(hdf5_path)}", 
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # 1. 轨迹俯视图
    ax_traj = fig.add_subplot(gs[0:2, 0:2])
    sc = ax_traj.scatter(positions[:, 0], positions[:, 1],
                         c=np.arange(n_steps), cmap='viridis', s=15, zorder=2)
    # 碰撞点标红
    coll_idx = np.where(collision > 0)[0]
    if len(coll_idx):
        ax_traj.scatter(positions[coll_idx, 0], positions[coll_idx, 1],
                        c='red', s=40, zorder=3, label=f'碰撞({len(coll_idx)})')
        ax_traj.legend(fontsize=8)
    plt.colorbar(sc, ax=ax_traj, label='时间步')
    ax_traj.set_title('轨迹俯视图 (XY)')
    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, alpha=0.3)

    # 2. 线速度 vs 角速度
    ax_vel = fig.add_subplot(gs[0, 2:])
    t = np.arange(n_steps)
    ax_vel.plot(t, lin_vel,  label='实际线速度', color='steelblue')
    ax_vel.plot(t, cmd_lin,  label='指令线速度', color='steelblue', linestyle='--', alpha=0.6)
    ax_vel.set_title('线速度')
    ax_vel.set_xlabel('时间步')
    ax_vel.set_ylabel('m/s')
    ax_vel.legend(fontsize=8)
    ax_vel.grid(True, alpha=0.3)

    ax_ang = fig.add_subplot(gs[1, 2:])
    ax_ang.plot(t, ang_vel,  label='实际角速度', color='coral')
    ax_ang.plot(t, cmd_ang,  label='指令角速度', color='coral', linestyle='--', alpha=0.6)
    ax_ang.set_title('角速度')
    ax_ang.set_xlabel('时间步')
    ax_ang.set_ylabel('rad/s')
    ax_ang.legend(fontsize=8)
    ax_ang.grid(True, alpha=0.3)

    # 3. 偏航角
    ax_yaw = fig.add_subplot(gs[2, 0])
    ax_yaw.plot(t, np.degrees(yaw), color='purple')
    ax_yaw.set_title('偏航角 (deg)')
    ax_yaw.set_xlabel('时间步')
    ax_yaw.grid(True, alpha=0.3)

    # 4. 碰撞标签
    ax_coll = fig.add_subplot(gs[2, 1])
    ax_coll.fill_between(t, collision, color='red', alpha=0.5)
    ax_coll.set_title(f'碰撞标签 (共{int(collision.sum())}帧)')
    ax_coll.set_xlabel('时间步')
    ax_coll.set_ylim(-0.1, 1.5)
    ax_coll.grid(True, alpha=0.3)

    # 5. 样本图像（最多4张）
    for idx, img in enumerate(images_left[:4]):
        ax_img = fig.add_subplot(gs[2, 2 + idx] if idx < 2 else gs[2, 2 + idx - 2])
        # 重新排列避免越界
        pass

    # 单独画图像行
    fig2, axes_img = plt.subplots(2, 4, figsize=(16, 5))
    fig2.suptitle(f"RGB Left 样本帧 — {os.path.basename(hdf5_path)}", fontsize=11)
    for i, ax in enumerate(axes_img.flat):
        if i < len(images_left):
            ax.imshow(images_left[i])
            ax.set_title(f'step {sample_indices[i]}', fontsize=8)
        ax.axis('off')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(hdf5_path))[0]
        fig.savefig(os.path.join(save_dir, f'{base}_stats.png'), dpi=100, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, f'{base}_images.png'), dpi=100, bbox_inches='tight')
        print(f"   ✅ 保存到 {save_dir}")
        plt.close('all')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Recon HDF5 数据集可视化')
    parser.add_argument('--hdf5',   type=str, help='单个 hdf5 文件路径')
    parser.add_argument('--folder', type=str, help='包含 hdf5 文件的文件夹')
    parser.add_argument('--max_steps', type=int, default=None, help='最多显示多少步')
    parser.add_argument('--save_dir',  type=str, default=None,  help='保存图片的目录（不填则弹窗显示）')
    parser.add_argument('--max_files', type=int, default=5,     help='文件夹模式下最多处理几个文件')
    args = parser.parse_args()

    if args.hdf5:
        visualize_trajectory(args.hdf5, args.max_steps, args.save_dir)
    elif args.folder:
        files = sorted(glob.glob(os.path.join(args.folder, '**/*.hdf5'), recursive=True))
        files += sorted(glob.glob(os.path.join(args.folder, '*.hdf5')))
        files = list(set(files))[:args.max_files]
        print(f"找到 {len(files)} 个 hdf5 文件，处理前 {args.max_files} 个")
        for f in files:
            visualize_trajectory(f, args.max_steps, args.save_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
