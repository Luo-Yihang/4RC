import os
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def visualize_results_recon(
    imgs,
    gt_tracks_world,
    pred_tracks_world,
    save_path=None,
    frame_stride=4,
    spatial_stride=4,
):
    """Visualize reconstruction results with 2D frames and 3D pointclouds."""
    spatial_stride = 1
    frame_stride = 8
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    if torch.is_tensor(gt_tracks_world):
        gt_tracks_world = gt_tracks_world.detach().cpu().numpy()
    if torch.is_tensor(pred_tracks_world):
        pred_tracks_world = pred_tracks_world.detach().cpu().numpy()

    gt_tracks_world = gt_tracks_world[:, ::spatial_stride, ::spatial_stride, :]
    pred_tracks_world = pred_tracks_world[:, ::spatial_stride, ::spatial_stride, :]

    y_flip_transform = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    for i in range(gt_tracks_world.shape[0]):
        valid_mask = np.linalg.norm(gt_tracks_world[i], axis=-1) > 0
        gt_tracks_world[i, valid_mask] = gt_tracks_world[i, valid_mask] @ y_flip_transform

        valid_mask = np.linalg.norm(pred_tracks_world[i], axis=-1) > 0
        pred_tracks_world[i, valid_mask] = pred_tracks_world[i, valid_mask] @ y_flip_transform

    for i, img in enumerate(imgs):
        img_save_path = os.path.join(save_path, f"frame_{i:03d}.png")
        img.save(img_save_path)

    frames_to_viz = list(range(0, len(imgs), frame_stride))

    from arc.dust3r.viz import SceneViz

    viz_combined = SceneViz()
    viz_gt_only = SceneViz()
    viz_pred_only = SceneViz()

    cmap = plt.get_cmap("rainbow")
    frame_colors = [
        cmap(i / max(1, len(frames_to_viz) - 1))[:3] for i in range(len(frames_to_viz))
    ]

    for idx, frame_idx in enumerate(frames_to_viz):
        frame_img = imgs[frame_idx]
        gt_pts3d = gt_tracks_world[frame_idx]
        pred_pts3d = pred_tracks_world[frame_idx]

        rgb_colors = (np.array(frame_img) / 255.0)[::spatial_stride, ::spatial_stride, :]

        gt_valid_mask = np.linalg.norm(gt_pts3d, axis=-1) > 0
        pred_valid_mask = np.linalg.norm(pred_pts3d, axis=-1) > 0

        frame_color = np.array(frame_colors[idx])

        gt_tinted_color = rgb_colors.copy()
        pred_tinted_color = rgb_colors.copy()

        gt_tinted_color[..., 2] = np.minimum(1.0, gt_tinted_color[..., 2] * 1.3)
        gt_tinted_color[..., 0] = gt_tinted_color[..., 0] * 0.8

        pred_tinted_color[..., 0] = np.minimum(1.0, pred_tinted_color[..., 0] * 1.3)
        pred_tinted_color[..., 2] = pred_tinted_color[..., 2] * 0.8

        viz_combined.add_pointcloud(gt_pts3d, gt_tinted_color, gt_valid_mask)
        viz_combined.add_pointcloud(pred_pts3d, pred_tinted_color, pred_valid_mask)
        viz_gt_only.add_pointcloud(gt_pts3d, rgb_colors, gt_valid_mask)
        viz_pred_only.add_pointcloud(pred_pts3d, rgb_colors, pred_valid_mask)

    glb_path_combined = os.path.join(save_path, "pointcloud_all_frames_tinted.glb")
    viz_combined.save_glb(glb_path_combined)

    glb_path_gt = os.path.join(save_path, "pointcloud_all_frames_gt_only.glb")
    viz_gt_only.save_glb(glb_path_gt)

    glb_path_pred = os.path.join(save_path, "pointcloud_all_frames_pred_only.glb")
    viz_pred_only.save_glb(glb_path_pred)

    print(f"Saved visualizations to {save_path}")
    return glb_path_combined


def visualize_results(imgs, gt_tracks_world, pred_tracks_world, extrinsics, intrinsics, save_path=None):
    """Visualizes and saves 2D and 3D track visualization."""
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        rgb_dir = os.path.join(save_path, "rgb_input")
        os.makedirs(rgb_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            if isinstance(img, Image.Image):
                img_to_save = img
            else:
                img_to_save = Image.fromarray(np.asarray(img))
            img_to_save.save(os.path.join(rgb_dir, f"{i:06d}.png"))

    num_gt = gt_tracks_world.shape[1]
    num_pred = pred_tracks_world.shape[1]
    num_tracks = min(num_gt, num_pred)
    if num_tracks > 300:
        random_indices = np.random.choice(num_tracks, 300, replace=False)
    else:
        random_indices = np.arange(num_tracks)

    first_frame_y = gt_tracks_world[0, :, 1]
    random_indices = random_indices[first_frame_y[random_indices].argsort()]
    gt_tracks_world = gt_tracks_world[:, random_indices]
    pred_tracks_world = pred_tracks_world[:, random_indices]

    gt_tracks_2d = project_points_to_2d(gt_tracks_world, extrinsics, intrinsics)
    pred_tracks_2d = project_points_to_2d(pred_tracks_world, extrinsics, intrinsics)

    visualize_2d_tracks_on_video(imgs, gt_tracks_2d, pred_tracks_2d, save_path)
    # video3d_viz_combined = plot_3d_tracks(gt_tracks_world, pred_tracks_world)

    # if save_path:
    #     media.write_video(save_path + "/3d_tracks.mp4", video3d_viz_combined, fps=15)
    # else:
    #     media.show_video(video3d_viz_combined, fps=15)


def project_points_to_2d(points_3d, extrinsics, intrinsics):
    num_frames, num_points = points_3d.shape[:2]
    points_2d = np.zeros((num_frames, num_points, 2))

    for t in range(num_frames):
        R = extrinsics[t, :3, :3]
        tvec = extrinsics[t, :3, 3]

        points_camera = (points_3d[t] @ R.T) + tvec

        u_d = points_camera[:, 0] / (points_camera[:, 2] + 1e-8)
        v_d = points_camera[:, 1] / (points_camera[:, 2] + 1e-8)
        f_u, f_v, c_u, c_v = intrinsics
        u_d = u_d * f_u + c_u
        v_d = v_d * f_v + c_v

        points_2d[t] = np.stack([u_d, v_d], axis=-1)

    return points_2d


def visualize_2d_tracks_on_video(imgs, gt_tracks_2d, pred_tracks_2d, save_path=None, tracks_leave_trace=8):
    num_frames, num_points = gt_tracks_2d.shape[:2]

    images = [np.array(img).copy() for img in imgs]
    frames = []

    for t in range(num_frames):
        frame = images[t].copy()
        alpha = 0.5
        for i in range(num_points):
            gt_point = gt_tracks_2d[t, i]
            pred_point = pred_tracks_2d[t, i]

            if np.any(np.isnan(gt_point)) or np.any(np.isnan(pred_point)):
                continue

            for trail in range(max(0, t - tracks_leave_trace), t + 1):
                if trail == t:
                    color_gt = (0, 255, 0)
                    color_pred = (255, 0, 0)
                else:
                    color_gt = (0, 128, 0)
                    color_pred = (128, 0, 0)

                gt_trail_point = gt_tracks_2d[trail, i]
                pred_trail_point = pred_tracks_2d[trail, i]

                cv2.circle(frame, (int(gt_trail_point[0]), int(gt_trail_point[1])), 2, color_gt, -1)
                cv2.circle(frame, (int(pred_trail_point[0]), int(pred_trail_point[1])), 2, color_pred, -1)

        frames.append(frame)

    if save_path:
        import mediapy as media
        out_path = os.path.join(save_path, "2d_tracks.mp4")
        media.write_video(out_path, frames, fps=15)
    else:
        import mediapy as media
        media.show_video(frames, fps=15)


def plot_3d_tracks(gt_tracks_world, pred_tracks_world, tracks_leave_trace=8):
    num_frames, num_points = gt_tracks_world.shape[:2]
    frames = []

    fig = plt.figure(figsize=(10, 10))
    canvas = FigureCanvasAgg(fig)

    colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, num_points))
    marker_size = 10
    line_width = 1.0

    for t in range(num_frames):
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=20, azim=45)

        for i in range(num_points):
            color = colors[i]
            gt_line = gt_tracks_world[max(0, t - tracks_leave_trace) : t + 1, i]
            pred_line = pred_tracks_world[max(0, t - tracks_leave_trace) : t + 1, i]

            if len(gt_line) > 1:
                ax.plot(
                    xs=gt_line[:, 0],
                    ys=gt_line[:, 2],
                    zs=gt_line[:, 1],
                    color=color,
                    linewidth=line_width,
                )
            ax.scatter(
                xs=gt_line[-1, 0],
                ys=gt_line[-1, 2],
                zs=gt_line[-1, 1],
                color=color,
                s=marker_size,
                marker="o",
                edgecolors="black",
                linewidth=0.5,
            )

            if len(pred_line) > 1:
                ax.plot(
                    xs=pred_line[:, 0],
                    ys=pred_line[:, 2],
                    zs=pred_line[:, 1],
                    color=color,
                    linewidth=line_width,
                    linestyle="dotted",
                    alpha=0.7,
                )
            ax.scatter(
                xs=pred_line[-1, 0],
                ys=pred_line[-1, 2],
                zs=pred_line[-1, 1],
                color=color,
                s=marker_size * 1.2,
                marker="x",
                linewidth=1.5,
            )

        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        canvas.draw()

        rgba = np.asarray(canvas.buffer_rgba())
        rgb = rgba[..., :3]
        frames.append(rgb)

    return np.array(frames)


def project_points_to_video_frame(tracks_xyz_cam, intrinsics, img_height, img_width):
    fx, fy, cx, cy = intrinsics
    u = (tracks_xyz_cam[..., 0] / (tracks_xyz_cam[..., 2] + 1e-8)) * fx + cx
    v = (tracks_xyz_cam[..., 1] / (tracks_xyz_cam[..., 2] + 1e-8)) * fy + cy
    valid = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    uv = np.stack([u, v], axis=-1)
    return uv, valid


def load_npz_data(npz_path, num_frames=None, normalize_cam=True):
    in_npz = np.load(npz_path, allow_pickle=True)

    images_jpeg_bytes = in_npz["images_jpeg_bytes"]
    tracks_xyz_cam = in_npz["tracks_XYZ"]
    intrinsics = in_npz["fx_fy_cx_cy"]
    visibility = in_npz["visibility"]

    if "extrinsics_w2c" in in_npz.files:
        extrinsics_w2c = in_npz["extrinsics_w2c"]
    else:
        extrinsics_w2c = None

    if num_frames is None:
        num_frames = len(images_jpeg_bytes)
    if len(images_jpeg_bytes) > num_frames:
        print(f"Loaded {len(images_jpeg_bytes)} frames from {npz_path}, subsampled to {num_frames}")
    else:
        print(f"Loaded {len(images_jpeg_bytes)} frames from {npz_path}")
    if num_frames is not None:
        images_jpeg_bytes = images_jpeg_bytes[:num_frames]
        tracks_xyz_cam = tracks_xyz_cam[:num_frames]
        visibility = visibility[:num_frames]
        if extrinsics_w2c is not None:
            extrinsics_w2c = extrinsics_w2c[:num_frames]

    video_list = []
    for frame_bytes in images_jpeg_bytes:
        arr = np.frombuffer(frame_bytes, np.uint8)
        image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        video_list.append(pil_image)

    h, w = video_list[0].height, video_list[0].width
    if tracks_xyz_cam.ndim == 4:
        tracks_uv = None
    else:
        tracks_uv, _ = project_points_to_video_frame(tracks_xyz_cam, intrinsics, h, w)

    if normalize_cam and (extrinsics_w2c is not None):
        first_inv = np.linalg.inv(extrinsics_w2c[0])
        for i in range(extrinsics_w2c.shape[0]):
            extrinsics_w2c[i] = extrinsics_w2c[i] @ first_inv

    if extrinsics_w2c is not None:
        if tracks_xyz_cam.ndim == 4:
            # Dense tracks are already in world coordinates.
            tracks_xyz_world = tracks_xyz_cam
        else:
            extrinsics_c2w = np.linalg.inv(extrinsics_w2c)
            tracks_xyz_world = np.zeros_like(tracks_xyz_cam)
            for i in range(tracks_xyz_cam.shape[0]):
                R = extrinsics_c2w[i, :3, :3]
                t = extrinsics_c2w[i, :3, 3]
                tracks_xyz_world[i] = (R @ tracks_xyz_cam[i].T).T + t
    else:
        tracks_xyz_world = tracks_xyz_cam
        extrinsics_w2c = np.tile(np.eye(4), (num_frames, 1, 1))

    video_name = os.path.splitext(os.path.basename(npz_path))[0]

    return (
        video_list,
        tracks_xyz_cam,
        tracks_uv,
        intrinsics,
        tracks_xyz_world,
        visibility,
        video_name,
        extrinsics_w2c,
    )


def load_npz_data_recon(npz_path, num_frames=None, normalize_cam=True):
    in_npz = np.load(npz_path, allow_pickle=True)
    images_jpeg_bytes = in_npz["images_jpeg_bytes"]
    depth_map = in_npz["depth_map"]
    intrinsics = in_npz["fx_fy_cx_cy"]
    visibility = in_npz["visibility"]

    if "extrinsics_w2c" in in_npz.files:
        extrinsics_w2c = in_npz["extrinsics_w2c"]
    else:
        extrinsics_w2c = None

    if num_frames is None:
        num_frames = len(images_jpeg_bytes)
    images_jpeg_bytes = images_jpeg_bytes[:num_frames]
    depth_map = depth_map[:num_frames]
    visibility = visibility[:num_frames]
    if extrinsics_w2c is not None:
        extrinsics_w2c = extrinsics_w2c[:num_frames]

    video_list = []
    for frame_bytes in images_jpeg_bytes:
        arr = np.frombuffer(frame_bytes, np.uint8)
        image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        video_list.append(pil_image)

    h, w = video_list[0].height, video_list[0].width

    if normalize_cam and (extrinsics_w2c is not None):
        first_inv = np.linalg.inv(extrinsics_w2c[0])
        for i in range(extrinsics_w2c.shape[0]):
            extrinsics_w2c[i] = extrinsics_w2c[i] @ first_inv

    extrinsics_c2w = np.linalg.inv(extrinsics_w2c)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map

    x = (u - intrinsics[2]) * z / intrinsics[0]
    y = (v - intrinsics[3]) * z / intrinsics[1]
    recon_xyz_cam = np.stack([x, y, z], axis=-1)
    recon_xyz_world = np.zeros_like(recon_xyz_cam)

    if extrinsics_w2c is not None:
        for i in range(recon_xyz_cam.shape[0]):
            R = extrinsics_c2w[i, :3, :3]
            t = extrinsics_c2w[i, :3, 3]

            recon_xyz_world_flat = (R @ recon_xyz_cam[i].reshape(-1, 3).T).T + t
            recon_xyz_world[i] = recon_xyz_world_flat.reshape(h, w, 3)
    else:
        recon_xyz_world = recon_xyz_cam

    video_name = os.path.splitext(os.path.basename(npz_path))[0]

    return (
        video_list,
        depth_map,
        intrinsics,
        recon_xyz_world,
        recon_xyz_cam,
        visibility,
        video_name,
        extrinsics_w2c,
    )
