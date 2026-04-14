# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import threading
import numpy as np
from tqdm.auto import tqdm
from matplotlib import cm
import cv2
from scipy import ndimage
import torch

import viser
import viser.transforms as tf
from arc.dust3r.utils.device import to_numpy
from arc.viz.motion_seg import prepare_refine_mask  # noqa: F401  (re-exported for callers)


# ----------------- Helper Functions -----------------

def detect_sky_mask(img_rgb):
    """
    Detect sky pixels using HSV color space and morphological operations.
    Args:
        img_rgb: RGB image normalized to [-1, 1]
    Returns:
        Boolean mask (as int8) where True indicates non-sky pixels.
    """
    img = ((img_rgb + 1) * 127.5).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([105, 50, 140])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_light_blue = np.array([95, 5, 150])
    upper_light_blue = np.array([145, 100, 255])
    mask_light_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue)

    lower_white = np.array([0, 0, 235])
    upper_white = np.array([180, 10, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask = mask_blue | mask_light_blue | mask_white

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = mask.astype(bool)
    labels, num_labels = ndimage.label(mask)
    if num_labels > 0:
        top_row_labels = set(labels[0, :])
        top_row_labels.discard(0)
        if top_row_labels:
            mask = np.isin(labels, list(top_row_labels))
            labels, num_labels = ndimage.label(mask)
            if num_labels > 0:
                sizes = ndimage.sum(mask, labels, range(1, num_labels + 1))
                mask_size = mask.size
                big_enough = sizes > mask_size * 0.01
                mask = np.isin(labels, np.where(big_enough)[0] + 1)
    return (~mask).astype(np.int8)

def is_outdoor_scene(frame_data_list):
    sky_ratios = []
    for fd in frame_data_list:
        mask = fd.get('sorted_not_sky', np.ones(1))
        sky_ratio = 1.0 - np.mean(mask)
        sky_ratios.append(float(sky_ratio))
    significant = sum(1 for ratio in sky_ratios if ratio > 0.2)
    return significant >= len(sky_ratios) / 4


# ----------------- Update Handlers -----------------

def update_points_filtering(server, frame_data_list, gui_conf_percentile, gui_mask_sky, gui_show_confidence_color):
    for i in range(len(frame_data_list)):
        fd = frame_data_list[i]
        total = len(fd['sorted_pts'])

        num_pts = max(1, int(total * (100 - gui_conf_percentile.value) / 100))

        if gui_mask_sky.value:
            mask = fd['sorted_not_sky'][:num_pts]
            pts = fd['sorted_pts'][:num_pts][mask > 0]

            if gui_show_confidence_color.value:
                colors = fd['colors_confidence'][:num_pts][mask > 0]
            else:
                colors = fd['colors_rgb'][:num_pts][mask > 0]
        else:
            pts = fd['sorted_pts'][:num_pts]

            if gui_show_confidence_color.value:
                colors = fd['colors_confidence'][:num_pts]
            else:
                colors = fd['colors_rgb'][:num_pts]

        fd['point_node'].points = pts
        fd['point_node'].colors = colors
                
        server.flush()

# ----------------- Main Visualization Function -----------------
def start_visualization(output, min_conf_thr_percentile=2.5, port=8020, point_size=0.0004,
                        refine_track_visual=None):
    server = viser.ViserServer(host='127.0.0.1', port=port)
    server.gui.set_panel_label("Show Controls")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False)

    @server.on_client_connect
    def on_client_connect(client: viser.ClientHandle) -> None:
        with client.atomic():
            client.camera.position = (-0.00141163, -0.01910395, -0.06794288)
            client.camera.look_at = (-0.00352821, -0.01143425, 0.0154939)
        client.flush()

    server.scene.set_up_direction((0.0, -1.0, 0.0))
    server.scene.world_axes.visible = False

    num_frames = len(output['preds'])
    frame_data_list = []
    cumulative_pts = []

    # Resolve refine_track_visual early so GUI can use it
    if refine_track_visual is None:
        refine_track_visual = bool(output.get("refine_track_visual", True))

    # ----------------- Grouped GUI Controls -----------------
    # 50% to reduce render time
    default_track_percent = 50
    default_track_len = min(10, num_frames)

    with server.gui.add_folder("Point and Camera Options", expand_by_default=False):
        gui_point_size = server.gui.add_slider("Point Size", min=1e-6, max=0.005, step=1e-5, initial_value=point_size)
        gui_track_width = server.gui.add_slider("Track Width", min=0.01, max=1.0, step=0.01, initial_value=0.5)
        gui_frustum_size_percent = server.gui.add_slider("Camera Size (%)", min=0.1, max=5.0, step=0.1, initial_value=1.0)
        gui_mask_sky = server.gui.add_checkbox("Mask Sky", True)
        gui_show_confidence_color = server.gui.add_checkbox("Show Confidence", False)

    with server.gui.add_folder("Playback Options", expand_by_default=True):
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames - 1, step=1, initial_value=0)
        gui_next_frame = server.gui.add_button("Next Frame")
        gui_prev_frame = server.gui.add_button("Prev Frame")
        gui_playing = server.gui.add_checkbox("Play", False)
        gui_framerate = server.gui.add_slider("FPS", min=0.25, max=30, step=0.25, initial_value=5)
        gui_framerate_options = server.gui.add_button_group("FPS options", ("1", "2", "5", "10"))

    with server.gui.add_folder("Display Options", expand_by_default=False):
        gui_show_points = server.gui.add_checkbox("Points", True)
        gui_show_track = server.gui.add_checkbox("Track", True)
        gui_show_camera = server.gui.add_checkbox("Camera", True)

    with server.gui.add_folder("Confidence Options", expand_by_default=False):
        gui_conf_percentile = server.gui.add_slider("Conf Percentile", min=0, max=100, step=1, initial_value=min_conf_thr_percentile)

    # ----------------- Dynamic Segmentation Info (only when refine_track_visual) -----------------
    if refine_track_visual:
        detected_text = output.get("track_dynamic_objects_text", "")
        query_img_uint8 = output.get("track_query_img_uint8")
        track_mask_list = output.get("track_text_mask")
        track_mask = track_mask_list[0] if (track_mask_list and track_mask_list[0] is not None) else None

        with server.gui.add_folder("Dynamic Segmentation", expand_by_default=False):
            _label = detected_text if detected_text else "—"
            server.gui.add_markdown(f"**Detected:** {_label}")
            if query_img_uint8 is not None and track_mask is not None:
                # Overlay mask as a green tint on the query frame
                overlay = query_img_uint8.copy().astype(np.float32)
                green_tint = np.array([0, 200, 80], dtype=np.float32)
                overlay[track_mask.astype(bool)] = (
                    overlay[track_mask.astype(bool)] * 0.45 + green_tint * 0.55
                )
                preview = overlay.clip(0, 255).astype(np.uint8)
                server.gui.add_image(preview, label="SAM2 Mask", format="jpeg")
            elif query_img_uint8 is not None:
                server.gui.add_image(query_img_uint8, label="Query Frame (no mask)", format="jpeg")

    @gui_next_frame.on_click
    def next_frame(_):
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def prev_frame(_):
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def playing_update(_):
        state = gui_playing.value
        gui_timestep.disabled = state
        gui_next_frame.disabled = state
        gui_prev_frame.disabled = state

    @gui_framerate_options.on_click
    def fps_options(_):
        gui_framerate.value = float(gui_framerate_options.value)

    server.scene.add_frame("/cams", show_axes=False)

    # ----------------- Frame Processing -----------------

    def _normalize_track_query_idx(track_query_idx, num_frames):
        if isinstance(track_query_idx, torch.Tensor):
            track_query_idx = track_query_idx.detach().cpu().flatten().tolist()
        elif isinstance(track_query_idx, (list, tuple)):
            track_query_idx = list(track_query_idx)
        else:
            track_query_idx = [int(track_query_idx)]
        track_query_idx = [int(idx) for idx in track_query_idx if 0 <= int(idx) < num_frames]
        if not track_query_idx:
            track_query_idx = [0]
        return track_query_idx

    first_pred = output['preds'][0]
    track_query_idx = _normalize_track_query_idx(
        first_pred.get('track_query_idx', 0), num_frames
    )[0]

    # Precompute track filter mask (constant across frames)
    # If refine_track_visual=True and no pre-built mask: auto-detect via VLA + SAM2.
    # If refine_track_visual=False: no mask applied.
    track_filter_mask = None
    if refine_track_visual:
        track_text_mask = output.get("track_text_mask")
        track_text_mask_list = None
        if isinstance(track_text_mask, (list, tuple)):
            track_text_mask_list = [
                mask.astype(bool) if mask is not None else None
                for mask in track_text_mask
            ]
        elif track_text_mask is not None:
            track_text_mask_list = [track_text_mask.astype(bool)]

        if track_text_mask_list is not None and len(track_text_mask_list) > 0:
            # Use pre-built mask (either passed externally or computed by prepare_refine_mask)
            mask = track_text_mask_list[0]
            if mask is not None:
                kernel = np.ones((3, 3), dtype=np.uint8)
                track_filter_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=3).astype(bool).reshape(-1)
        else:
            print("[refine_track_visual] No mask found. Call prepare_refine_mask() before launching the subprocess.")

    print(f"refine_track_visual: {refine_track_visual}")
    print(f"track_query_idx: {track_query_idx}")

    for i in tqdm(range(num_frames)):
        pred = output['preds'][i]
        view = output['views'][i]

        img_rgb_orig = to_numpy(view['img'].cpu().squeeze().permute(1,2,0))
        not_sky_mask = detect_sky_mask(img_rgb_orig).flatten().astype(np.int8)

        pts3d = to_numpy(pred['pts'].cpu().squeeze()).reshape(-1, 3)
        conf = to_numpy(pred['conf'].cpu().squeeze()).flatten()

        img_rgb = to_numpy(view['img'].cpu().squeeze().permute(1,2,0))
        img_rgb_flat = img_rgb.reshape(-1, 3)

        cumulative_pts.append(pts3d)

        sort_idx = np.argsort(-conf)
        sorted_conf = conf[sort_idx]
        sorted_pts = pts3d[sort_idx]
        sorted_img_rgb = img_rgb_flat[sort_idx]
        sorted_not_sky = not_sky_mask[sort_idx]

        track_multi = pred.get("track_multi")
        conf_track_multi = pred.get("conf_track_multi")
        if track_multi is not None and conf_track_multi is not None:
            pts3d_track = to_numpy(track_multi[:, 0].cpu().squeeze()).reshape(-1, 3)
            conf_track_flat = to_numpy(conf_track_multi[:, 0].cpu().squeeze()).flatten()
        else:
            pts3d_track = to_numpy(pred["track"].cpu().squeeze()).reshape(-1, 3)
            conf_track_flat = to_numpy(pred["conf_track"].cpu().squeeze()).flatten()

        colors_rgb = ((sorted_img_rgb + 1) * 127.5).astype(np.uint8) / 255.0

        conf_norm = (sorted_conf - sorted_conf.min()) / (sorted_conf.max() - sorted_conf.min() + 1e-8)
        colormap = cm.turbo
        colors_confidence = colormap(conf_norm)[:, :3]

        rainbow_color_for_frame = tuple(np.array(cm.hsv(i / num_frames)[:3]))

        c2w = to_numpy(pred['extrinsic'].cpu().squeeze())
        height, width = view['img'].shape[2], view['img'].shape[3]
        focal_length = float(pred['intrinsic'][0, 0])
        img_rgb_reshaped = img_rgb.reshape(height, width, 3)
        img_rgb_normalized = ((img_rgb_reshaped + 1) * 127.5).astype(np.uint8)
        img_downsampled = img_rgb_normalized[::4, ::4]

        frame_data = {
            'sorted_pts': sorted_pts,
            'colors_rgb': colors_rgb,
            'colors_confidence': colors_confidence,
            'sorted_not_sky': sorted_not_sky,
            'conf_flat': conf,
            'pts3d_track': pts3d_track,
            'conf_track_flat': conf_track_flat,
            'pts3d_track_masked': None,
            'c2w': c2w,
            'height': height,
            'width': width,
            'focal_length': focal_length,
            'img_downsampled': img_downsampled,
            'rainbow_color': rainbow_color_for_frame,
        }

        frame_data_list.append(frame_data)

    # Percentile for scene extent calculation
    extent_percentile = 80
    cumulative_pts_combined = np.concatenate(cumulative_pts, axis=0)
    min_coords = np.percentile(cumulative_pts_combined, 100 - extent_percentile, axis=0)
    max_coords = np.percentile(cumulative_pts_combined, extent_percentile, axis=0)
    scene_extent = max_coords - min_coords
    max_extent = np.max(scene_extent)

    # ----------------- Create Visualization Nodes -----------------
    _track_lock = threading.Lock()
    track_state = {'track_valid_mask': None, 'moving_mask': None, 'norms': None}

    def _recompute_track_data(conf_pct):
        """Recompute track masked data and valid/moving masks based on conf percentile."""
        n_pixels = len(frame_data_list[0]['pts3d_track'])

        # 1. Aggregate track conf across all frames
        agg_track_conf = np.zeros(n_pixels)
        for fd in frame_data_list:
            agg_track_conf += fd['conf_track_flat']

        pts0 = frame_data_list[0]['pts3d_track']
        max_disp = np.zeros(n_pixels, dtype=np.float32)
        for fd in frame_data_list[1:]:
            np.maximum(max_disp, np.linalg.norm(fd['pts3d_track'] - pts0, axis=1), out=max_disp)
        dynamic_ratio = (max_disp > 0.025).mean()
        effective_pct = conf_pct if dynamic_ratio / 2.0 >= conf_pct / 100 else dynamic_ratio / 2.0
        num_keep = max(1, int(n_pixels * (100 - effective_pct) / 100))

        # 2. Track conf mask: top pixels by aggregated track conf
        top_idx = np.argsort(-agg_track_conf)[:num_keep]
        track_conf_mask = np.zeros(n_pixels, dtype=bool)
        track_conf_mask[top_idx] = True

        # 3. Pts conf mask at query frame
        query_conf = frame_data_list[track_query_idx]['conf_flat']
        pts_top_idx = np.argsort(-query_conf)[:num_keep]
        pts_conf_mask = np.zeros(n_pixels, dtype=bool)
        pts_conf_mask[pts_top_idx] = True

        # 4. Intersection of track conf mask and pts conf mask
        combined = track_conf_mask & pts_conf_mask

        # 5. Also apply text-based filter mask if exists
        if track_filter_mask is not None:
            combined = combined & track_filter_mask

        # 6. Rebuild masked data for all frames using combined mask
        for fd in frame_data_list:
            fd['pts3d_track_masked'] = fd['pts3d_track'][combined]

        # Recompute track valid and moving masks
        track_points_first = frame_data_list[track_query_idx]['pts3d_track_masked']
        tv_mask = ~(np.all(np.abs(track_points_first) < 0.001, axis=1))
        if tv_mask.sum() == 0:
            tv_mask = np.ones(track_points_first.shape[0], dtype=bool)
        track_points_first_valid = track_points_first[tv_mask]

        delta = np.zeros_like(track_points_first_valid)
        for frame_i in range(num_frames - 1, num_frames):
            tp = frame_data_list[frame_i]['pts3d_track_masked'][tv_mask]
            delta += abs(track_points_first_valid - tp)
        norms = np.linalg.norm(delta, axis=1)
        threshold = np.percentile(norms, 0) if norms.size > 0 else 0.0
        mv_mask = norms >= threshold

        track_state['track_valid_mask'] = tv_mask
        track_state['moving_mask'] = mv_mask
        track_state['norms'] = norms
        track_state['agg_conf_final'] = agg_track_conf[combined][tv_mask][mv_mask]

    _recompute_track_data(min_conf_thr_percentile)

    def _build_track_lines_for_frame(frame_idx):
        tv_mask = track_state['track_valid_mask']
        motion_mask = track_state['moving_mask']

        lines = []
        start_idx = max(frame_idx - default_track_len + 1, 0)
        for j in range(start_idx, frame_idx + 1):
            track_points_prev = frame_data_list[max(j - 1, 0)]['pts3d_track_masked'][tv_mask][motion_mask]
            track_points_next = frame_data_list[j]['pts3d_track_masked'][tv_mask][motion_mask]
            line = np.stack((track_points_prev, track_points_next), axis=1)
            lines.append(line)

        if not lines:
            empty = np.zeros((0, 2, 3))
            return empty, empty

        lines = np.stack(lines, axis=1)
        lines = lines[::10]
        num_lines, len_t = lines.shape[:2]

        track_colors = generate_uniform_colors(num_lines)
        track_colors = track_colors[:, np.newaxis, np.newaxis, :]
        track_colors = np.tile(track_colors, (1, len_t, 2, 1))

        if num_lines == 0:
            lines_flat = lines.reshape(-1, 2, 3)
            colors_flat = track_colors.reshape(-1, 2, 3)
        else:
            keep_count = max(1, int(round(num_lines * (default_track_percent / 100.0))))
            agg_conf = track_state.get('agg_conf_final')
            w = agg_conf[:num_lines].astype(np.float64)
            w = w - w.min()
            w_sum = w.sum()
            prob = w / w_sum if w_sum > 0 else np.ones(num_lines) / num_lines
            keep_idx = np.random.choice(num_lines, size=min(keep_count, num_lines), replace=False, p=prob)
            lines_flat = lines[keep_idx].reshape(-1, 2, 3)
            colors_flat = track_colors[keep_idx].reshape(-1, 2, 3)

        return lines_flat, colors_flat

    import matplotlib.colors as mcolors
    def generate_uniform_colors(N):
        hues = np.linspace(0, 1, N, endpoint=False)
        hsv = np.stack([hues, np.ones(N), np.ones(N)], axis=1)
        rgb = mcolors.hsv_to_rgb(hsv)
        rgb_255 = (rgb * 255).astype(int)
        return rgb_255

    for i in tqdm(range(num_frames)):
        fd = frame_data_list[i]
        frame_node = server.scene.add_frame(f"/cams/t{i}", show_axes=False, visible=False)

        lines, track_colors = _build_track_lines_for_frame(i)

        track_node = server.scene.add_line_segments(
            name = f"/track/t{i}",
            points = lines,
            colors = track_colors,
            line_width=gui_track_width.value,
            visible=False,
        )

        point_node = server.scene.add_point_cloud(
            name=f"/points/t{i}",
            points=fd['sorted_pts'],
            colors=fd['colors_rgb'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=False,
        )

        rotation_matrix = fd['c2w'][:3, :3]
        position = fd['c2w'][:3, 3]
        rotation_quaternion = tf.SO3.from_matrix(rotation_matrix).wxyz
        try:
            fov = 2 * np.arctan2(fd['height'] / 2, fd['focal_length'])
        except Exception as e:
            print(f"Error calculating FOV: {e}")
            fov = 60
        aspect_ratio = fd['width'] / fd['height']
        frustum_scale = max_extent * (gui_frustum_size_percent.value / 100.0)

        frustum_node = server.scene.add_camera_frustum(
            name=f"/cams/t{i}/frustum",
            fov=fov,
            aspect=aspect_ratio,
            scale=frustum_scale,
            color=fd['rainbow_color'],
            image=fd['img_downsampled'],
            wxyz=rotation_quaternion,
            position=position,
            visible=False,
        )

        fd['frame_node'] = frame_node
        fd['point_node'] = point_node
        fd['track_node'] = track_node
        fd['frustum_node'] = frustum_node

    # Initially set all nodes hidden
    for fd in frame_data_list:
        fd['frame_node'].visible = False
        fd['point_node'].visible = False
        fd['track_node'].visible = False
        fd['frustum_node'].visible = False
    server.flush()

    gui_timestep.value = 0
    gui_playing.value = False

    # Scene type detection and sky masking initialization
    is_outdoor = is_outdoor_scene(frame_data_list)
    gui_mask_sky.value = True

    print("\nScene type detection:")
    sky_ratios = [1.0 - np.mean(fd['sorted_not_sky']) for fd in frame_data_list]
    significant = sum(1 for r in sky_ratios if r > 0.2)
    print(f"- Found {significant}/{len(sky_ratios)} frames with significant sky presence (>20% sky pixels)")
    print(f"- Scene classified as: {'outdoor' if is_outdoor else 'indoor'}, setting mask_sky to {is_outdoor}")

    # Initial visibility setup — apply conf percentile filtering from the start
    with server.atomic():
        for i in range(num_frames):
            fd = frame_data_list[i]
            if i == gui_timestep.value:
                fd['frame_node'].visible = True
                fd['frustum_node'].visible = gui_show_camera.value
            else:
                fd['frame_node'].visible = False
                fd['frustum_node'].visible = False

            total = len(fd['sorted_pts'])
            num_pts = max(1, int(total * (100 - gui_conf_percentile.value) / 100))
            pts = fd['sorted_pts'][:num_pts]

            if gui_show_confidence_color.value:
                colors = fd['colors_confidence'][:num_pts]
            else:
                colors = fd['colors_rgb'][:num_pts]

            if is_outdoor and gui_mask_sky.value:
                mask = fd['sorted_not_sky'][:num_pts]
                pts = pts[mask > 0]
                colors = colors[mask > 0]

            fd['point_node'].points = pts
            fd['point_node'].colors = colors
            if i == gui_timestep.value:
                fd['point_node'].visible = gui_show_points.value
                fd['track_node'].visible = gui_show_track.value
            else:
                fd['point_node'].visible = False
                fd['track_node'].visible = False

    server.flush()

    # ----------------- GUI Callback Updates -----------------
    @gui_timestep.on_update
    def _(_):
        current = int(gui_timestep.value)
        with server.atomic():
            for i in range(num_frames):
                fd = frame_data_list[i]
                if i == current:
                    fd['frame_node'].visible = True
                    fd['frustum_node'].visible = gui_show_camera.value
                    fd['point_node'].visible = gui_show_points.value
                    fd['track_node'].visible = gui_show_track.value
                else:
                    fd['frame_node'].visible = False
                    fd['frustum_node'].visible = False
                    fd['point_node'].visible = False
                    fd['track_node'].visible = False
        server.flush()

    @gui_point_size.on_update
    def _(_):
        with server.atomic():
            for fd in frame_data_list:
                fd['point_node'].point_size = gui_point_size.value
        server.flush()

    @gui_track_width.on_update
    def _(_):
        with server.atomic():
            for fd in frame_data_list:
                fd['track_node'].line_width = gui_track_width.value
        server.flush()

    @gui_frustum_size_percent.on_update
    def _(_):
        frustum_scale = max_extent * (gui_frustum_size_percent.value / 100.0)
        with server.atomic():
            for fd in frame_data_list:
                fd['frustum_node'].scale = frustum_scale
        server.flush()

    @gui_show_confidence_color.on_update
    def _(_):
        update_points_filtering(server, frame_data_list, gui_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color)

    @gui_conf_percentile.on_update
    def _(_):
        update_points_filtering(server, frame_data_list, gui_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color)
        with _track_lock:
            _recompute_track_data(gui_conf_percentile.value)
            with server.atomic():
                for frame_i, fd in enumerate(frame_data_list):
                    lines, colors = _build_track_lines_for_frame(frame_i)
                    fd['track_node'].points = lines
                    fd['track_node'].colors = colors
            server.flush()

    @gui_mask_sky.on_update
    def _(_):
        update_points_filtering(server, frame_data_list, gui_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color)

    @gui_show_points.on_update
    def _(_):
        current = int(gui_timestep.value)
        with server.atomic():
            frame_data_list[current]['point_node'].visible = gui_show_points.value
        server.flush()

    @gui_show_track.on_update
    def _(_):
        current = int(gui_timestep.value)
        with server.atomic():
            frame_data_list[current]['track_node'].visible = gui_show_track.value
        server.flush()

    @gui_show_camera.on_update
    def _(_):
        current = int(gui_timestep.value)
        with server.atomic():
            frame_data_list[current]['frustum_node'].visible = gui_show_camera.value
        server.flush()

    # ----------------- Start Playback Loop -----------------
    def local_playback_loop():
        while True:
            if gui_playing.value:
                gui_timestep.value = (int(gui_timestep.value) + 1) % num_frames
            time.sleep(1.0 / float(gui_framerate.value))
    playback_thread = threading.Thread(target=local_playback_loop)
    playback_thread.start()

    return server


# ----------------- NPZ I/O -----------------

def load_output_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    n = int(data["n_frames"])

    preds, views = [], []
    for i in range(n):
        pred = {k[len(f"pred_{i}_"):]: torch.from_numpy(data[k])
                for k in data.files if k.startswith(f"pred_{i}_")}
        preds.append(pred)
        view = {k[len(f"view_{i}_"):]: torch.from_numpy(data[k])
                for k in data.files if k.startswith(f"view_{i}_")}
        views.append(view)

    out = {"preds": preds, "views": views}

    out["refine_track_visual"] = bool(data["refine_track_visual"].item()) \
        if "refine_track_visual" in data.files else False

    for k in ("track_dynamic_objects_text", "track_query_img_uint8"):
        if k in data.files:
            v = data[k]
            out[k] = v.item() if v.ndim == 0 else v

    if "track_text_mask_n" in data.files:
        masks = [data[f"track_text_mask_{i}"] if f"track_text_mask_{i}" in data.files else None
                 for i in range(int(data["track_text_mask_n"]))]
        out["track_text_mask"] = masks

    return out


# ----------------- CLI entry point -----------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a 4RC .npz result with viser")
    parser.add_argument("--npz_path", required=True, help="Path to .npz saved by run_inference.py")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--point_size", type=float, default=0.0016)
    args = parser.parse_args()

    output_dict = load_output_npz(args.npz_path)
    server = start_visualization(
        output_dict,
        port=args.port,
        point_size=args.point_size,
        refine_track_visual=output_dict.get("refine_track_visual", False),
    )
    print(f"Visualization running at http://127.0.0.1:{server.get_port()}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass