import os
import math
import glob
import time
import json
import torch
import torchvision.transforms as tvf
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from natsort import natsorted
from typing import Optional

from arc.dust3r.utils.device import collate_with_cat
from arc.dust3r.utils.image import ImgNorm

from tapvid3d_utils import load_npz_data
from track_eval_util import compute_average_pts_within_thresh

ToTensor = tvf.ToTensor()


def _resize_pil_image(img, long_edge_size, nearest=False):
    import PIL.Image
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS if not nearest else PIL.Image.NEAREST
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def _prepare_views_from_pil_list(pil_list, size=512, patch_size=14, crop=False, square_ok=True):
    views = []
    for i, img in enumerate(pil_list):
        img = img.convert("RGB")
        W1, H1 = img.size
        if size == 256:
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 256:
            half = min(cx, cy)
            if crop:
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:
                img = img.resize((2 * half, 2 * half))
        else:
            halfw, halfh = ((2 * cx) // patch_size) * (patch_size // 2), ((2 * cy) // patch_size) * (patch_size // 2)
            if not square_ok and W == H:
                halfh = 3 * halfw / 4
            if crop:
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
            else:
                img = img.resize((2 * halfw, 2 * halfh))

        mask = ~(ToTensor(img)[None].sum(1) <= 0.01)
        views.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(views),
                instance=str(len(views)),
                mask=mask,
                dynamic_mask=torch.zeros_like(mask),
            )
        )
    assert views, "No images provided for evaluation."
    return views


def save_prediction_as_npy(pred_tracks, save_dir, video_name):
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(pred_tracks, torch.Tensor):
        pred_tracks = pred_tracks.detach().cpu().numpy()
    save_path = os.path.join(save_dir, f"{video_name}_track.npy")
    np.save(save_path, pred_tracks)
    print(f"Saved prediction to {save_path}")
    return save_path


def eval_ours_tapvid3d(
    args,
    model,
    device,
    data_type="pstudio",
    dyn_static=True,
    save_predictions=True,
    align="sim3",
    data_root="./data",
):
    return eval_tapvid3d(
        args.output_dir,
        args.num_frames,
        args=args,
        model=model,
        device=device,
        data_type=data_type,
        dyn_static=dyn_static,
        save_predictions=save_predictions,
        align=align,
        data_root=data_root,
    )


def get_inference_output(args, model, device, filelist, tracks_uv=None, visibility=None, size=None):
    model.eval()
    if size is None:
        size = args.size

    with torch.no_grad():
        if args.pose_eval_stride > 1:
            filelist = filelist[:: args.pose_eval_stride]
        if args.num_frames is not None:
            filelist = filelist[: args.num_frames]

        views = _prepare_views_from_pil_list(
            filelist,
            size=size,
            patch_size=args.patch_size,
            crop=False,
            square_ok=True,
        )
        if tracks_uv is not None and visibility is not None:
            visibility_mask = visibility[0]
            if visibility_mask.sum() > 0:
                query_uv = np.array(tracks_uv)[0, visibility_mask]
                gt_image_size = filelist[0].size
                pred_image_size = views[0]["img"].shape[-2:][::-1]
                query_uv = query_uv * np.array(pred_image_size) / np.array(gt_image_size)
                query_points = torch.tensor(query_uv, dtype=torch.float32)
                for view in views:
                    view["query_points"] = query_points

        views = collate_with_cat([tuple(views)])

        for view in views:
            for name in "img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres query_points".split():
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        with torch.no_grad():
            predictions = model(views, inference_track=True)

    pred_tracks = torch.cat([pred["track"] for pred in predictions], dim=0)  # (T, H, W, 3)
    return pred_tracks


def eval_tapvid3d(
    output_dir,
    num_frames,
    args=None,
    model=None,
    device=None,
    data_type="pstudio",
    dyn_static=True,
    save_predictions=False,
    align="sim3",
    data_root="./data",
):
    if save_predictions:
        pred_dir = os.path.join(output_dir, f"saved_predictions_{data_type}")
        os.makedirs(pred_dir, exist_ok=True)

    track_list = glob.glob(f"{data_root}/{data_type}/*.npz")
    track_list = natsorted(track_list)
    if len(track_list) == 0:
        print(f"No .npz files found in {data_root}/{data_type}.")
        return []

    local_results = []
    saved_prediction_paths = []

    for track_npz in tqdm(track_list):
        (
            filelist,
            tracks_xyz_cam,
            tracks_uv,
            intrinsics,
            tracks_xyz_world,
            visibility,
            video_name,
            extrinsics_w2c,
        ) = load_npz_data(track_npz, num_frames=num_frames)

        try:
            pred_tracks = get_inference_output(
                args, model, device, filelist,
                tracks_uv=tracks_uv, visibility=visibility,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            print(f"Warning: OOM on {video_name} at size={args.size}, retrying at size=336")
            pred_tracks = get_inference_output(
                args, model, device, filelist,
                tracks_uv=tracks_uv, visibility=visibility, size=336,
            )
        if save_predictions:
            save_path = save_prediction_as_npy(pred_tracks, pred_dir, video_name)
            saved_prediction_paths.append(save_path)

        pred_tracks = pred_tracks[:num_frames]

        visibility_mask = visibility[0]
        if visibility_mask.sum() == 0:
            print(f"Warning: No visible points in {video_name}")
            continue

        query_uv = np.array(tracks_uv)[0, visibility_mask]
        gt_tracks_filtered = tracks_xyz_world[:num_frames, visibility_mask]

        gt_image_size = filelist[0].size
        pred_image_size = pred_tracks.shape[1:3][::-1]
        query_uv = query_uv * np.array(pred_image_size) / np.array(gt_image_size)

        oob_mask = (
            (query_uv[:, 0] >= 0)
            & (query_uv[:, 0] < pred_image_size[0])
            & (query_uv[:, 1] >= 0)
            & (query_uv[:, 1] < pred_image_size[1])
        )
        if oob_mask.sum() < len(query_uv):
            print(
                f"Warning: {len(query_uv) - oob_mask.sum()} out-of-bounds points in {video_name}, "
                f"{query_uv.max(0)}"
            )
            query_uv = query_uv[oob_mask]
            gt_tracks_filtered = gt_tracks_filtered[:, oob_mask]

        pred_tracks_filtered = pred_tracks[
            :num_frames,
            query_uv[:, 1].astype(int),
            query_uv[:, 0].astype(int),
        ]

        if dyn_static:
            total_motion = gt_tracks_filtered[1:] - gt_tracks_filtered[:-1]
            total_motion_norm = np.linalg.norm(total_motion, axis=-1).sum(0)
            dyn_mask = total_motion_norm > 0.01
            print(
                f"fraction of dynamic points: {dyn_mask.mean()}, "
                f"number of dynamic points: {dyn_mask.sum()}, "
                f"maximum motion: {total_motion_norm.max()}"
            )
        else:
            dyn_mask = None

        nan = float("nan")

        if align == "global":
            avg_pts, _, fractions, _, epe = compute_average_pts_within_thresh(
                gt_tracks_filtered, pred_tracks_filtered,
                scaling="global", intrinsics_params=intrinsics, compute_epe=True)
            avg_pts_dyn, fractions_dyn, epe_dyn = nan, {}, nan
            if dyn_mask is not None and dyn_mask.sum() > 0:
                avg_pts_dyn, _, fractions_dyn, _, epe_dyn = compute_average_pts_within_thresh(
                    gt_tracks_filtered[:, dyn_mask], pred_tracks_filtered[:, dyn_mask],
                    scaling="global", use_fixed_metric_threshold=True,
                    intrinsics_params=intrinsics, pred_aligned=None)

        elif align == "pertraj":
            avg_pts, _, fractions, _, epe = compute_average_pts_within_thresh(
                gt_tracks_filtered, pred_tracks_filtered,
                scaling="per_traj", intrinsics_params=intrinsics, compute_epe=True)
            avg_pts_dyn, fractions_dyn, epe_dyn = nan, {}, nan

        elif align == "sim3":
            avg_pts, _, fractions, _, epe = compute_average_pts_within_thresh(
                gt_tracks_filtered, pred_tracks_filtered,
                scaling="sim3", intrinsics_params=intrinsics, compute_epe=True)
            avg_pts_dyn, fractions_dyn, epe_dyn = nan, {}, nan
            if dyn_mask is not None and dyn_mask.sum() > 0:
                avg_pts_dyn, _, fractions_dyn, _, epe_dyn = compute_average_pts_within_thresh(
                    gt_tracks_filtered[:, dyn_mask], pred_tracks_filtered[:, dyn_mask],
                    scaling="sim3", use_fixed_metric_threshold=True,
                    intrinsics_params=intrinsics, pred_aligned=None)

        elif align == "sim3_closed":
            avg_pts, _, fractions, _, epe = compute_average_pts_within_thresh(
                gt_tracks_filtered, pred_tracks_filtered,
                scaling="sim3_closed", intrinsics_params=intrinsics, compute_epe=True)
            avg_pts_dyn, fractions_dyn, epe_dyn = nan, {}, nan
            if dyn_mask is not None and dyn_mask.sum() > 0:
                avg_pts_dyn, _, fractions_dyn, _, epe_dyn = compute_average_pts_within_thresh(
                    gt_tracks_filtered[:, dyn_mask], pred_tracks_filtered[:, dyn_mask],
                    scaling="sim3_closed", use_fixed_metric_threshold=True,
                    intrinsics_params=intrinsics, pred_aligned=None)

        result_dict = {
            "video_name": video_name,
            "avg_pts": float(avg_pts),
            "avg_pts_dyn": float(avg_pts_dyn),
            "epe": float(epe),
            "epe_dyn": float(epe_dyn),
            "fractions": dict(fractions),
            "fractions_dyn": dict(fractions_dyn),
        }
        local_results.append(result_dict)

        print(f"video_name: {video_name}")
        print(f"avg_pts: {avg_pts:.4f}")
        if not math.isnan(avg_pts_dyn):
            print(f"avg_pts_dyn: {avg_pts_dyn:.4f}")
        print()

        log_path = os.path.join(output_dir, "log.txt")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"video_name: {video_name}\n")
            log_file.write(f"avg_pts ({align}): {avg_pts:.4f}\n")
            if not math.isnan(avg_pts_dyn):
                log_file.write(f"avg_pts_dyn ({align}): {avg_pts_dyn:.4f}\n")
            log_file.write("\n")

    if len(local_results) == 0:
        print(f"Warning: No valid evaluation results for {data_type}")
        return []

    sum_pts, sum_pts_dyn, sum_epe, sum_epe_dyn = 0.0, 0.0, 0.0, 0.0
    cnt, cnt_dyn, cnt_epe = 0, 0, 0
    fraction_aggregator = defaultdict(list)
    fraction_dyn_aggregator = defaultdict(list)

    for res in local_results:
        if not math.isnan(res["avg_pts"]):
            sum_pts += res["avg_pts"]
            cnt += 1
        if not math.isnan(res["avg_pts_dyn"]):
            sum_pts_dyn += res["avg_pts_dyn"]
            cnt_dyn += 1
        if not math.isnan(res["epe"]):
            sum_epe += res["epe"]
            cnt_epe += 1
        if not math.isnan(res["epe_dyn"]):
            sum_epe_dyn += res["epe_dyn"]
        for thr, val in res["fractions"].items():
            fraction_aggregator[thr].append(val)
        for thr, val in res["fractions_dyn"].items():
            fraction_dyn_aggregator[thr].append(val)

    final_avg_pts = sum_pts / cnt if cnt > 0 else nan
    final_avg_pts_dyn = sum_pts_dyn / cnt_dyn if cnt_dyn > 0 else nan
    final_epe = sum_epe / cnt_epe if cnt_epe > 0 else nan
    final_epe_dyn = sum_epe_dyn / cnt_epe if cnt_epe > 0 else nan
    final_fractions = {thr: float(np.mean(vals)) for thr, vals in fraction_aggregator.items()}
    final_fractions_dyn = {thr: float(np.mean(vals)) for thr, vals in fraction_dyn_aggregator.items()}

    log_str = "\n=== Final Results ===\n"
    log_str += f"\nAlign: {align}\n"
    log_str += "\nThreshold-based metrics:\n"
    log_str += f"  avg_pts:     {final_avg_pts:.4f}\n"
    if not math.isnan(final_avg_pts_dyn):
        log_str += f"  avg_pts_dyn: {final_avg_pts_dyn:.4f}\n"
    log_str += "\nEnd Point Error (EPE):\n"
    log_str += f"  epe:     {final_epe:.4f}\n"
    if not math.isnan(final_epe_dyn):
        log_str += f"  epe_dyn: {final_epe_dyn:.4f}\n"
    if final_fractions:
        log_str += "\nDetailed fraction metrics:\n"
        for thr, val in sorted(final_fractions.items()):
            log_str += f"  threshold={thr}: {val:.4f}\n"
    if final_fractions_dyn:
        log_str += "\nDetailed fraction metrics (dyn):\n"
        for thr, val in sorted(final_fractions_dyn.items()):
            log_str += f"  threshold={thr}: {val:.4f}\n"

    print(log_str)

    if output_dir:
        log_file = os.path.join(output_dir, f"track_eval_{data_type}.txt")
        with open(log_file, "w") as f:
            f.write(f"=== Evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(log_str)

    if save_predictions and saved_prediction_paths:
        mapping_file = os.path.join(output_dir, f"prediction_mapping_{data_type}.json")
        mapping = {}
        for npz_path, pred_path in zip(track_list, saved_prediction_paths):
            mapping[npz_path] = {
                "prediction_path": pred_path,
                "video_name": os.path.splitext(os.path.basename(npz_path))[0],
            }
        with open(mapping_file, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved prediction mapping to {mapping_file}")

    return (final_avg_pts, final_avg_pts_dyn, final_epe, final_epe_dyn)
