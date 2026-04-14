import argparse
import glob
import os
import time

import numpy as np
import torch

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(input_path):
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTS:
            from arc.viz.video_utils import extract_frames_from_video
            tmp_dir = f"/tmp/4rc_frames_{int(time.time() * 1000)}"
            os.makedirs(tmp_dir, exist_ok=True)
            paths = extract_frames_from_video(input_path, tmp_dir)
            return paths, True
        return [input_path], False
    if os.path.isdir(input_path):
        paths = sorted(
            p for p in glob.glob(os.path.join(input_path, "*"))
            if os.path.splitext(p)[1].lower() in IMAGE_EXTS
        )
        return paths, False
    raise ValueError(f"Input not found: {input_path}")


def save_npz(output_dict, path):
    flat = {"n_frames": np.array(len(output_dict["preds"]))}

    for i, pred in enumerate(output_dict["preds"]):
        for k, v in pred.items():
            arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            if isinstance(arr, np.ndarray):
                flat[f"pred_{i}_{k}"] = arr

    for i, view in enumerate(output_dict["views"]):
        for k, v in view.items():
            arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            if isinstance(arr, np.ndarray):
                flat[f"view_{i}_{k}"] = arr

    if output_dict.get("track_text_mask"):
        masks = output_dict["track_text_mask"]
        flat["track_text_mask_n"] = np.array(len(masks))
        for i, m in enumerate(masks):
            if m is not None:
                flat[f"track_text_mask_{i}"] = m

    for k in ("track_dynamic_objects_text", "track_query_img_uint8"):
        if k in output_dict and output_dict[k] is not None:
            flat[k] = np.array(output_dict[k])

    flat["refine_track_visual"] = np.array(bool(output_dict.get("refine_track_visual", False)))

    np.savez_compressed(path, **flat)
    print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image folder, single image, or video file")
    parser.add_argument("--save", required=True, help="Output .npz path")
    parser.add_argument("--checkpoint_dir", default="Luo-Yihang/4RC")
    parser.add_argument("--track_query_idx", type=int, default=-1, help="Frame index for tracking query; -1 = middle frame")
    parser.add_argument("--refine_track_visualization", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, is_video = collect_images(args.input)
    if is_video or len(paths) > 30:
        indices = np.linspace(0, len(paths) - 1, 30, dtype=int)
        paths = [paths[i] for i in indices]
    print(f"Processing {len(paths)} frames...")

    from arc.models.arc import Arc
    from arc.dust3r.inference_multiview import inference
    from arc.dust3r.utils.image import load_images

    model = Arc.from_pretrained(args.checkpoint_dir).to(device).eval()
    imgs = load_images(paths, size=512, verbose=True, patch_size=14)

    track_query_idx = args.track_query_idx if args.track_query_idx >= 0 else len(imgs) // 2
    q = [track_query_idx] if track_query_idx < len(imgs) else [len(imgs) // 2]
    for img in imgs:
        img["track_query_idx"] = torch.tensor(q)

    output_dict, profiling = inference(
        imgs, model, device, dtype="bf16-mixed", verbose=True, profiling=True, use_center_as_anchor=False
    )
    print(f"Inference: {profiling['total_time']:.2f}s")

    for pred in output_dict["preds"]:
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                pred[k] = v.cpu()
    for view in output_dict["views"]:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_dict["refine_track_visual"] = args.refine_track_visualization
    if args.refine_track_visualization:
        from arc.viz.motion_seg import prepare_refine_mask
        prepare_refine_mask(output_dict, device, refine_track_visual=True)

    save_npz(output_dict, args.save)


if __name__ == "__main__":
    main()
