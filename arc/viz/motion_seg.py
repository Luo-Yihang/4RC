import numpy as np
import torch
from PIL import Image

from arc.dust3r.utils.device import to_numpy

_sam2_models = None
_gdino_models = None
_vla_model = None
_vla_processor = None

_SAM2_WEIGHTS = "facebook/sam2-hiera-large"
_GDINO_WEIGHTS = "IDEA-Research/grounding-dino-tiny"
_VLA_WEIGHTS = "Qwen/Qwen2-VL-2B-Instruct"

VLA_PROMPT = (
    "Look at this image and identify the main foreground dynamic objects. "
    "Include anything that is a moving or potentially moving object (e.g., people, car, dog, robots), "
    "Output only a comma-separated list of specific general object category names "
    "(e.g. 'people in white, car in red, dog', not vague terms like 'moving things' or 'objects'). "
    "Do NOT include static background elements like walls, floor, road, sky, or furniture."
)


def _load_sam2(device: torch.device):
    global _sam2_models
    if _sam2_models is not None:
        return _sam2_models
    from transformers import Sam2Processor, Sam2Model
    processor = Sam2Processor.from_pretrained(_SAM2_WEIGHTS)
    model = Sam2Model.from_pretrained(_SAM2_WEIGHTS).to(device).eval()
    _sam2_models = (processor, model)
    return _sam2_models


def _load_gdino(device: torch.device):
    global _gdino_models
    if _gdino_models is not None:
        return _gdino_models
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    processor = AutoProcessor.from_pretrained(_GDINO_WEIGHTS)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(_GDINO_WEIGHTS).to(device).eval()
    _gdino_models = (processor, model)
    return _gdino_models


def _load_vla(device: torch.device):
    global _vla_model, _vla_processor
    if _vla_model is not None:
        return _vla_model, _vla_processor
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    _vla_processor = AutoProcessor.from_pretrained(_VLA_WEIGHTS)
    _vla_model = Qwen2VLForConditionalGeneration.from_pretrained(
        _VLA_WEIGHTS, torch_dtype=torch.float16
    ).to(device).eval()
    return _vla_model, _vla_processor


@torch.inference_mode()
def _detect_dynamic_objects(img_rgb_uint8: np.ndarray, device: torch.device) -> str:
    vla_model, vla_processor = _load_vla(device)
    image = Image.fromarray(img_rgb_uint8)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": VLA_PROMPT},
            ],
        }
    ]
    from qwen_vl_utils import process_vision_info
    text = vla_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vla_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(device)
    generated_ids = vla_model.generate(**inputs, max_new_tokens=64)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return vla_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


@torch.inference_mode()
def _segment_text_sam2(img_rgb_uint8: np.ndarray, detected_text: str, device: torch.device):
    H, W = img_rgb_uint8.shape[:2]
    image = Image.fromarray(img_rgb_uint8)

    # Grounding DINO: text → boxes
    gdino_processor, gdino_model = _load_gdino(device)
    labels = [p.strip() for p in detected_text.split(",") if p.strip()]
    gdino_text = " . ".join(labels) + " ."
    gdino_inputs = gdino_processor(images=image, text=gdino_text, return_tensors="pt").to(device)
    gdino_outputs = gdino_model(**gdino_inputs)
    detections = gdino_processor.post_process_grounded_object_detection(
        gdino_outputs,
        gdino_inputs.input_ids,
        threshold=0.3,
        text_threshold=0.25,
        target_sizes=[(H, W)],
    )[0]
    boxes = detections["boxes"]  # [N, 4] xyxy, pixel coords
    if len(boxes) == 0:
        return None

    # SAM2: boxes → masks
    sam2_processor, sam2_model = _load_sam2(device)
    sam2_inputs = sam2_processor(
        images=image,
        input_boxes=[boxes.cpu().tolist()],
        return_tensors="pt",
    ).to(device)
    sam2_outputs = sam2_model(**sam2_inputs)
    masks = sam2_processor.post_process_masks(
        sam2_outputs.pred_masks.cpu(),
        sam2_inputs["original_sizes"],
    )  # list of [N_boxes, 3, H, W]

    if not masks or masks[0].numel() == 0:
        return None

    # Pick best mask per box (highest IoU score) and union
    iou_scores = sam2_outputs.iou_scores  # [1, N_boxes, 3]
    best_idx = iou_scores[0].argmax(dim=-1)  # [N_boxes]
    m = masks[0]  # [N_boxes, 3, H, W]
    combined = torch.zeros(H, W, dtype=torch.bool)
    for i, idx in enumerate(best_idx):
        combined |= m[i, idx].bool().cpu()

    return combined.numpy().astype(np.uint8)


def build_dynamic_mask(img_rgb_uint8: np.ndarray, device: torch.device):
    detected_text = _detect_dynamic_objects(img_rgb_uint8, device)
    if not detected_text:
        return None, detected_text

    combined_mask = _segment_text_sam2(img_rgb_uint8, detected_text, device)
    if combined_mask is None:
        return None, detected_text

    h, w = combined_mask.shape[:2]
    mh = max(1, int(round(h * 0.03)))
    mw = max(1, int(round(w * 0.03)))
    combined_mask[:mh, :] = False
    combined_mask[-mh:, :] = False
    combined_mask[:, :mw] = False
    combined_mask[:, -mw:] = False
    return combined_mask.astype(np.uint8), detected_text


def prepare_refine_mask(output: dict, device: torch.device, refine_track_visual: bool = True) -> None:
    if not refine_track_visual:
        return
    existing = output.get("track_text_mask")
    if existing is not None and (
        not isinstance(existing, (list, tuple)) or any(m is not None for m in existing)
    ):
        return

    def _normalize_idx(val, n):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().flatten().tolist()
        elif not isinstance(val, (list, tuple)):
            val = [int(val)]
        val = [int(i) for i in val if 0 <= int(i) < n]
        return val[0] if val else 0

    num_frames = len(output["preds"])
    track_query_idx = _normalize_idx(output["preds"][0].get("track_query_idx", 0), num_frames)

    query_img_np = to_numpy(output["views"][track_query_idx]["img"].cpu().squeeze().permute(1, 2, 0))
    query_img_uint8 = ((query_img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)

    mask, detected_text = build_dynamic_mask(query_img_uint8, device)

    output["track_dynamic_objects_text"] = detected_text
    output["track_query_img_uint8"] = query_img_uint8
    if mask is not None:
        output["track_text_mask"] = [mask]
