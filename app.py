# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
4RC Demo
==============
Upload multiple unordered images of a scene for 3D reconstruction and tracking.
"""

import os
import time
import glob
import shutil
import traceback
import base64
import torch
import gradio as gr
import multiprocessing as mp
from rich.console import Console
import argparse
import json
import numpy as np

from arc.dust3r.inference_multiview import inference
from arc.dust3r.utils.image import load_images
from arc.viz.viser_visualizer_track import start_visualization, prepare_refine_mask
from arc.viz.video_utils import extract_frames_from_video

global_manager_req_queue = None
global_manager_resp_queue = None

# -------------------------------
# run_viser_server
# -------------------------------
def run_viser_server(pipe_conn, output, min_conf_thr_percentile, point_size=0.0004):
    """
    Launches the visualization server and sends its share URL.
    """
    try:
        server = start_visualization(
            output=output,
            min_conf_thr_percentile=min_conf_thr_percentile,
            point_size=point_size,
        )
        # share_url = server.request_share_url() # currently we use gcp and do not use share url
        share_url = f"http://127.0.0.1:{server.get_port()}"
        pipe_conn.send({"share_url": share_url})
        pipe_conn.close()
        while True:
            time.sleep(3600)
    except Exception as e:
        try:
            pipe_conn.send({"error": str(e), "traceback": traceback.format_exc()})
        except Exception:
            pass
        pipe_conn.close()


# -------------------------------
# ViserServerManager
# -------------------------------
class ViserServerManager:
    """
    Manages visualization servers launched as separate processes.
    """
    def __init__(self, req_queue, resp_queue):
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self.servers = {}  # server_id -> server info
        self.session_servers = {}  # session_id -> list of server_ids
        self.console = Console()
        self.next_server_id = 1

    def run(self):
        self.console.log("[bold green]ViserServerManager started[/bold green]")
        while True:
            try:
                cmd = self.req_queue.get(timeout=1)
            except Exception:
                continue

            # Extract message_id if present
            message_id = cmd.get("message_id")
            
            if cmd["cmd"] == "launch":
                server_id = self.next_server_id
                self.next_server_id += 1
                session_id = cmd.get("session_id", f"default_session_{server_id}")
                
                self.console.log(f"Launching viser server with id {server_id} for session {session_id} (message_id: {message_id})")
                try:
                    output = cmd["output"]
                    parent_conn, child_conn = mp.Pipe()
                    p = mp.Process(
                        target=run_viser_server,
                        args=(
                            child_conn,
                            output,
                            cmd.get("min_conf_thr_percentile", 2.5),
                            cmd.get("point_size", 0.0016),
                        )
                    )
                    p.start()
                    child_conn.close()
                    result = parent_conn.recv()
                    if "error" in result:
                        self.console.log(f"[red]Error launching server: {result.get('error')}[/red]")
                        if result.get("traceback"):
                            self.console.log(result["traceback"])
                        self.resp_queue.put({
                            "cmd": "launch", 
                            "error": result["error"],
                            "message_id": message_id
                        })
                        p.terminate()
                        p.join(timeout=5)
                    else:
                        share_url = result["share_url"]
                        self.servers[server_id] = {"share_url": share_url, "process": p, "session_id": session_id}
                        
                        # Track which servers belong to which session
                        if session_id not in self.session_servers:
                            self.session_servers[session_id] = []
                        self.session_servers[session_id].append(server_id)
                        
                        self.console.log(f"Server {server_id} launched with URL {share_url} (pid: {p.pid}) for session {session_id}")
                        self.resp_queue.put({
                            "cmd": "launch", 
                            "server_id": server_id, 
                            "session_id": session_id,
                            "share_url": share_url,
                            "message_id": message_id
                        })
                except Exception as e:
                    self.console.log(f"[red]Error launching server: {e}[/red]")
                    self.resp_queue.put({
                        "cmd": "launch", 
                        "error": str(e),
                        "message_id": message_id
                    })

            elif cmd["cmd"] == "terminate_server":
                server_id = cmd["server_id"]
                if server_id in self.servers:
                    process = self.servers[server_id].get("process")
                    session_id = self.servers[server_id].get("session_id")
                    self.console.log(f"Terminating server with id {server_id} (pid: {process.pid if process else 'N/A'}) from session {session_id}")
                    try:
                        if process is not None:
                            process.kill()
                            process.join(timeout=10)
                        
                        # Remove from session tracking
                        if session_id in self.session_servers and server_id in self.session_servers[session_id]:
                            self.session_servers[session_id].remove(server_id)
                            if not self.session_servers[session_id]:  # If empty list
                                del self.session_servers[session_id]
                        
                        del self.servers[server_id]
                        self.resp_queue.put({
                            "cmd": "terminate_server", 
                            "server_id": server_id, 
                            "status": "terminated",
                            "message_id": message_id
                        })
                    except Exception as e:
                        self.console.log(f"[red]Error terminating server {server_id}: {e}[/red]")
                        self.resp_queue.put({
                            "cmd": "terminate_server", 
                            "server_id": server_id, 
                            "error": str(e),
                            "message_id": message_id
                        })
                else:
                    self.console.log(f"[red]Server with id {server_id} not found[/red]")
                    self.resp_queue.put({
                        "cmd": "terminate_server", 
                        "server_id": server_id, 
                        "error": "ID not found",
                        "message_id": message_id
                    })
            
            elif cmd["cmd"] == "terminate_session":
                session_id = cmd["session_id"]
                if session_id in self.session_servers:
                    server_ids = self.session_servers[session_id].copy()  # Copy to avoid modification during iteration
                    self.console.log(f"Terminating all servers for session {session_id}: {server_ids}")
                    
                    terminated_servers = []
                    errors = []
                    
                    for server_id in server_ids:
                        if server_id in self.servers:
                            process = self.servers[server_id].get("process")
                            try:
                                if process is not None:
                                    process.kill()
                                    process.join(timeout=10)
                                del self.servers[server_id]
                                terminated_servers.append(server_id)
                            except Exception as e:
                                errors.append(f"Error terminating server {server_id}: {e}")
                    
                    # Clean up session tracking
                    if session_id in self.session_servers:
                        del self.session_servers[session_id]
                    
                    self.resp_queue.put({
                        "cmd": "terminate_session", 
                        "session_id": session_id, 
                        "terminated_servers": terminated_servers,
                        "errors": errors,
                        "status": "terminated" if not errors else "partial_termination",
                        "message_id": message_id
                    })
                else:
                    self.console.log(f"[red]No servers found for session {session_id}[/red]")
                    self.resp_queue.put({
                        "cmd": "terminate_session", 
                        "session_id": session_id, 
                        "error": "Session ID not found",
                        "message_id": message_id
                    })
            else:
                self.console.log(f"Unknown command: {cmd}")
                self.resp_queue.put({
                    "cmd": "error", 
                    "error": "Unknown command",
                    "message_id": message_id
                })


# -------------------------------
# start_manager
# -------------------------------
def start_manager():
    """
    Starts the ViserServerManager.
    
    Returns: req_queue, resp_queue, manager_process.
    """
    req_queue = mp.Queue()
    resp_queue = mp.Queue()
    manager_process = mp.Process(target=ViserServerManager(req_queue, resp_queue).run)
    manager_process.start()
    return req_queue, resp_queue, manager_process


# -------------------------------
# load_model
# -------------------------------
def load_model(checkpoint_dir, device: torch.device):
    """
    Loads the model from the checkpoint.

    Returns: model.
    """
    from arc.models.arc import Arc

    model = Arc.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    return model


# -------------------------------
# update_gallery
# -------------------------------
def update_gallery(files):
    """
    Returns a list of file paths for gallery preview.
    """
    files.sort()
    if files is None:
        return []
    preview = []
    for f in files:
        if isinstance(f, str):
            preview.append(f)
        elif isinstance(f, dict) and "data" in f:
            preview.append(f["data"])
    return preview


def parse_track_query_idx(track_query_idx, num_frames):
    if track_query_idx is None:
        return [num_frames // 2]
    if isinstance(track_query_idx, (list, tuple, np.ndarray)):
        query_list = [int(x) for x in track_query_idx]
    elif isinstance(track_query_idx, str):
        cleaned = track_query_idx.replace(";", ",").replace("|", ",").replace(" ", ",")
        parts = [p for p in cleaned.split(",") if p.strip()]
        query_list = [int(p) for p in parts]
    else:
        query_list = [int(track_query_idx)]

    query_list = [idx for idx in query_list if 0 <= idx < num_frames]
    if not query_list:
        query_list = [0]
    return query_list


# -------------------------------
# process_images
# -------------------------------
def process_images(uploaded_files, video_file, state,
                   model, device,
                   global_manager_req_queue, global_manager_resp_queue,
                   output_dir, examples_dir,
                   image_size=512, rotate_clockwise_90=False, crop_to_landscape=False,
                   track_query_idx=None, refine_track_visual=False):
    """
    Processes input images/video:
      - Saves files to the output directory (unless it's an example)
      - Runs model inference.
      - Launches the visualization server.
    
    This function yields a consistent 3-tuple on every update:
      1. status_html: HTML (placeholder / loading / hidden)
      2. result_html: HTML (hidden / vis info + iframe)
      3. state: State
    """
    if not uploaded_files:
        yield (
            gr.update(value=(
                '<div class="viewer-placeholder">'
                '<div class="ph-icon">⚠️</div>'
                '<span>No input detected</span>'
                '<span class="ph-sub">Please use the panel on the left to upload images or a video first</span>'
                '</div>'
            ), visible=True),
            gr.update(visible=False),
            state
        )
        return
    
    start_total = time.time()

    # Create timestamp for the current session
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S-") + str(int(time.time() * 1000))
    state["current_timestamp"] = timestamp    

    # Save files to output directory (unless it's an example)
    is_example = state.get("is_example", False)
    filelist = []
    
    if not is_example and uploaded_files:
        # Create session directories
        save_dir = os.path.join(output_dir, timestamp)
        img_dir = os.path.join(save_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        
        # Save all files
        for i, file_obj in enumerate(uploaded_files):
            src_path = file_obj[0]
            dst_path = os.path.join(img_dir, f'image_{i}.jpg')
            shutil.copy2(src_path, dst_path)
            filelist.append(dst_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'num_images': len(filelist),
            'source_type': 'video' if video_file else 'images'
        }
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    else:
        # For examples, just use the original file paths but still save metadata
        filelist = [file_obj[0] if isinstance(file_obj, tuple) else file_obj 
                   for file_obj in uploaded_files]
        
        # Create directory for example scene metadata
        example_dir = os.path.join(output_dir, "example_scenes", timestamp)
        os.makedirs(example_dir, exist_ok=True)
        
        # Save metadata for the example scene
        example_metadata = {
            'timestamp': timestamp,
            'num_images': len(filelist),
            'source_type': 'example',
            'example_name': os.path.basename(video_file) if video_file else None
        }
        with open(os.path.join(example_dir, 'metadata.json'), 'w') as f:
            json.dump(example_metadata, f, indent=2)
    
    loading_html = """
    <div class="loading-box">
        <div class="loading-title">Reconstructing your scene...</div>
        <div class="loading-subtitle">Running inference</div>
        <div>
            <span class="loading-emoji">🔮</span>
            <span class="loading-emoji">🌐</span>
            <span class="loading-emoji">✨</span>
        </div>
        <p>This may take a moment depending on input size</p>
    </div>
    """
    yield (
        gr.update(value=loading_html, visible=True),
        gr.update(visible=False),
        state,
    )
    
    end_image = time.time()
    image_prep_time = end_image - start_total
    
    # Load and resize images
    start_load_time = time.time()

    imgs = load_images(
        filelist,
        size=image_size,
        verbose=True,
        rotate_clockwise_90=rotate_clockwise_90,
        crop_to_landscape=crop_to_landscape,
        patch_size=14,
    )

    def evenly_spaced_elements(lst, num_elements):
        if num_elements >= len(lst):
            return lst
        indices = np.linspace(0, len(lst) - 1, num_elements, dtype=int)
        return [lst[i] for i in indices]

    if video_file:
        imgs = evenly_spaced_elements(imgs, 30)

    track_query_idx_list = parse_track_query_idx(track_query_idx, len(imgs))
    track_query_idx_tensor = torch.tensor(track_query_idx_list)
    for img in imgs:
        img["track_query_idx"] = track_query_idx_tensor

    end_load_time = time.time()
    load_time = end_load_time - start_load_time
    print(f"Image loading and cropping time: {load_time:.2f} seconds")
    
    yield (
        gr.update(),
        gr.update(),
        state,
    )
    
    # Run inference directly
    output_dict, profiling_info = inference(
        imgs,
        model,
        device,
        dtype="bf16-mixed",
        verbose=True,
        profiling=True,
        use_center_as_anchor=False
    )
    model_forward_time = profiling_info['total_time']

    output_dict["refine_track_visual"] = bool(refine_track_visual)

    # Must run in main process (CUDA available) before the viser subprocess is forked.
    prepare_refine_mask(output_dict, device, refine_track_visual=bool(refine_track_visual))

    rendering_html = f"""
    <div class="loading-box">
        <div class="loading-title">Preparing 3D visualization...</div>
        <div class="loading-subtitle">Inference done in {model_forward_time:.1f}s, now rendering</div>
        <div>
            <span class="loading-emoji">🔮</span>
            <span class="loading-emoji">🌐</span>
            <span class="loading-emoji">✨</span>
        </div>
        <p>{len(imgs)} frames processed</p>
    </div>
    """
    yield (
        gr.update(value=rendering_html),
        gr.update(),
        state,
    )
    
    # Process predictions and move tensors to CPU.
    try:
        for pred in output_dict['preds']:
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    pred[k] = v.cpu()
        for view in output_dict['views']:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.cpu()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: {e}")
    
    start_vis_prep = time.time()
    
    # Generate a unique message ID for this request
    message_id = f"msg_{int(time.time()*1000)}_{os.getpid()}_{id(state)}"
    
    # Launch the visualization server with message_id
    session_id = state.get("session_id")
    if not session_id:  # if no session_id, create a new one
        session_id = f"session_{int(time.time()*1000)}"
        state["session_id"] = session_id
    
    cmd = {
        "cmd": "launch",
        "output": output_dict,
        "min_conf_thr_percentile": 2.5,
        "point_size": 0.0016,
        "session_id": session_id,
        "message_id": message_id
    }
    global_manager_req_queue.put(cmd)
    
    # Wait for server response with dynamic loading, but only accept responses with matching message_id
    loading_dots = [".", "..", "..."]
    loading_idx = 0
    start_wait = time.time()
    timeout = 600
    while True:
        # Check for timeout first
        if time.time() - start_wait > timeout:
            raise gr.Error("Timeout waiting for visualization server. Please try again.")
        
        try:
            resp = global_manager_resp_queue.get_nowait()
            # Only accept responses with matching message_id
            if resp.get("message_id") == message_id:
                break
            else:
                # Put back responses meant for other sessions
                global_manager_resp_queue.put(resp)
                time.sleep(0.1)  # Small delay to avoid CPU spinning
        except:
            loading_idx = (loading_idx + 1) % len(loading_dots)
            yield (
                gr.update(),
                gr.update(),
                state
            )
            time.sleep(0.3)
    
    if "error" in resp:
        share_url = f"ERROR: {resp['error']}"
    else:
        share_url = resp["share_url"]
    
    end_vis_prep = time.time()
    vis_prep_time = end_vis_prep - start_vis_prep
    total_time = end_vis_prep - start_total
    
    # Store the server_id from the response if available
    server_id = resp.get("server_id")
    state["urls"].append((share_url, server_id))

    result_html = (
        '<div style="display:flex; align-items:center; justify-content:space-between; '
        'padding:8px 14px; font-size:12px; opacity:0.55; border-bottom:1px solid rgba(255,255,255,0.06);">'
        f'<span>{len(imgs)} frames &middot; Inference: {model_forward_time:.1f}s &middot; Rendering: {vis_prep_time:.1f}s &middot; Total: {total_time:.1f}s</span>'
        '<span>Drag to rotate &middot; Right-click to pan &middot; Scroll to zoom</span>'
        '</div>'
        f"<iframe src='{share_url}' width='100%' frameborder='0' "
        "style='border-radius: 10px 10px 10px 10px; border: none; height: 75vh; min-height: 500px;'></iframe>"
    )

    yield (
        gr.update(visible=False),
        gr.update(value=result_html, visible=True),
        state
    )

# -------------------------------
# delete_visers_callback
# -------------------------------
def delete_visers_callback(state):
    """
    Cleans up visualization servers when the session ends.
    """
    session_id = state.get("session_id")
    if session_id:
        try:
            # Generate a unique message ID for this request
            message_id = f"term_{int(time.time()*1000)}_{os.getpid()}_{id(state)}"
            
            # Terminate all servers for this session
            term_cmd = {
                "cmd": "terminate_session", 
                "session_id": session_id,
                "message_id": message_id
            }
            global_manager_req_queue.put(term_cmd)
            
            # Wait for response with matching message_id
            start_wait = time.time()
            timeout = 600
            while True:
                # Check for timeout first
                if time.time() - start_wait > timeout:
                    print(f"Timeout waiting for termination response for session {session_id}")
                    break
                
                try:
                    resp = global_manager_resp_queue.get(timeout=1)
                    if resp.get("message_id") == message_id:
                        print(f"Terminated servers for session {session_id}, Response: {resp}")
                        break
                    else:
                        # Put back responses meant for other sessions
                        global_manager_resp_queue.put(resp)
                except Exception:
                    # Just continue the loop if queue is empty
                    continue
            
        except Exception as e:
            print(f"Error terminating servers for session {session_id}: {e}")
    print(f"All viser servers for session {session_id} cleaned up.")


# -------------------------------
# create_demo
# -------------------------------
EXAMPLES_DIR = "./examples"
_EXAMPLE_ORDER = ["robot_jump", "robot_arm", "exercise", "car_turn", "street", "hockey", "robot_dance", "static_room"]


def create_demo(checkpoint_dir, output_dir, device: torch.device):
    """
    Creates the Gradio demo interface.

    Layout: Left narrow panel (inputs + submit + examples) | Right wide panel (3D visualization)
    """
    global global_manager_req_queue, global_manager_resp_queue

    global_manager_req_queue, global_manager_resp_queue, manager_process = start_manager()

    model = load_model(checkpoint_dir, device=device)

    examples_dir = EXAMPLES_DIR
    examples_abs_dir = os.path.abspath(examples_dir) if os.path.isdir(examples_dir) else None

    # Collect per-example image lists ordered by _EXAMPLE_ORDER
    _subdir_map = {}
    if examples_abs_dir:
        for subdir in os.listdir(examples_abs_dir):
            subdir_path = os.path.join(examples_abs_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            paths = sorted(
                glob.glob(os.path.join(subdir_path, "*.jpg")) +
                glob.glob(os.path.join(subdir_path, "*.png"))
            )
            if paths:
                _subdir_map[subdir] = paths

    def _order_key(name):
        nl = name.lower()
        for i, kw in enumerate(_EXAMPLE_ORDER):
            if kw in nl:
                return i
        return len(_EXAMPLE_ORDER)

    example_all_paths = [
        _subdir_map[k] for k in sorted(_subdir_map, key=_order_key)
    ]

    video_input = gr.Video(label="Upload a Video", sources=["upload"])

    global_css = """
    /* ---- Loading animation ---- */
    .loading-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        color: #e0e0e0;
        padding: 50px 30px;
        border-radius: 16px;
        margin: 80px auto;
        text-align: center;
        max-width: 380px;
    }
    .loading-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
        animation: pulse 2s ease-in-out infinite;
    }
    .loading-subtitle {
        font-size: 13px;
        margin: 4px 0;
        opacity: 0.5;
    }
    .loading-subtitle::after {
        content: '';
        animation: dots 2s steps(1, end) infinite;
    }
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80% { content: '...'; }
    }
    .loading-emoji {
        font-size: 18px;
        margin: 16px 6px 6px;
        display: inline-block;
        animation: bounce 1s infinite;
    }
    .loading-emoji:nth-child(2) { animation-delay: 0.15s; }
    .loading-emoji:nth-child(3) { animation-delay: 0.3s; }
    .loading-box p { font-size: 12px; opacity: 0.4; margin-top: 8px; }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* ---- Right panel placeholder (NOT an upload zone) ---- */
    .viewer-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 80vh;
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        color: rgba(255,255,255,0.18);
        font-size: 14px;
        gap: 6px;
        user-select: none;
    }
    .viewer-placeholder .ph-icon {
        font-size: 36px;
        margin-bottom: 4px;
        opacity: 0.4;
    }
    .viewer-placeholder .ph-sub {
        font-size: 12px;
        opacity: 0.5;
    }

    /* ---- Header banner ---- */
    .arc-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 24px;
        margin: -16px -16px 16px -16px;
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .arc-banner-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .arc-banner-logo {
        height: 40px;
        width: auto;
        object-fit: contain;
        vertical-align: middle;
    }
    .arc-banner-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .arc-banner-sep {
        width: 1px;
        height: 20px;
        background: rgba(255,255,255,0.12);
    }
    .arc-banner-tagline {
        font-size: 13px;
        opacity: 0.5;
        font-weight: 400;
    }
    .arc-banner-links {
        display: flex;
        gap: 8px;
    }
    .arc-banner-links a {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 7px 14px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        text-decoration: none;
        color: #d0d0d0;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.15s ease;
    }
    .arc-banner-links a:hover {
        background: rgba(255,255,255,0.12);
        color: #fff;
        border-color: rgba(255,255,255,0.15);
    }

    /* ---- Header banner: responsive ---- */
    @media (max-width: 640px) {
        .arc-banner {
            flex-direction: column;
            align-items: flex-start;
            gap: 12px;
            padding: 14px 16px;
        }
        .arc-banner-tagline {
            display: none;
        }
        .arc-banner-sep {
            display: none;
        }
        .arc-banner-links {
            flex-wrap: wrap;
        }
        .arc-banner-links a {
            font-size: 12px;
            padding: 6px 10px;
        }
        .arc-banner-title {
            font-size: 22px;
        }
        .arc-banner-logo {
            height: 32px;
        }
    }
    @media (max-width: 420px) {
        .arc-banner-links a span {
            display: none;
        }
    }

    /* ---- Step label ---- */
    .step-label {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 4px 0 8px 0;
        font-size: 13px;
        font-weight: 600;
        opacity: 0.6;
    }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px; height: 20px;
        border-radius: 50%;
        background: rgba(96,165,250,0.15);
        color: #60a5fa;
        font-size: 11px;
        font-weight: 700;
    }
    .or-divider {
        text-align: center;
        font-size: 11px;
        opacity: 0.3;
        margin: 2px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ---- Left column: no clipping ---- */
    .left-panel-col {
        overflow: visible !important;
    }

    /* ---- Gallery: single row, horizontal scroll only ---- */
    .gradio-gallery .thumbnails {
        overflow-y: hidden !important;
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
    .gradio-gallery .thumbnail-item {
        flex-shrink: 0 !important;
    }

    /* ---- Gallery & Video drop highlight ---- */
    .gallery-drop-highlight .upload-container,
    .gallery-drop-highlight .thumbnails {
        transition: outline 0.15s ease, background 0.15s ease;
    }
    /* Gradio uses [data-testid] wrappers; target the upload area broadly */
    .gradio-gallery .upload-area:hover,
    .gradio-gallery .upload-area:focus-within {
        outline: 2px solid rgba(96,165,250,0.5) !important;
        outline-offset: -2px;
    }

    /* ---- Right panel: content top-aligned ---- */
    .right-panel {
        align-self: flex-start !important;
    }

    /* ---- Footer ---- */
    footer { display: none !important; }

    /* ---- Examples: horizontal row ---- */
    #arc-examples .dataset table {
        display: flex !important;
    }
    #arc-examples .dataset thead {
        display: none !important;
    }
    #arc-examples .dataset tbody {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
    }
    #arc-examples .dataset tbody tr {
        display: block !important;
        cursor: pointer;
    }
    #arc-examples .dataset tbody tr td {
        display: block !important;
        padding: 0 !important;
    }
    #arc-examples .dataset tbody tr td img {
        width: 100px !important;
        height: 72px !important;
        object-fit: cover !important;
        border-radius: 6px !important;
        display: block !important;
    }

    """

    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        logo_tag = f'<img src="data:image/png;base64,{logo_b64}" class="arc-banner-logo" alt="4RC">'
    else:
        logo_tag = '<span class="arc-banner-title">4RC</span>'

    header_html = f"""
    <div class="arc-banner">
        <div class="arc-banner-left">
            {logo_tag}
            <span class="arc-banner-sep"></span>
            <span class="arc-banner-tagline">4D Reconstruction via Conditional Querying Anytime and Anywhere</span>
        </div>
        <div class="arc-banner-links">
            <a href="https://yihangluo.com/projects/4RC/" target="_blank">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
                Project Page
            </a>
            <a href="https://arxiv.org/abs/2602.10094" target="_blank">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>
                arXiv Paper
            </a>
        </div>
    </div>
    """

    placeholder_html = """
    <div class="viewer-placeholder">
        <div class="ph-icon">🎬</div>
        <span>3D visualization will appear here</span>
        <span class="ph-sub">Upload input on the left and click Submit to start</span>
    </div>
    """

    with gr.Blocks(title="4RC Demo", css=global_css, theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )) as demo:
        state = gr.State(value={"session_id": "", "urls": []}, delete_callback=delete_visers_callback)

        gr.HTML(header_html)

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="step-label"><span class="step-num">1</span> Upload images or a video</div>')
                gallery = gr.Gallery(
                    label="Drop images here (click or drag)",
                    columns=4,
                    rows=1,
                    height=260,
                    object_fit="cover",
                    interactive=True,
                    elem_classes=["gradio-gallery"],
                    buttons=["download", "fullscreen"],
                    sources=["upload", "clipboard"]
                )
                gr.HTML('<div class="or-divider">— or —</div>')
                video_input.render()

                gr.HTML('<div class="step-label" style="margin-top:12px;"><span class="step-num">2</span> Run reconstruction</div>')

                refine_track_toggle = gr.Checkbox(
                    label="Refine Track Visualization",
                    value=False,
                    info="Use VLA + SAM2 to auto-segment dynamic objects and filter their trajectories for better visulization.",
                    elem_classes=["refine-track-toggle"],
                )

                submit_button = gr.Button("Submit", variant="primary", size="lg")

                gr.HTML('<div class="step-label" style="margin-top:16px;">Examples</div>')
                gr.Examples(
                    examples=[[paths] for paths in example_all_paths],
                    inputs=gallery,
                    label="",
                    examples_per_page=4,
                    elem_id="arc-examples",
                )

            with gr.Column(scale=3, elem_classes=["right-panel"]):
                status_html = gr.HTML(value=placeholder_html)
                result_html = gr.HTML(visible=False)

        just_uploaded_video = gr.State(value=True)

        def update_gallery_upload(gallery_images, video, just_uploaded_video, state):
            if gallery_images:
                # Detect if images come from the examples directory
                first = gallery_images[0]
                first_path = first[0] if isinstance(first, (tuple, list)) else str(first)
                if examples_abs_dir and os.path.abspath(str(first_path)).startswith(examples_abs_dir):
                    state = {**state, "is_example": True}
                else:
                    state = {**state, "is_example": False}
                if video and just_uploaded_video:
                    return video, False, state
                else:
                    return None, just_uploaded_video, state
            else:
                return None, just_uploaded_video, state

        def update_video_upload(gallery_images, video, just_uploaded_video, state):
            if video:
                state["is_example"] = False
                temp_dir = os.path.join("temp_preview_frames", f"preview_{int(time.time()*1000)}")
                os.makedirs(temp_dir, exist_ok=True)
                frame_paths = extract_frames_from_video(video, temp_dir)
                return frame_paths, True, state
            else:
                state["is_example"] = False
                return [], False, state

        gallery.change(fn=update_gallery_upload, inputs=[gallery, video_input, just_uploaded_video, state],
                       outputs=[video_input, just_uploaded_video, state])
        video_input.change(fn=update_video_upload, 
                          inputs=[gallery, video_input, just_uploaded_video, state],
                          outputs=[gallery, just_uploaded_video, state])

        def process_images_wrapper(uploaded_files, video_file, state, refine_track_visual):
            if not state.get("is_example", False):
                state = state.copy()
                state["is_example"] = False

            generator = process_images(uploaded_files, video_file, state,
                                       model, device,
                                       global_manager_req_queue, global_manager_resp_queue,
                                       output_dir, examples_dir,
                                       refine_track_visual=refine_track_visual)

            for output in generator:
                yield output

        submit_button.click(
            fn=process_images_wrapper,
            inputs=[gallery, video_input, state, refine_track_toggle],
            outputs=[status_html, result_html, state],
        )

    return demo


def main():
    os.environ["GRADIO_TEMP_DIR"] = "./gradio_tmp_dir"
    parser = argparse.ArgumentParser(description="4RC Demo")
    parser.add_argument("--checkpoint_dir", type=str, default="Luo-Yihang/4RC")
    parser.add_argument("--output_dir", type=str, default="./demo_outputs",
                        help="Directory to store processed scenes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo = create_demo(args.checkpoint_dir, args.output_dir, device=device)
    demo.queue(default_concurrency_limit=2)
    demo.launch(share=False)


if __name__ == "__main__":
    main()
