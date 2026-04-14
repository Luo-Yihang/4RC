# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Video utilities for visualization.
"""

import os
import cv2
import numpy as np
import subprocess

def extract_frames_from_video(video_path, output_dir):
    """
    Extracts frames from a video (1 FPS) and saves them as JPEG files.
    Handles regular video files, webcam captures, and webm files, including truncated files.
    
    Returns: List of file paths.
    """
    saved_frames = []
    
    # For WebM files, use FFmpeg directly which is more robust
    if video_path.lower().endswith('.webm'):
        try:
            print(f"Processing WebM file using FFmpeg: {video_path}")
            
            # Create a unique output pattern for the frames
            frame_pattern = os.path.join(output_dir, "frame_%04d.jpg")
            
            # Use FFmpeg to extract frames at 1 FPS
            ffmpeg_cmd = [
                "ffmpeg", 
                "-i", video_path,
                # "-vf", "fps=1",  # 1 frame per second
                "-q:v", "2",     # High quality
                frame_pattern
            ]
            
            # Run FFmpeg process
            process = subprocess.Popen(
                ffmpeg_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            
            # Collect all extracted frames
            for file in sorted(os.listdir(output_dir)):
                if file.startswith("frame_") and file.endswith(".jpg"):
                    frame_path = os.path.join(output_dir, file)
                    saved_frames.append(frame_path)
            
            if saved_frames:
                print(f"Successfully extracted {len(saved_frames)} frames from WebM using FFmpeg")
                return saved_frames
            
            print("FFmpeg extraction failed, falling back to OpenCV")
        except Exception as e:
            print(f"FFmpeg extraction error: {str(e)}, falling back to OpenCV")
    
    # Standard OpenCV method for non-WebM files or as fallback
    try:
        # Configure OpenCV for video files
        if video_path.lower().endswith('.webm'):
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'protocol_whitelist;file,rtp,udp,tcp'
        
        cap = cv2.VideoCapture(video_path)
        
        # For webm files, try setting more robust decoder options
        if video_path.lower().endswith('.webm'):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'VP80'))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        # frame_interval = max(1, int(fps))  # Ensure minimum 1 frame interval
        frame_interval = 1  # get every frame for tracking task
        frame_count = 0
        
        try:
            cv2.setLogLevel(0)
        except AttributeError:
            pass
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                try:
                    # Additional check for valid frame data
                    if frame is not None and frame.size > 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(output_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                        saved_frames.append(output_path)
                except Exception as e:
                    print(f"Warning: Failed to process frame {frame_count}: {str(e)}")
                    
            frame_count += 1
            
            # Safety limit to prevent infinite loops
            if frame_count > 1000:
                break
                
        cap.release()
        print(f"Extracted {len(saved_frames)} frames from video using OpenCV")
        
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        
    # If we couldn't extract any frames, create a placeholder
    if not saved_frames:
        try:
            print("Creating placeholder frame for failed video")
            # Create a blank image with text
            placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
            cv2.putText(
                placeholder, 
                "Video processing failed", 
                (80, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            placeholder_path = os.path.join(output_dir, "placeholder.jpg")
            cv2.imwrite(placeholder_path, placeholder)
            saved_frames.append(placeholder_path)
        except Exception as e:
            print(f"Failed to create placeholder: {str(e)}")
            
    return saved_frames 
