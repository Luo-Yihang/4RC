<div align="center">
<div style="text-align: center;">
    <img src="./assets/logo.png" alt="4RC Logo" style="height: 100px;">
    <h2>4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere</h2>
</div>

<div>
    <a href='https://scholar.google.com/citations?user=fZxK2B0AAAAJ&hl=en' target='_blank'>Yihang Luo</a><sup>1</sup>&emsp;
    <a href='https://shangchenzhou.com/' target='_blank'>Shangchen Zhou</a><sup>1</sup>&emsp;
    <a href="https://nirvanalan.github.io/" target='_blank'>Yushi Lan</a><sup>2</sup>&emsp;
    <a href="https://xingangpan.github.io/" target='_blank'>Xingang Pan</a><sup>1</sup>&emsp;
    <a href="https://www.mmlab-ntu.com/person/ccloy/" target='_blank'>Chen Change Loy</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp; 
    <sup>2</sup>University of Oxford&emsp; 
</div>


<div>
    <h4 align="center">
        <a href="https://luo-yihang.github.io/projects/4RC/" target='_blank'>
        <img src="https://img.shields.io/badge/🌐-Project%20Page-blue">
        </a>
        <a href="http://arxiv.org/abs/2602.10094" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2602.10094-b31b1b.svg">
        </a>
        <img src="https://api.infinitescript.com/badgen/count?name=sczhou/4RC&ltext=Visitors&color=3977dd">
    </h4>
</div>

<strong>4RC <em>(pronounced "ARC")</em> enables unified and complete 4D reconstruction via conditional querying from monocular videos in a single feed-forward pass.</strong>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="assets/teaser.png">
</div>

:sparkler: For more visual results, go checkout our <a href="https://yihangluo.com/projects/4RC/" target="_blank">project page</a>

---
</div>

<details>
<summary><b>Introducing 4RC</b></summary>
    <br>
    <div align="center">
        <img width="820" alt="framework" src="assets/framework.png">
        <p align="justify">
            We present 4RC, a unified feed-forward framework for 4D reconstruction from monocular videos. 
            Unlike existing methods that typically decouple motion from geometry or produce limited 4D attributes, 
            such as sparse trajectories or two-view scene flow, 4RC learns a holistic 4D representation that 
            jointly captures dense scene geometry and motion dynamics. At its core, 4RC introduces a novel 
            encode-once, query-anywhere and anytime paradigm: a transformer backbone encodes the entire video 
            into a compact spatio-temporal latent space, from which a conditional decoder can efficiently query 
            3D geometry and motion for any query frame at any target timestamp. To facilitate learning, we 
            represent per-view 4D attributes in a minimally factorized form, decomposing them into base 
            geometry and time-dependent relative motion. Extensive experiments demonstrate that 4RC outperforms 
            prior and concurrent methods across a wide range of 4D reconstruction tasks.
        </p>
    </div>
</details>

## 🔥 News
- [2026/04/13] Our inference code and weights are released!

## 🔧 Installation

1. Clone Repo
    ```bash
    git clone https://github.com/Luo-Yihang/4RC
    cd 4RC
    ```

2. Create Conda Environment
    ```bash
    conda create -n 4rc python=3.11 cmake=3.14.0 -y
    conda activate 4rc
    ```

3. Install Python Dependencies

    **Important:** Install [Torch](https://pytorch.org/get-started/locally/) based on your CUDA version. For example, for *Torch 2.8.0 + CUDA 12.6*:

    ```bash
    # Install Torch
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

    # Install other dependencies
    pip install -r requirements.txt

    # Install 4RC as a package
    pip install -e .
    ```

## :computer: Inference

You can now try 4RC with the following code. The checkpoint will be downloaded automatically from [Hugging Face](https://huggingface.co/Luo-Yihang/4RC). 

```python
import torch

from arc.models.arc.arc import Arc
from arc.dust3r.inference_multiview import inference
from arc.dust3r.utils.image import load_images

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Arc.from_pretrained("Luo-Yihang/4RC").to(device)
model.eval()

example_dir = "examples/robot_arm"
images = load_images(example_dir, size=512, patch_size=14, verbose=True)

with torch.no_grad():
    predictions, profiling = inference(
        images,
        model,
        device,
        dtype="bf16-mixed",
        profiling=True,
        verbose=True,
        use_center_as_anchor=False,
    )
``` 

## :zap: Demo

Launch the interactive Gradio demo:

```bash
python app.py
```

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="assets/gradio_demo.png">
</div>

## :mag: CLI

For the command-line workflow without the Gradio UI, use the two-step pipeline:

**Step 1: Run inference and save to `.npz`:**

```bash
python inference.py --input ./examples/robot_arm --save result.npz
```

***[Optional]*** 
- *Use `--refine_track_visualization` to enable VLA + SAM2 to auto-segment dynamic objects and filter their trajectories for better visulization.*
- *Use `--checkpoint_dir Luo-Yihang/4RC_geofinetune` to use the checkpoint finetuned on more geometry datasets for even better geometry prediction.*

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="assets/viser_demo.gif">
</div>

**Step 2: Visualize with viser directly from `.npz`:**

```bash
python arc/viz/viser_visualizer_track.py --npz_path result.npz --port 8020
```

Open `http://localhost:8020` in your browser to interact with the 3D visualization.

## 📁 Code Structure

```text
4RC/
├── arc/
│   ├── models/
│   │   └── arc/
│   ├── dust3r/
│   ├── croco/
│   └── viz/
├── assets/
├── examples/
├── app.py
├── inference.py
├── requirements.txt
├── setup.py
└── README.md
```

## :calendar: TODO

🐎 Pushing the bandwidth limit!

- [ ] Release evaluation code.
- [ ] Release training code.


## 📝 Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
  @article{luo20264rc,
      title     = {4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere},
      author    = {Yihang Luo and Shangchen Zhou and Yushi Lan and Xingang Pan and Chen Change Loy},
      journal   = {arXiv preprint arXiv:2602.10094},
      year      = {2026}
  }
   ```

## :pencil: Acknowledgments
We recognize several concurrent works on the 4D reconstruction. We encourage you to check them out:
  
[St4RTrack](https://github.com/HavenFeng/St4RTrack) &nbsp;|&nbsp; [TraceAnything](https://github.com/ByteDance-Seed/TraceAnything) &nbsp;|&nbsp; [V-DPM](https://github.com/eldar/vdpm) &nbsp;|&nbsp; [Any4D](https://github.com/Any-4D/Any4D) &nbsp;|&nbsp; [D4RT](https://d4rt-paper.github.io/)

4RC is built on the shoulders of several outstanding open-source projects. Many thanks to the following exceptional projects:

[DA3](https://github.com/ByteDance-Seed/depth-anything-3) &nbsp;|&nbsp; [VGGT](https://github.com/facebookresearch/vggt) &nbsp;|&nbsp; [Fast3R](https://github.com/facebookresearch/fast3r) &nbsp;|&nbsp; [DUSt3R](https://github.com/naver/dust3r) &nbsp;|&nbsp; [Viser](https://github.com/nerfstudio-project/viser)


## 📫 Contact

If you have any questions, please feel free to reach us at `luo_yihang@outlook.com`.
