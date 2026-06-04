# ==============================================================================
# PART 1: IMPORTS & ENVIRONMENT SETUP
# ==============================================================================
import modal
import subprocess
import time
import os
import json
import shutil
import threading
import aiohttp
import urllib.request
import asyncio
import ctypes
import base64
import math
import re
from fastapi import Request, Response, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import Optional

# ==============================================================================
# PART 2: BASE IMAGE & OS CONFIGURATION
# ==============================================================================
base_image = modal.Image.from_registry(
    "nvidia/cuda:12.5.1-devel-ubuntu24.04",
    add_python="3.12"
).apt_install(
    "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0",
    "build-essential", "ninja-build", "cmake", "clang", "llvm",
    "libgoogle-perftools-dev" 
).env({
    "FORCE_REBUILD_INDEX": "383"  # Bumping for fresh deployment
})

# ==============================================================================
# PART 3: CORE PYTHON DEPENDENCIES & ENVIRONMENT VARIABLES
# ==============================================================================
build_image = base_image.env({
    "CUDA_HOME": "/usr/local/cuda",
    "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
    "FORCE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": "8.9",
    "MAX_JOBS": "1",
    "CC": "gcc",
    "CXX": "g++"
}).run_commands(
    "python3.12 -m pip install --no-cache-dir fastapi aiohttp boto3 triton>=3.1.0 ninja setuptools>=70.0.0 wheel pip>=24.0 Pillow",
    "python3.12 -m pip install --no-cache-dir pandas numexpr pytz python-dateutil scipy matplotlib colorama torchvision librosa soundfile decord imageio scikit-image numba einops bitsandbytes"
)

# ==============================================================================
# PART 4: COMFYUI & CUSTOM NODES CLONING
# ==============================================================================
torch_image = build_image.run_commands(
    "python3.12 -m pip install --no-cache-dir torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124",
    "python3.12 -m pip install --no-cache-dir diffusers accelerate transformers>=4.49.0 torchsde numpy==1.26.4 kornia==0.7.3",
    "python3.12 -m pip install --no-cache-dir sageattention==1.0.6"
)

clone_image = torch_image.run_commands(
    "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
    "GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    "git clone --depth 1 https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git /workspace/ComfyUI/custom_nodes/WhatDreamsCost-ComfyUI",
    "git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone --depth 1 https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes",
    "git clone --depth 1 https://github.com/liconstudio/ComfyUI-Licon-MSR /workspace/ComfyUI/custom_nodes/ComfyUI-Licon-MSR",
    "git clone --depth 1 https://github.com/regiellis/ComfyUI-EasyColorCorrector /workspace/ComfyUI/custom_nodes/ComfyUI-EasyColorCorrector",
    # Specific Dependencies for the New Workflow
    "git clone --depth 1 https://github.com/kijai/ComfyUI-MelBandRoformer.git /workspace/ComfyUI/custom_nodes/ComfyUI-MelBandRoformer",
    "git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
    "git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git /workspace/ComfyUI/custom_nodes/rgthree-comfy",
    "git clone --depth 1 https://github.com/cubiq/ComfyUI_essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_essentials",
    "git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /workspace/ComfyUI/custom_nodes/ComfyUI-Impact-Pack",
    "git clone --depth 1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts",
    "git clone --depth 1 https://github.com/twardowski/comfyui-resolution-master.git /workspace/ComfyUI/custom_nodes/comfyui-resolution-master",
    "git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux",
    "git clone --depth 1 https://github.com/chflame163/ComfyUI_LayerStyle.git /workspace/ComfyUI/custom_nodes/ComfyUI_LayerStyle",
    "git clone --depth 1 https://github.com/Suzie1/ComfyUI_ListHelper.git /workspace/ComfyUI/custom_nodes/ComfyUI_ListHelper"
)

deps_image = clone_image.run_commands(
    "sed -i '/torch/d' /workspace/ComfyUI/requirements.txt",
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec sed -i '/torch/d' {} \;",
    "python3.12 -m pip install --no-cache-dir -r /workspace/ComfyUI/requirements.txt",
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec python3.12 -m pip install --no-cache-dir -r {} \;"
)

final_image = deps_image.run_commands(
    "echo '' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py",
    "echo 'sageattn_qk_int8_pv_fp16_triton = sageattn' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py",
    "echo 'float8_e8m0fnu = int8' >> /usr/local/lib/python3.12/site-packages/torch/__init__.py",
    env={
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.9"
    }
)

# ==============================================================================
# PART 5: MODAL APP CONFIGURATION & CLOUD VOLUMES (L40S GPU TARGET)
# ==============================================================================
app = modal.App("media-worker-ltx23")
weights_volume = modal.Volume.from_name("Ltx-23-model-weights-new", create_if_missing=False)

@app.cls(
    gpu="L40S", 
    image=final_image,
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("custom-secret")],
    memory=8192, 
    scaledown_window=12,
    timeout=3600
)
class LTX23Engine:

    def _log_reader(self):
        for line in iter(self.process.stdout.readline, ""):
            if line: 
                print(f"[ComfyUI] {line.strip()}")

    async def _ram_squeezer(self):
        while True:
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1\n')
            except Exception: 
                pass
            await asyncio.sleep(15)

    @modal.enter()
    def start_comfy(self):
        import boto3
        
        # 🛡️ THE MASTER FIX: Monkey-patch LiconMSR to remove the 41-frame limit!
        msr_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-Licon-MSR/licon_msr.py"
        if os.path.exists(msr_path):
            try:
                with open(msr_path, "r") as f:
                    content = f.read()
                content = re.sub(r'\[\s*17\s*,\s*25\s*,\s*33\s*,\s*41\s*\]', '"INT"', content)
                with open(msr_path, "w") as f:
                    f.write(content)
                print("⚡ HACK SUCCESS: Unlocked LiconMSR frame_count limit to allow INFINITE video length!")
            except Exception as e:
                print(f"⚠️ Failed to patch LiconMSR: {e}")

        print("🎨 Injecting Smart Nodes, Caches & VAE Memory Protections...")
        custom_nodes_path = "/workspace/ComfyUI/custom_nodes/LTXCustomPipeline.py"
        with open(custom_nodes_path, "w") as f:
            f.write("""
import torch
import torchvision.transforms.functional as TF
import nodes

LTX_CACHE = {}

class MemoryCacheWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "scene_id": ("STRING", {"default": "0"})
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
                "guide_data": ("GUIDE_DATA",),
                "frame_rate": ("FLOAT", {"default": 24.0, "forceInput": True}),
                "audio": ("AUDIO",),  
            }
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "write_cache"
    CATEGORY = "LTXBatch"

    def write_cache(self, model, positive, scene_id, negative=None, video_latent=None, audio_latent=None, guide_data=None, frame_rate=24.0, audio=None):
        global LTX_CACHE
        LTX_CACHE[str(scene_id)] = {
            "model": model, 
            "positive": positive,
            "negative": negative,
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "guide_data": guide_data,
            "frame_rate": frame_rate,
            "audio": audio
        }
        print(f"\\n[Two-Pass System] 💾 Encoded & Saved Conditionings & AUDIO for Scene {scene_id} into RAM\\n")
        return ()

class MemoryCacheReader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "scene_id": ("STRING", {"default": "0"})
        }}
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "LATENT", "GUIDE_DATA", "FLOAT", "AUDIO")
    RETURN_NAMES = ("model", "positive", "negative", "video_latent", "audio_latent", "guide_data", "frame_rate", "audio")
    FUNCTION = "read_cache"
    CATEGORY = "LTXBatch"

    def read_cache(self, scene_id):
        global LTX_CACHE
        data = LTX_CACHE.get(str(scene_id))
        if data is None:
            raise ValueError(f"Cache for Scene {scene_id} not found in RAM! Text Encoder Pass failed.")
        print(f"\\n[Two-Pass System] 🚀 Loaded Pre-Cached Conditionings & AUDIO for Scene {scene_id}\\n")
        return (data.get("model"), data.get("positive"), data.get("negative"), data.get("video_latent"), data.get("audio_latent"), data.get("guide_data"), data.get("frame_rate", 24.0), data.get("audio"))

class FastVAEDecode(nodes.VAEDecode):
    def decode(self, vae, samples):
        print("\\n[Two-Pass System] 🛡️ Auto-Routing to Tiled VAE Decoding to protect 22B UNet VRAM state.\\n")
        try:
            return (vae.decode_tiled(samples["samples"], tile_x=512, tile_y=512), )
        except Exception as e:
            return super().decode(vae, samples)

NODE_CLASS_MAPPINGS = {
    "MemoryCacheWriter": MemoryCacheWriter,
    "MemoryCacheReader": MemoryCacheReader,
    "VAEDecode": FastVAEDecode
}
""")

        print("🔗 Running Atomic Model Folder Linker for LTX 2.3...")
        base_models_dir = "/workspace/ComfyUI/models"
        dirs = ["unet", "vae", "clip", "text_encoders", "checkpoints", "loras", "upscale_models", "latent_upscale_models"]
        for d in dirs: os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin")): continue
                    src_path = os.path.join(root_dir, filename)
                    for target_dir in dirs:
                        dest = os.path.join(base_models_dir, target_dir, filename)
                        if not os.path.exists(dest):
                            try: os.symlink(src_path, dest)
                            except FileExistsError: pass

        self.s3 = boto3.client(
            service_name='s3', 
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", 
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], 
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], 
            region_name="auto"
        )

        print("🚀 Launching Two-Pass Server Engine on L40S GPU (48GB Native)...")
        os.makedirs("/tmp/comfy_swap", exist_ok=True)

        env_vars = os.environ.copy()
        env_vars["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
        env_vars["TORCH_NUM_THREADS"] = "1"
        env_vars["OMP_NUM_THREADS"] = "1"
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:64"
        env_vars["CUDA_MODULE_LOADING"] = "LAZY" 
        
        self.process = subprocess.Popen([
            "python3.12", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--mmap-torch-files", "--cache-none", "--temp-directory", "/tmp/comfy_swap", 
            "--bf16-vae", "--use-sage-attention", "--fp8_e4m3fn-unet", "--fp8_e4m3fn-text-enc"
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        start_time = time.time()
        comfy_ready = False
        while time.time() - start_time < 300:
            if self.process.poll() is not None: os._exit(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1) as response:
                    if response.status == 200: 
                        comfy_ready = True
                        break
            except Exception: 
                time.sleep(2)
                
        if not comfy_ready: os._exit(1)
        print("Base pipeline active. Awaiting Two-Pass API Batch triggers.")

    async def clear_comfy_memory(self, session, unload_models=False):
        try:
            async with session.post("http://127.0.0.1:8188/free", json={"unload_models": unload_models, "free_memory": True}) as r:
                await r.read()
        except Exception: pass
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        try: ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception: pass
        await asyncio.sleep(1)

    async def execute_comfy_workflow(self, session, workflow_json):
        async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow_json}) as r:
            if r.status != 200:
                err_text = await r.text()
                raise HTTPException(status_code=500, detail=f"Failed to queue prompt: {r.status} - {err_text}")
            res = await r.json()
            prompt_id = res["prompt_id"]

        while True:
            async with session.get(f"http://127.0.0.1:8188/history/{prompt_id}") as r:
                if r.status == 200:
                    history_data = await r.json()
                    if prompt_id in history_data:
                        step_data = history_data[prompt_id]
                        if "status" in step_data and "messages" in step_data["status"]:
                            for msg in step_data["status"]["messages"]:
                                if msg[0] == "execution_error":
                                    raise HTTPException(status_code=500, detail=f"ComfyUI execution error: {msg[1]}")
                        return step_data
            
            if self.process.poll() is not None:
                raise HTTPException(status_code=500, detail="ComfyUI server process crashed.")
            await asyncio.sleep(1)

    # ==============================================================================
    # PART 6: TWO-PASS HIGH-SPEED BATCH ENDPOINT 
    # ==============================================================================
    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != "testing-modal-workflow-2": 
            raise HTTPException(status_code=403, detail="Unauthorized Pipeline Request")
        
        body = await request.json()
        if isinstance(body, dict):
            if "json" in body: body = body["json"]
            elif "body" in body: body = body["body"]

        async def process_pipeline():
            from urllib.parse import urlparse
            import botocore.exceptions

            date_folder = body.get("date_folder", time.strftime('%Y-%m-%d'))
            batch_scenes = body.get("batch_scenes", [])
            subgraph_1 = body.get("subgraph_1")
            subgraph_2 = body.get("subgraph_2")

            if not batch_scenes: raise HTTPException(status_code=400, detail="Missing batch_scenes array.")
            if not subgraph_1 or not subgraph_2: raise HTTPException(status_code=400, detail="Missing Subgraph definitions.")

            dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
            if os.path.exists(dynamic_guides_dir): shutil.rmtree(dynamic_guides_dir)
            os.makedirs(dynamic_guides_dir, exist_ok=True)

            # Known ComfyUI Camera LoRA Mappings for Text2Video Action Generation
            camera_loras_map = {
                "dolly_in": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
                "dolly_out": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
                "dolly_left": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
                "dolly_right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
                "jib_up": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
                "jib_down": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
                "static": "ltx-2-19b-lora-camera-control-static.safetensors"
            }

            ram_task = asyncio.create_task(self._ram_squeezer())
            generated_outputs = []

            try:
                async with aiohttp.ClientSession() as session:
                    # Dimensions locked to multiples of 32 for UNet safety
                    custom_w = int(body.get("custom_width", 576))
                    custom_h = int(body.get("custom_height", 1024))
                    custom_w = (custom_w // 32) * 32
                    custom_h = (custom_h // 32) * 32

                    # ==============================================================================
                    # PRE-COMPUTE: Download ALL Reference Images & Audio
                    # ==============================================================================
                    for idx, scene in enumerate(batch_scenes):
                        relative_scene_dir = f"dynamic_guides/scene_{idx}"
                        scene_img_dir = os.path.join("/workspace/ComfyUI/input", relative_scene_dir)
                        os.makedirs(scene_img_dir, exist_ok=True)
                        
                        bg_path = os.path.join(scene_img_dir, "black_bg.png")
                        from PIL import Image
                        Image.new('RGB', (custom_w, custom_h), color='black').save(bg_path)
                        scene["_bg_img_path"] = f"{relative_scene_dir}/black_bg.png"
                        
                        image_urls = scene.get("image_urls", [])
                        if not image_urls and scene.get("image_url"):
                            image_urls = [scene.get("image_url")]

                        valid_relative_paths = []

                        if not image_urls:
                            target_path = os.path.join(scene_img_dir, "default.png")
                            Image.new('RGB', (custom_w, custom_h), color='black').save(target_path)
                            valid_relative_paths.append(f"{relative_scene_dir}/default.png")
                        else:
                            for img_i, url_str in enumerate(image_urls):
                                target_path = os.path.join(scene_img_dir, f"img_{img_i}.png")
                                try:
                                    parsed = urlparse(url_str)
                                    if "r2.cloudflarestorage.com" in url_str or "pub-" in url_str or parsed.netloc == "" or not parsed.scheme:
                                        file_key = parsed.path.lstrip('/')
                                        await asyncio.get_event_loop().run_in_executor(None, self.s3.download_file, "video-asset-files-storage-workflow", file_key, target_path)
                                    else:
                                        async with session.get(url_str, timeout=120) as r:
                                            if r.status == 200:
                                                with open(target_path, "wb") as f: f.write(await r.read())
                                except Exception as e:
                                    print(f"Image Download Failed: {e}")
                                
                                if os.path.exists(target_path):
                                    valid_relative_paths.append(f"{relative_scene_dir}/img_{img_i}.png")

                        # 🎵 Audio Handling
                        audio_url = scene.get("audio_url", "")
                        if audio_url:
                            audio_target_path = os.path.join(scene_img_dir, "scene_audio.mp3")
                            try:
                                parsed = urlparse(audio_url)
                                if "r2.cloudflarestorage.com" in audio_url or "pub-" in audio_url or parsed.netloc == "" or not parsed.scheme:
                                    file_key = parsed.path.lstrip('/')
                                    await asyncio.get_event_loop().run_in_executor(None, self.s3.download_file, "video-asset-files-storage-workflow", file_key, audio_target_path)
                                else:
                                    async with session.get(audio_url, timeout=120) as r:
                                        if r.status == 200:
                                            with open(audio_target_path, "wb") as f: f.write(await r.read())
                                scene["_audio_path"] = f"{relative_scene_dir}/scene_audio.mp3"
                            except Exception as e:
                                scene["_audio_path"] = ""
                        else:
                            scene["_audio_path"] = ""

                        # 🧠 Text Prompting & Timeline Formulation
                        actions = scene.get("kinetic_actions", ["The subject moves dynamically across the cinematic scene."])
                        audio_prompt = scene.get("audio_prompt", "") 
                        
                        total_words = sum(len(str(a).split()) for a in actions)
                        seconds = max(total_words / (130 / 60.0), 2.5)  
                        total_frames = max(33, min(int(math.ceil(((seconds * 24) - 1) / 8) * 8 + 1), 257))
                        
                        keyframe_steps = [int(i * (total_frames - 1) / max(1, len(actions) - 1)) for i in range(len(actions))]

                        static_env = f"{scene.get('subject', '')} {scene.get('style', '')} {scene.get('background', '')}".strip()
                        
                        local_prompts_list = []
                        for step_frame, action_text in zip(keyframe_steps, actions):
                            fused_prompt = f"{action_text} {scene.get('camera', '')}".strip()
                            if audio_prompt:
                                fused_prompt += f". Speech and Audio details: {audio_prompt}."
                            fused_prompt += f" Cinematic environment: {static_env}"
                            local_prompts_list.append(f"{step_frame}: {fused_prompt}")
                            
                        scene["_local_prompts_str"] = "\n".join(local_prompts_list)
                        scene["_total_frames"] = total_frames
                        scene["_image_paths_str"] = "\n".join(valid_relative_paths)
                        scene["_seed"] = scene.get("seed", int(time.time() * 1000) % 1000000)

                    # ==============================================================================
                    # PASS 1: DYNAMIC GRAPH TOPOLOGY BATCHING (Injecting LoRAs & Context)
                    # ==============================================================================
                    print("\n[Two-Pass System] 🎬 PASS 1 START: Enconding Texts, Negatives, Audio & Dynamic LoRAs...")
                    
                    global_nodes = {}
                    scene_template = {}
                    for n_id, n_data in subgraph_1.items():
                        c_type = n_data.get("class_type")
                        if c_type in ["UNETLoader", "VAELoaderKJ", "DualCLIPLoader", "MelBandRoFormerModelLoader"]:
                            global_nodes[n_id] = n_data
                        else:
                            scene_template[n_id] = n_data
                            
                    pass1_workflow = {}
                    
                    for n_id, n_data in global_nodes.items():
                        pass1_workflow[n_id] = json.loads(json.dumps(n_data))

                    for idx, scene in enumerate(batch_scenes):
                        
                        # -------------------------------------------------------------
                        # BUILD ACTIVE LORA LIST FOR THIS SCENE
                        # -------------------------------------------------------------
                        active_loras = [
                            {"name": "LTX_2.3_Crisp_Enhance_Style_LoRa.safetensors", "strength": 0.5},
                            {"name": "VBVR-official-comfyui.safetensors", "strength": 0.7},
                            {"name": "LTX_2.3_Soft_Enhance_Style_LoRa.safetensors", "strength": 0.5},
                            {"name": "LTX-2.3_Cinematic_hardcut.safetensors", "strength": 0.75}
                        ]
                        
                        if "loras" in scene and isinstance(scene["loras"], list):
                            for lora_obj in scene["loras"]:
                                active_loras.append({"name": lora_obj["name"], "strength": lora_obj.get("strength", 1.0)})
                                
                        cam_string = scene.get("camera", "")
                        if cam_string:
                            cams = [c.strip().lower() for c in cam_string.split("+")]
                            for c in cams[:3]:
                                if c in camera_loras_map:
                                    active_loras.append({"name": camera_loras_map[c], "strength": 1.0})
                        if not any(c in camera_loras_map.values() for c in [l["name"] for l in active_loras]):
                            active_loras.append({"name": camera_loras_map["static"], "strength": 1.0})

                        for n_id, n_data in scene_template.items():
                            new_id = f"{n_id}_{idx}"
                            new_node = json.loads(json.dumps(n_data))
                            c_type = new_node.get("class_type")
                            
                            # Preserve internal and external topological links
                            for in_key, in_val in new_node.get("inputs", {}).items():
                                if isinstance(in_val, list) and len(in_val) == 2 and isinstance(in_val[0], str):
                                    target_id = in_val[0]
                                    if target_id in scene_template:
                                        new_node["inputs"][in_key] = [f"{target_id}_{idx}", in_val[1]]
                                    elif target_id in global_nodes:
                                        new_node["inputs"][in_key] = [target_id, in_val[1]]

                            # -------------------------------------------------------------
                            # DYNAMIC LORA LOADER INJECTION
                            # Intercept the rgthree Power Lora Loader and replace it entirely 
                            # with a native sequential chain of LoraLoaderModelOnly nodes.
                            # -------------------------------------------------------------
                            if c_type == "Power Lora Loader (rgthree)":
                                base_model_link = new_node["inputs"]["model"]
                                current_model_link = base_model_link
                                
                                for l_idx, lora in enumerate(active_loras):
                                    # The very final LoRA node in the chain adopts the ID of the replaced 
                                    # rgthree node so downstream connections naturally succeed.
                                    lora_node_id = new_id if l_idx == len(active_loras) - 1 else f"dynamic_lora_{idx}_{l_idx}"
                                    
                                    pass1_workflow[lora_node_id] = {
                                        "class_type": "LoraLoaderModelOnly",
                                        "inputs": {
                                            "lora_name": lora["name"],
                                            "strength_model": lora["strength"],
                                            "model": current_model_link
                                        }
                                    }
                                    current_model_link = [lora_node_id, 0]
                                
                                # We deliberately DO NOT append the rgthree node to the workflow payload
                                continue
                            
                            # Safely inject parameters by Class Type
                            if c_type == "LTXDirector":
                                new_node["inputs"]["duration_frames"] = scene["_total_frames"]
                                new_node["inputs"]["local_prompts"] = scene["_local_prompts_str"]
                                new_node["inputs"]["custom_width"] = custom_w
                                new_node["inputs"]["custom_height"] = custom_h
                                
                            elif c_type == "MemoryCacheWriter":
                                new_node["inputs"]["scene_id"] = str(idx)
                                
                            elif c_type in ["VHS_LoadAudioUpload", "LoadAudio"]:
                                if scene.get("_audio_path"):
                                    new_node["inputs"]["audio"] = scene["_audio_path"]

                            elif c_type == "CLIPTextEncode":
                                current_text = str(new_node.get("inputs", {}).get("text", "")).lower()
                                if "blurry" in current_text or "distorted" in current_text or "no humans" in current_text:
                                    custom_neg = scene.get("negative_prompt", "no humans, bad quality, distorted, blurry, watermark, mutated, glitches")
                                    new_node["inputs"]["text"] = custom_neg

                            pass1_workflow[new_id] = new_node

                    print(f"🚀 Queuing Math Encoding Pass for {len(batch_scenes)} Scenes simultaneously...")
                    await self.execute_comfy_workflow(session, pass1_workflow)
                    await self.clear_comfy_memory(session, unload_models=False)

                    # ==============================================================================
                    # PASS 2: THE SAMPLING BLAST (Iterative per scene)
                    # ==============================================================================
                    print("\n[Two-Pass System] 🚀 PASS 2 START: Initiating Pure Sampling Blast...")
                    
                    for idx, scene in enumerate(batch_scenes):
                        print(f"\n🎬 Rendering Native Scene [{idx+1}/{len(batch_scenes)}]: {scene.get('name', 'Clip')}")
                        pass2_workflow = json.loads(json.dumps(subgraph_2))
                        out_dir = "/workspace/ComfyUI/output"
                        if os.path.exists(out_dir): shutil.rmtree(out_dir)
                        os.makedirs(out_dir)

                        scene_seed = scene["_seed"]

                        for n_id, n_data in pass2_workflow.items():
                            c_type = n_data.get("class_type")
                            
                            # Cache injection 
                            if c_type == "MemoryCacheReader":
                                n_data["inputs"]["scene_id"] = str(idx)
                                n_data["inputs"].pop("cache_id", None)
                                
                            # Safe seed injection
                            if "noise_seed" in n_data.get("inputs", {}):
                                n_data["inputs"]["noise_seed"] = scene_seed
                            elif "seed" in n_data.get("inputs", {}):
                                n_data["inputs"]["seed"] = scene_seed
                                
                            # Configuration formatting
                            if c_type == "VHS_VideoCombine":
                                n_data["inputs"]["frame_rate"] = 24
                                if "pingpong" in n_data.get("inputs", {}): 
                                    n_data["inputs"]["pingpong"] = False

                        await self.execute_comfy_workflow(session, pass2_workflow)

                        output_files = []
                        for root_p, _, filenames in os.walk(out_dir):
                            for name in filenames:
                                if name.endswith((".mp4", ".gif", ".webm")):
                                    output_files.append(os.path.join(root_p, name))

                        if not output_files: raise Exception("Inference finished but no output media files were detected.")
                        output_files.sort(key=os.path.getmtime)
                        target_video_file = output_files[-1]
                        saved_filename = os.path.basename(target_video_file)

                        target_key = f"{date_folder}/generated clips/{int(time.time())}_{scene.get('name', 'clip')}_{saved_filename}"
                        print(f"📤 Syncing Finished Asset to R2: {target_key}")
                        
                        await asyncio.get_event_loop().run_in_executor(
                             None, self.s3.upload_file, target_video_file, "video-asset-files-storage-workflow", target_key
                        )

                        public_path_url = f"https://pub-4d91f4d3d0366568a54ffa32ffcb7bf4.r2.dev/{target_key}" 
                        generated_outputs.append({
                            "scene": scene.get("name", f"Clip_{idx+1}"),
                            "status": "success",
                            "file_key": target_key,
                            "public_url": public_path_url,
                            "filename": saved_filename
                        })
                        await self.clear_comfy_memory(session, unload_models=False)
                    
                    return generated_outputs

            finally:
                ram_task.cancel()

        async def stream_response():
            task = asyncio.create_task(process_pipeline())
            while not task.done():
                yield b" "  
                done, pending = await asyncio.wait([task], timeout=10.0)
                if task in done: break
            try:
                result = task.result()
                if isinstance(result, (dict, list)):
                    yield json.dumps(result).encode("utf-8")
                else:
                    yield str(result).encode("utf-8")
            except HTTPException as e:
                yield json.dumps({"status": "error", "detail": e.detail}).encode("utf-8")
            except Exception as e:
                yield json.dumps({"status": "error", "detail": str(e)}).encode("utf-8")

        return StreamingResponse(stream_response(), media_type="application/json")
