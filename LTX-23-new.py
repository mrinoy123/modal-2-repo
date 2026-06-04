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
    "FORCE_REBUILD_INDEX": "366"  # Bumping for safe runtime cache refresh
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
    "git clone --depth 1 https://github.com/regiellis/ComfyUI-EasyColorCorrector /workspace/ComfyUI/custom_nodes/ComfyUI-EasyColorCorrector"
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
        
        print("🎨 Injecting Smart Nodes, Caches & VAE Memory Protections...")
        custom_nodes_path = "/workspace/ComfyUI/custom_nodes/LTXCustomPipeline.py"
        with open(custom_nodes_path, "w") as f:
            f.write("""
import torch
import torchvision.transforms.functional as TF
import nodes

class LTXColorFixer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "target_brightness": ("FLOAT", {"default": 0.40, "min": 0.1, "max": 1.0, "step": 0.05}),
            "max_boost": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 3.0, "step": 0.1}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"

    def process(self, image, target_brightness, max_boost):
        r = image[:, :, :, 0]
        g = image[:, :, :, 1]
        b = image[:, :, :, 2]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        mean_lum = torch.mean(luminance).item()
        
        if mean_lum < target_brightness and mean_lum > 0.01:
            boost = target_brightness / mean_lum
            boost = min(boost, max_boost)
            img_t = image.permute(0, 3, 1, 2)
            img_t = TF.adjust_brightness(img_t, boost)
            sat_boost = 1.0 + ((boost - 1.0) * 0.4) 
            img_t = TF.adjust_saturation(img_t, sat_boost)
            image = img_t.permute(0, 2, 3, 1)
            
        return (image,)

LTX_CACHE = {}

class MemoryCacheWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "video_latent": ("LATENT",),
            "audio_latent": ("LATENT",),
            "guide_data": ("GUIDE_DATA",),
            "frame_rate": ("FLOAT", {"default": 24.0, "forceInput": True}),
            "scene_id": ("STRING", {"default": "0"})
        }}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "write_cache"
    CATEGORY = "LTXBatch"

    def write_cache(self, model, positive, negative, video_latent, audio_latent, guide_data, frame_rate, scene_id):
        global LTX_CACHE
        LTX_CACHE[str(scene_id)] = {
            "model": model, 
            "positive": positive,
            "negative": negative,
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "guide_data": guide_data,
            "frame_rate": frame_rate
        }
        print(f"\\n[Two-Pass System] 💾 Encoded & Saved Conditionings for Scene {scene_id} into RAM\\n")
        return ()

class MemoryCacheReader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "scene_id": ("STRING", {"default": "0"})
        }}
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "LATENT", "GUIDE_DATA", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "video_latent", "audio_latent", "guide_data", "frame_rate")
    FUNCTION = "read_cache"
    CATEGORY = "LTXBatch"

    def read_cache(self, scene_id):
        global LTX_CACHE
        data = LTX_CACHE.get(str(scene_id))
        if data is None:
            raise ValueError(f"Cache for Scene {scene_id} not found in RAM! Text Encoder Pass failed.")
        print(f"\\n[Two-Pass System] 🚀 Loaded Pre-Cached Conditionings for Scene {scene_id}\\n")
        return (data["model"], data["positive"], data["negative"], data["video_latent"], data["audio_latent"], data["guide_data"], data["frame_rate"])

class FastVAEDecode(nodes.VAEDecode):
    def decode(self, vae, samples):
        print("\\n[Two-Pass System] 🛡️ Auto-Routing to Tiled VAE Decoding to protect 22B UNet VRAM state.\\n")
        try:
            return (vae.decode_tiled(samples["samples"], tile_x=512, tile_y=512), )
        except Exception as e:
            return super().decode(vae, samples)

NODE_CLASS_MAPPINGS = {
    "LTXColorFixer": LTXColorFixer,
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
        print("✅ Base pipeline active. Awaiting Two-Pass API Batch triggers.")

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
            date_folder = body.get("date_folder", time.strftime('%Y-%m-%d'))
            batch_scenes = body.get("batch_scenes", [])
            subgraph_1 = body.get("subgraph_1")
            subgraph_2 = body.get("subgraph_2")

            if not batch_scenes: raise HTTPException(status_code=400, detail="Missing batch_scenes array.")
            if not subgraph_1 or not subgraph_2: raise HTTPException(status_code=400, detail="Missing Subgraph definitions.")

            # 🛡️ FIX 1: We use relative directory structures for ComfyUI's strict Input security rules
            comfy_input_dir = "/workspace/ComfyUI/input"
            dynamic_guides_rel = "dynamic_guides"
            dynamic_guides_abs = os.path.join(comfy_input_dir, dynamic_guides_rel)
            
            if os.path.exists(dynamic_guides_abs): shutil.rmtree(dynamic_guides_abs)
            os.makedirs(dynamic_guides_abs, exist_ok=True)

            ram_task = asyncio.create_task(self._ram_squeezer())
            generated_outputs = []

            try:
                async with aiohttp.ClientSession() as session:
                    custom_w = body.get("custom_width", 576)
                    custom_h = body.get("custom_height", 1024)

                    # ==============================================================================
                    # PRE-COMPUTE: Download ALL Reference Images & Calculate Dimensions
                    # ==============================================================================
                    for idx, scene in enumerate(batch_scenes):
                        relative_scene_dir = f"{dynamic_guides_rel}/scene_{idx}"
                        scene_img_dir = os.path.join(comfy_input_dir, relative_scene_dir)
                        os.makedirs(scene_img_dir, exist_ok=True)
                        
                        image_urls = scene.get("image_urls", [])
                        if not image_urls and scene.get("image_url"):
                            image_urls = [scene.get("image_url")]

                        if not image_urls:
                            target_path = os.path.join(scene_img_dir, "default.png")
                            from PIL import Image
                            Image.new('RGB', (custom_w, custom_h), color='black').save(target_path)
                            segments = [{"frame": 0, "image": ""}, {"frame": 1, "image": ""}]
                        else:
                            segments = []
                            for img_i, url in enumerate(image_urls):
                                target_path = os.path.join(scene_img_dir, f"img_{img_i}.png")
                                try:
                                    async with session.get(url, timeout=120) as r:
                                        if r.status == 200:
                                            with open(target_path, "wb") as f: f.write(await r.read())
                                            from PIL import Image
                                            img = Image.open(target_path).convert("RGB")
                                            img = img.resize((custom_w, custom_h), Image.Resampling.LANCZOS)
                                            img.save(target_path)
                                            
                                            if img_i == 0:
                                                with open(target_path, "rb") as f:
                                                    b64_img = base64.b64encode(f.read()).decode("utf-8")
                                                    segments.append({"frame": 0, "image": f"data:image/png;base64,{b64_img}"})
                                                    segments.append({"frame": 1, "image": f"data:image/png;base64,{b64_img}"})
                                except Exception as e:
                                    print(f"Failed to download image {url}: {e}")
                        
                        timeline_data_str = json.dumps({"segments": segments, "audioSegments": []})

                        actions = scene.get("kinetic_actions", [])
                        if not actions: actions = ["The subject moves dynamically across the cinematic scene."]
                        
                        total_words = sum(len(str(a).split()) for a in actions)
                        seconds = max(total_words / (130 / 60.0), 2.5)  
                        raw_frames = seconds * 24
                        total_frames = int(math.ceil((raw_frames - 1) / 8) * 8 + 1)
                        total_frames = max(33, min(total_frames, 257))
                        
                        num_actions = len(actions)
                        keyframe_steps = [int(i * (total_frames - 1) / max(1, num_actions - 1)) for i in range(num_actions)]

                        # 🛡️ FIX 2: Clamp LiconMSR Guide frames to its strict valid combo box selection
                        valid_msr_frames = [17, 25, 33, 41]
                        msr_frames = 41
                        for vf in reversed(valid_msr_frames):
                            if total_frames >= vf:
                                msr_frames = vf
                                break

                        style = scene.get("style", "")
                        subject = scene.get("subject", "")
                        bg = scene.get("background", "")
                        light = scene.get("lighting", "")
                        cam = scene.get("camera", "")
                        
                        static_env = f"{subject} {style} {bg} {light}".strip()
                        
                        local_prompts_list = []
                        for step_frame, action_text in zip(keyframe_steps, actions):
                            action_cam = f"{action_text} {cam}".strip()
                            fused_prompt = f"{action_cam}. Cinematic environment and styling: {static_env}"
                            local_prompts_list.append(f"{step_frame}: {fused_prompt}")
                            
                        local_prompts_str = "\n".join(local_prompts_list)
                        
                        scene["_timeline_data_str"] = timeline_data_str
                        scene["_local_prompts_str"] = local_prompts_str
                        scene["_total_frames"] = total_frames
                        scene["_msr_frames"] = msr_frames
                        scene["_img_dir"] = relative_scene_dir  # Safely mapped relative path
                        scene["_seed"] = scene.get("seed", int(time.time() * 1000) % 1000000)

                    # ==============================================================================
                    # PASS 1: TEXT ENCODING, IC-LORAS & DYNAMIC LORA MATRIX BATCHING
                    # ==============================================================================
                    print("\n[Two-Pass System] 🎬 PASS 1 START: Initiating Text Encoding, IC-LoRAs & Matrix Mapping...")
                    pass1_workflow = json.loads(json.dumps(subgraph_1))
                    
                    if "98" in pass1_workflow: 
                        pass1_workflow["98"]["inputs"]["unet_name"] = "ltx-2.3-22b-distilled-fp8.safetensors"
                        pass1_workflow["98"]["inputs"]["weight_dtype"] = "fp8_e4m3fn"
                    if "97" in pass1_workflow: pass1_workflow["97"]["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                    if "102" in pass1_workflow: pass1_workflow["102"]["inputs"]["vae_name"] = "LTX23_audio_vae_bf16.safetensors"
                    if "101" in pass1_workflow: 
                        pass1_workflow["101"]["inputs"]["clip_name1"] = "gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors"
                        pass1_workflow["101"]["inputs"]["clip_name2"] = "ltx-2.3_text_projection_bf16.safetensors"

                    tpl_107 = pass1_workflow.pop("107", None)
                    tpl_200 = pass1_workflow.pop("200", None)
                    tpl_46 = pass1_workflow.pop("46", None)
                    tpl_94_5 = pass1_workflow.pop("94:5", None)
                    tpl_300 = pass1_workflow.pop("300", None)
                    tpl_351 = pass1_workflow.pop("351", None)
                    tpl_352 = pass1_workflow.pop("352", None)
                    tpl_353 = pass1_workflow.pop("353", None)
                    tpl_354 = pass1_workflow.pop("354", None)
                    tpl_320 = pass1_workflow.pop("320", None)
                    tpl_330 = pass1_workflow.pop("330", None)

                    camera_loras_map = {
                        "dolly_in": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
                        "dolly_out": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
                        "dolly_left": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
                        "dolly_right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
                        "jib_up": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
                        "jib_down": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
                        "static": "ltx-2-19b-lora-camera-control-static.safetensors"
                    }

                    for idx, scene in enumerate(batch_scenes):
                        scene_107 = json.loads(json.dumps(tpl_107))
                        scene_107["inputs"]["lora_1"] = "LTX_2.3_Crisp_Enhance_Style_LoRa.safetensors"
                        scene_107["inputs"]["strength_1"] = 0.5
                        scene_107["inputs"]["lora_2"] = "VBVR-official-comfyui.safetensors"
                        scene_107["inputs"]["strength_2"] = 0.7
                        scene_107["inputs"]["lora_3"] = "LTX_2.3_Soft_Enhance_Style_LoRa.safetensors"
                        scene_107["inputs"]["strength_3"] = 0.5
                        scene_107["inputs"]["lora_4"] = "LTX-2.3_Cinematic_hardcut.safetensors"
                        scene_107["inputs"]["strength_4"] = 0.75 
                        scene_107["inputs"]["enabled_4"] = True

                        active_cameras = []
                        cam_string = scene.get("camera", "")
                        if cam_string:
                            cams = [c.strip().lower() for c in cam_string.split("+")]
                            for c in cams[:3]:
                                if c in camera_loras_map:
                                    active_cameras.append(camera_loras_map[c])
                        if not active_cameras: active_cameras.append(camera_loras_map["static"])

                        for i in range(3):
                            slot = i + 5
                            if i < len(active_cameras):
                                scene_107["inputs"][f"lora_{slot}"] = active_cameras[i]
                                scene_107["inputs"][f"strength_{slot}"] = 1.0
                                scene_107["inputs"][f"enabled_{slot}"] = True
                            else:
                                scene_107["inputs"][f"lora_{slot}"] = "__none__"
                                scene_107["inputs"][f"strength_{slot}"] = 0.0
                                scene_107["inputs"][f"enabled_{slot}"] = False
                        pass1_workflow[f"107_{idx}"] = scene_107

                        scene_200 = json.loads(json.dumps(tpl_200))
                        scene_200["inputs"]["text"] = scene.get("negative_prompt", "no humans, bad quality, distorted, blurry, watermark")
                        scene_200["inputs"]["clip"] = [f"107_{idx}", 1]
                        pass1_workflow[f"200_{idx}"] = scene_200

                        scene_353 = json.loads(json.dumps(tpl_353))
                        scene_353["inputs"]["model"] = [f"107_{idx}", 0]
                        scene_353["inputs"]["lora_name"] = "ltx-2.3-22b-ic-lora-refocus.safetensors"
                        scene_353["inputs"]["strength_model"] = 1.0
                        pass1_workflow[f"353_{idx}"] = scene_353

                        scene_354 = json.loads(json.dumps(tpl_354))
                        scene_354["inputs"]["model"] = [f"353_{idx}", 0]
                        scene_354["inputs"]["lora_name"] = "LTX2.3-Licon-MSR-test_version.safetensors"
                        scene_354["inputs"]["strength_model"] = 1.0
                        pass1_workflow[f"354_{idx}"] = scene_354

                        scene_46 = json.loads(json.dumps(tpl_46))
                        scene_46["inputs"]["duration_frames"] = scene["_total_frames"]
                        scene_46["inputs"]["local_prompts"] = scene["_local_prompts_str"]
                        scene_46["inputs"]["timeline_data"] = scene["_timeline_data_str"]
                        scene_46["inputs"]["model"] = [f"354_{idx}", 0]
                        scene_46["inputs"]["clip"] = [f"107_{idx}", 1]
                        scene_46["inputs"]["custom_width"] = custom_w
                        scene_46["inputs"]["custom_height"] = custom_h
                        scene_46["inputs"]["frame_rate"] = 24 
                        pass1_workflow[f"46_{idx}"] = scene_46

                        # 🛡️ FIX 1 APPLIED: Safe Relative Paths for Strict Validation
                        scene_351 = json.loads(json.dumps(tpl_351))
                        scene_351["inputs"]["image_paths"] = scene["_img_dir"]
                        scene_351["inputs"]["width"] = custom_w
                        scene_351["inputs"]["height"] = custom_h
                        pass1_workflow[f"351_{idx}"] = scene_351

                        scene_94_5 = json.loads(json.dumps(tpl_94_5))
                        scene_94_5["inputs"]["positive"] = [f"46_{idx}", 1]
                        scene_94_5["inputs"]["negative"] = [f"200_{idx}", 0]
                        scene_94_5["inputs"]["frame_rate"] = [f"46_{idx}", 5]
                        pass1_workflow[f"94:5_{idx}"] = scene_94_5

                        # 🛡️ FIX 2 APPLIED: Clamped Combo-Box limits
                        scene_320 = json.loads(json.dumps(tpl_320))
                        scene_320["inputs"]["1"] = [f"351_{idx}", 0]
                        scene_320["inputs"]["2"] = [f"351_{idx}", 0]
                        scene_320["inputs"]["3"] = [f"351_{idx}", 0]
                        scene_320["inputs"]["4"] = [f"351_{idx}", 0]
                        scene_320["inputs"]["width"] = custom_w
                        scene_320["inputs"]["height"] = custom_h
                        scene_320["inputs"]["frame_count"] = scene["_msr_frames"]
                        pass1_workflow[f"320_{idx}"] = scene_320

                        scene_330 = json.loads(json.dumps(tpl_330))
                        scene_330["inputs"]["positive"] = [f"94:5_{idx}", 0]
                        scene_330["inputs"]["negative"] = [f"94:5_{idx}", 1]
                        scene_330["inputs"]["latent"] = [f"46_{idx}", 2]
                        scene_330["inputs"]["image"] = [f"320_{idx}", 0]
                        pass1_workflow[f"330_{idx}"] = scene_330

                        scene_352 = json.loads(json.dumps(tpl_352))
                        scene_352["inputs"]["positive"] = [f"330_{idx}", 0]
                        scene_352["inputs"]["negative"] = [f"330_{idx}", 1]
                        scene_352["inputs"]["latent"] = [f"330_{idx}", 2]
                        scene_352["inputs"]["multi_input"] = [f"351_{idx}", 0]
                        pass1_workflow[f"352_{idx}"] = scene_352

                        scene_300 = json.loads(json.dumps(tpl_300))
                        scene_300["inputs"]["model"] = [f"46_{idx}", 0]
                        scene_300["inputs"]["positive"] = [f"352_{idx}", 0]
                        scene_300["inputs"]["negative"] = [f"352_{idx}", 1]
                        scene_300["inputs"]["video_latent"] = [f"352_{idx}", 2]
                        scene_300["inputs"]["audio_latent"] = [f"46_{idx}", 3]
                        scene_300["inputs"]["guide_data"] = [f"46_{idx}", 4]
                        scene_300["inputs"]["frame_rate"] = [f"46_{idx}", 5]
                        scene_300["inputs"]["scene_id"] = str(idx)
                        pass1_workflow[f"300_{idx}"] = scene_300

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

                        if "103" in pass2_workflow: pass2_workflow["103"]["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                        if "102" in pass2_workflow: pass2_workflow["102"]["inputs"]["vae_name"] = "LTX23_audio_vae_bf16.safetensors"
                        if "94:105" in pass2_workflow: pass2_workflow["94:105"]["inputs"]["model_name"] = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
                        if "400" in pass2_workflow: pass2_workflow["400"]["inputs"]["cache_id"] = str(idx)

                        scene_seed = scene["_seed"]
                        if "94:28" in pass2_workflow: pass2_workflow["94:28"]["inputs"]["noise_seed"] = scene_seed
                        if "94:108" in pass2_workflow: pass2_workflow["94:108"]["inputs"]["noise_seed"] = scene_seed

                        if "94:11" in pass2_workflow and "inputs" in pass2_workflow["94:11"]:
                            pass2_workflow["94:11"]["inputs"]["steps"] = 12
                            if "denoise" in pass2_workflow["94:11"]["inputs"]: pass2_workflow["94:11"]["inputs"]["denoise"] = 1.0
                        if "94:54" in pass2_workflow and "inputs" in pass2_workflow["94:54"]:
                            pass2_workflow["94:54"]["inputs"]["steps"] = 16
                            if "denoise" in pass2_workflow["94:54"]["inputs"]: pass2_workflow["94:54"]["inputs"]["denoise"] = 0.42
                        if "94:49" in pass2_workflow and "inputs" in pass2_workflow["94:49"]: 
                            if "cfg" in pass2_workflow["94:49"]["inputs"]: pass2_workflow["94:49"]["inputs"]["cfg"] = 1.5
                        if "109" in pass2_workflow:
                            pass2_workflow["109"]["inputs"]["frame_rate"] = 24
                            if "pingpong" in pass2_workflow["109"]["inputs"]: pass2_workflow["109"]["inputs"]["pingpong"] = False
                        if "401" in pass2_workflow:
                            pass2_workflow["401"]["inputs"]["effect_strength"] = 0.6
                            pass2_workflow["401"]["inputs"]["pop_factor"] = 0.7

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
