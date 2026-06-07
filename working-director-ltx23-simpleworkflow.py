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
# PART 2 & 3: BASE IMAGE & OS CONFIGURATION
# ==============================================================================
base_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04", # Locked to 12.4 to perfectly match PyTorch wheels
    add_python="3.12"
).apt_install(
    "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0",
    "build-essential", "ninja-build", "cmake", "clang", "llvm",
    "libgoogle-perftools-dev" 
).env({
    "FORCE_REBUILD_INDEX": "405"  # Bumped to force Modal rebuild
})

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
# 🛡️ UPGRADED to PyTorch 2.6.0 to fix float8_e8m0fnu errors and unlock CUDA optimizations
torch_image = build_image.run_commands(
    "python3.12 -m pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124",
    "python3.12 -m pip install --no-cache-dir diffusers accelerate transformers==4.49.0 torchsde numpy==1.26.4 kornia==0.7.3",
    "python3.12 -m pip install --no-cache-dir sageattention==1.0.6"
)

clone_image = torch_image.run_commands(
    "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
    "GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    "git clone --depth 1 https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git /workspace/ComfyUI/custom_nodes/WhatDreamsCost-ComfyUI",
    "git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone --depth 1 https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes"
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
    env={"CUDA_HOME": "/usr/local/cuda", "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""), "FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": "8.9"}
)

# ==============================================================================
# PART 5: MODAL APP CONFIGURATION & CLOUD VOLUMES (L4 24GB GPU TARGET)
# ==============================================================================
app = modal.App("media-worker-ltx23")
weights_volume = modal.Volume.from_name("Ltx-23-model-weights-new", create_if_missing=False)

@app.cls(
    gpu="L4", # 🛡️ Target L4 GPU
    image=final_image,
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("custom-secret")],
    memory=8192, # 🛡️ Kept exactly at 8192 per instructions
    scaledown_window=12,
    timeout=3600
)
class LTX23Engine:

    def _log_reader(self):
        for line in iter(self.process.stdout.readline, ""):
            if line: print(f"[ComfyUI] {line.strip()}")

    async def _ram_squeezer(self):
        while True:
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1\n')
            except Exception: pass
            await asyncio.sleep(15)

    @modal.enter()
    def start_comfy(self):
        import boto3
        
        # 🔥 CUSTOM NODES: Two-Pass Cache Writers & VAE Armor Patch
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
        return {"required": {"image": ("IMAGE",), "target_brightness": ("FLOAT", {"default": 0.40, "min": 0.1, "max": 1.0, "step": 0.05}), "max_boost": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 3.0, "step": 0.1})}}
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
            "frame_rate": ("FLOAT", {"default": 25.0, "forceInput": True}),
            "scene_id": ("STRING", {"default": "0"})
        }}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "write_cache"
    CATEGORY = "LTXBatch"

    def write_cache(self, model, positive, negative, video_latent, audio_latent, guide_data, frame_rate, scene_id):
        global LTX_CACHE
        LTX_CACHE[str(scene_id)] = {
            "model": model, "positive": positive, "negative": negative,
            "video_latent": video_latent, "audio_latent": audio_latent,
            "guide_data": guide_data, "frame_rate": frame_rate
        }
        print(f"\\n[Two-Pass System] 💾 Saved POS & NEG Conditionings + Audio Latent for Scene {scene_id} into RAM\\n")
        return ()

class MemoryCacheReader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"scene_id": ("STRING", {"default": "0"})}}
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "LATENT", "GUIDE_DATA", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "video_latent", "audio_latent", "guide_data", "frame_rate")
    FUNCTION = "read_cache"
    CATEGORY = "LTXBatch"

    def read_cache(self, scene_id):
        global LTX_CACHE
        data = LTX_CACHE.get(str(scene_id))
        if data is None:
            raise ValueError(f"Cache for Scene {scene_id} not found in RAM! Text Encoder Pass failed.")
        return (data["model"], data["positive"], data["negative"], data["video_latent"], data["audio_latent"], data["guide_data"], data["frame_rate"])

class FastVAEDecode(nodes.VAEDecode):
    def decode(self, vae, samples):
        try:
            return (vae.decode_tiled(samples["samples"], tile_x=512, tile_y=512), )
        except Exception:
            return super().decode(vae, samples)

NODE_CLASS_MAPPINGS = {
    "LTXColorFixer": LTXColorFixer, "MemoryCacheWriter": MemoryCacheWriter,
    "MemoryCacheReader": MemoryCacheReader, "VAEDecode": FastVAEDecode
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

        print("🚀 Launching Two-Pass Server Engine on L4 GPU (24GB)...")
        os.makedirs("/tmp/comfy_swap", exist_ok=True)

        env_vars = os.environ.copy()
        env_vars["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
        env_vars["TORCH_NUM_THREADS"] = "1"
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:64"
        env_vars["CUDA_MODULE_LOADING"] = "LAZY" 
        
        # 🛡️ ARCHITECTURAL FIX: Force GPU array concats into system CPU RAM to prevent the exact 'torch.cat' OutOfMemoryError.
        env_vars["COMFYUI_INTERMEDIATE_DEVICE"] = "cpu"
        
        # 🛡️ ARCHITECTURAL FIX: Replaced disable-smart-memory with --novram flag. 
        # This unloads models completely between micro-batches keeping 24GB canvas perfectly clean.
        self.process = subprocess.Popen([
            "python3.12", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--mmap-torch-files", "--novram", "--temp-directory", "/tmp/comfy_swap", 
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
            except Exception: time.sleep(2)
                
        if not comfy_ready: os._exit(1)
        print("✅ Base pipeline active. Awaiting Two-Pass API Batch triggers.")

    async def clear_comfy_memory(self, session, unload_models=False):
        try:
            async with session.post("http://127.0.0.1:8188/free", json={"unload_models": unload_models, "free_memory": True}) as r: await r.read()
        except Exception: pass
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
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
                                if msg[0] == "execution_error": raise HTTPException(status_code=500, detail=f"ComfyUI execution error: {msg[1]}")
                        return step_data
            if self.process.poll() is not None: raise HTTPException(status_code=500, detail="ComfyUI server process crashed.")
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

            dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
            if os.path.exists(dynamic_guides_dir): shutil.rmtree(dynamic_guides_dir)
            os.makedirs(dynamic_guides_dir, exist_ok=True)

            ram_task = asyncio.create_task(self._ram_squeezer())
            generated_outputs = []

            try:
                async with aiohttp.ClientSession() as session:
                    custom_w = body.get("custom_width", 704)
                    custom_h = body.get("custom_height", 1280)

                    # ==============================================================================
                    # PRE-COMPUTE: Download Images & Audio Context Setup
                    # ==============================================================================
                    for idx, scene in enumerate(batch_scenes):
                        # --- Image Handling ---
                        target_img_name = f"guide_anchor_{idx}.png"
                        target_path = os.path.join(dynamic_guides_dir, target_img_name)
                        image_url = scene.get("image_url", "")
                        if image_url:
                            try:
                                async with session.get(image_url, timeout=120) as r:
                                    if r.status == 200:
                                        with open(target_path, "wb") as f: f.write(await r.read())
                            except Exception: pass
                        
                        if not os.path.exists(target_path):
                            from PIL import Image
                            Image.new('RGB', (custom_w, custom_h), color='black').save(target_path)
                            
                        from PIL import Image
                        img = Image.open(target_path).convert("RGB")
                        img = img.resize((custom_w, custom_h), Image.Resampling.LANCZOS)
                        img.save(target_path)
                        
                        segments = []
                        for frame in [0, 1]:
                            with open(target_path, "rb") as f:
                                b64_img = base64.b64encode(f.read()).decode("utf-8")
                                segments.append({"frame": frame, "image": f"data:image/png;base64,{b64_img}"})

                        # --- AUDIO LOGIC FIX (Crucial for Lip Sync) ---
                        audio_url = scene.get("audio_url", "")
                        has_audio = False
                        audio_path = None
                        if audio_url:
                            audio_ext = "wav"
                            audio_path = os.path.join(dynamic_guides_dir, f"audio_{idx}.{audio_ext}")
                            try:
                                async with session.get(audio_url, timeout=120) as r:
                                    if r.status == 200:
                                        with open(audio_path, "wb") as f: f.write(await r.read())
                                        has_audio = True
                            except Exception as e: print(f"Audio download failed: {e}")

                        # Inject Local Audio File directly into Director Timeline
                        timeline_data = {"segments": segments, "audioSegments": []}
                        if has_audio:
                            timeline_data["audioSegments"].append({"audio": audio_path, "start": 0})
                        
                        scene["_timeline_data_str"] = json.dumps(timeline_data)
                        scene["_has_audio"] = has_audio

                        # --- Frames & Duration ---
                        actions = scene.get("kinetic_actions", ["The subject moves dynamically across the cinematic scene."])
                        total_words = sum(len(str(a).split()) for a in actions)
                        seconds = max(total_words / (130 / 60.0), 2.5)  
                        raw_frames = seconds * 25 # 🛡️ Forced to 25 FPS for Talking Head support
                        total_frames = int(math.ceil((raw_frames - 1) / 8) * 8 + 1)
                        total_frames = max(33, min(total_frames, 257))
                        
                        num_actions = len(actions)
                        keyframe_steps = [int(i * (total_frames - 1) / max(1, num_actions - 1)) for i in range(num_actions)]

                        # --- Dynamic Prompt Generation ---
                        static_env = f"{scene.get('subject', '')} {scene.get('style', '')} {scene.get('background', '')} {scene.get('lighting', '')}".strip()
                        speech_transcript = scene.get("speech_transcript", "")
                        
                        local_prompts_list = []
                        for step_frame, action_text in zip(keyframe_steps, actions):
                            action_cam = f"{action_text} {scene.get('camera', '')}".strip()
                            fused_prompt = f"{action_cam}. Cinematic environment and styling: {static_env}"
                            
                            # 🛡️ IF Audio is present, inject Talking Head Trigger & Transcript
                            if has_audio:
                                fused_prompt = f"OHWXPERSON, {fused_prompt}. The person is talking, and says: \"{speech_transcript}\""
                                
                            local_prompts_list.append(f"{step_frame}: {fused_prompt}")
                            
                        scene["_local_prompts_str"] = "\n".join(local_prompts_list)
                        scene["_total_frames"] = total_frames
                        scene["_seed"] = scene.get("seed", int(time.time() * 1000) % 1000000)

                    # ==============================================================================
                    # PASS 1: TEXT ENCODING & DYNAMIC LORA MATRIX (MICRO-BATCHED)
                    # ==============================================================================
                    # 🛡️ ARCHITECTURAL FIX: Executes sequentially, removing multi-scene node parallelization VRAM spikes
                    print("\n[Two-Pass System] 🎬 PASS 1 START: Initiating Text Encoding Sequentially (Micro-Batching)...")
                    
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
                        print(f"  -> Encoding Logic Matrix for Scene {idx+1}/{len(batch_scenes)}...")
                        
                        # Generate a fresh, clean payload just for this specific scene
                        pass1_workflow = json.loads(json.dumps(subgraph_1))
                        
                        # 🛡️ Inject FP4 Model for VRAM efficiency
                        if "98" in pass1_workflow: 
                            pass1_workflow["98"]["inputs"]["unet_name"] = "LTX-2.3-22B-Distilled-FP4ME.safetensors"
                            pass1_workflow["98"]["inputs"]["weight_dtype"] = "fp8_e4m3fn" 
                            
                        if "97" in pass1_workflow: pass1_workflow["97"]["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                        if "102" in pass1_workflow: pass1_workflow["102"]["inputs"]["vae_name"] = "LTX23_audio_vae_bf16.safetensors"
                        if "101" in pass1_workflow: 
                            pass1_workflow["101"]["inputs"]["clip_name1"] = "gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors"
                            pass1_workflow["101"]["inputs"]["clip_name2"] = "ltx-2.3_text_projection_bf16.safetensors"

                        # 1. 🛡️ DYNAMIC LORA LOGIC (8 Slots Total mapped directly)
                        if "107" in pass1_workflow:
                            lora_stack = [
                                ("LTX_2.3_Crisp_Enhance_Style_LoRa.safetensors", 0.5),
                                ("LTX_2.3_Soft_Enhance_Style_LoRa.safetensors", 0.5),
                                ("VBVR-official-comfyui.safetensors", 0.7),
                                ("LTX-2.3_Cinematic_hardcut.safetensors", 0.6)
                            ]
                            if scene["_has_audio"]:
                                lora_stack.append(("LTX_2.3_22b_AV_LoRA_talking_head.safetensors", 1.0))
                                lora_stack.append(("LTX_2.3_RL_OmniNFT_LoRa.safetensors", 0.8))
                                
                            cam_string = scene.get("camera", "")
                            if cam_string:
                                for c in [x.strip().lower() for x in cam_string.split("+")]:
                                    if c in camera_loras_map and len(lora_stack) < 8:
                                        lora_stack.append((camera_loras_map[c], 1.0))
                            
                            for i in range(8):
                                slot = i + 1
                                if i < len(lora_stack):
                                    pass1_workflow["107"]["inputs"][f"lora_{slot}"] = lora_stack[i][0]
                                    pass1_workflow["107"]["inputs"][f"strength_{slot}"] = lora_stack[i][1]
                                    pass1_workflow["107"]["inputs"][f"enabled_{slot}"] = True
                                else:
                                    pass1_workflow["107"]["inputs"][f"lora_{slot}"] = "__none__"
                                    pass1_workflow["107"]["inputs"][f"strength_{slot}"] = 0.0
                                    pass1_workflow["107"]["inputs"][f"enabled_{slot}"] = False

                        # 2. 🛡️ Negative Prompt Processing
                        if "200" in pass1_workflow:
                            default_neg = "no humans, bad quality, distorted, blurry, watermark"
                            pass1_workflow["200"]["inputs"]["text"] = scene.get("negative_prompt", default_neg)

                        # 3. LTX Director
                        if "46" in pass1_workflow:
                            pass1_workflow["46"]["inputs"]["duration_frames"] = scene["_total_frames"]
                            pass1_workflow["46"]["inputs"]["local_prompts"] = scene["_local_prompts_str"]
                            pass1_workflow["46"]["inputs"]["timeline_data"] = scene["_timeline_data_str"]
                            pass1_workflow["46"]["inputs"]["custom_width"] = custom_w
                            pass1_workflow["46"]["inputs"]["custom_height"] = custom_h
                            pass1_workflow["46"]["inputs"]["frame_rate"] = 25

                        # 4. Memory Writer - Assign to RAM Global Cache per index
                        if "300" in pass1_workflow:
                            pass1_workflow["300"]["inputs"]["scene_id"] = str(idx)

                        # Execute specifically for this single scene, saving it, then forcefully wiping VRAM.
                        await self.execute_comfy_workflow(session, pass1_workflow)
                        await self.clear_comfy_memory(session, unload_models=True) # Unload heavily after each loop to keep L4 clear.

                    # ==============================================================================
                    # PASS 2: THE MACRO-GRAPH SAMPLING BLAST (Single API Payload)
                    # ==============================================================================
                    # 🛡️ TRUE SINGLE-SHOT EXECUTION: Compiling all subgraphs into ONE network payload 
                    # so ComfyUI locks the UNet in VRAM for all clips sequentially without dropping it.
                    print(f"\n[Two-Pass System] 🚀 PASS 2 START: Constructing Macro-Graph for {len(batch_scenes)} Scenes...")
                    
                    macro_workflow = {}
                    out_dir = "/workspace/ComfyUI/output"
                    if os.path.exists(out_dir): shutil.rmtree(out_dir)
                    os.makedirs(out_dir)

                    for idx, scene in enumerate(batch_scenes):
                        print(f"  -> Merging Scene [{idx+1}/{len(batch_scenes)}] into Macro-Graph payload...")
                        scene_pass2 = json.loads(json.dumps(subgraph_2))
                        original_keys = list(scene_pass2.keys())
                        scene_seed = scene["_seed"]
                        fixer_id = f"9999_color_fixer_{idx}"

                        # Stage standard inputs inside the template mapping
                        if "103" in scene_pass2: scene_pass2["103"]["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                        if "102" in scene_pass2: scene_pass2["102"]["inputs"]["vae_name"] = "LTX23_audio_vae_bf16.safetensors"
                        if "94:105" in scene_pass2: scene_pass2["94:105"]["inputs"]["model_name"] = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

                        # Retrieve correct scene conditioning from RAM Cache
                        if "400" in scene_pass2: scene_pass2["400"]["inputs"]["scene_id"] = str(idx)

                        if "94:28" in scene_pass2: scene_pass2["94:28"]["inputs"]["noise_seed"] = scene_seed
                        if "94:108" in scene_pass2: scene_pass2["94:108"]["inputs"]["noise_seed"] = scene_seed

                        if "94:11" in scene_pass2 and "inputs" in scene_pass2["94:11"]:
                            scene_pass2["94:11"]["inputs"]["steps"] = 12
                            if "denoise" in scene_pass2["94:11"]["inputs"]: scene_pass2["94:11"]["inputs"]["denoise"] = 1.0
                        
                        if "94:54" in scene_pass2 and "inputs" in scene_pass2["94:54"]:
                            scene_pass2["94:54"]["inputs"]["steps"] = 16
                            if "denoise" in scene_pass2["94:54"]["inputs"]: scene_pass2["94:54"]["inputs"]["denoise"] = 0.42
                            
                        if "94:49" in scene_pass2 and "inputs" in scene_pass2["94:49"]: 
                            if "cfg" in scene_pass2["94:49"]["inputs"]: scene_pass2["94:49"]["inputs"]["cfg"] = 1.5

                        # 🛡️ Route Outputs Safely
                        if "109" in scene_pass2:
                            scene_pass2["109"]["inputs"]["frame_rate"] = 25
                            if "pingpong" in scene_pass2["109"]["inputs"]: scene_pass2["109"]["inputs"]["pingpong"] = False
                            # Tag output file dynamically so it maps safely
                            scene_pass2["109"]["inputs"]["filename_prefix"] = f"scene_{idx}_output"

                        if "109" in scene_pass2 and "images" in scene_pass2["109"]["inputs"]:
                            original_image_source = scene_pass2["109"]["inputs"]["images"]
                            scene_pass2[fixer_id] = {
                                "class_type": "LTXColorFixer", 
                                "inputs": {
                                    "image": original_image_source, 
                                    "target_brightness": 0.40, 
                                    "max_boost": 2.0
                                }
                            }
                            scene_pass2["109"]["inputs"]["images"] = [fixer_id, 0]

                        # Re-Map Internal Link Pointers explicitly with Suffixes
                        for old_node_id, node_data in scene_pass2.items():
                            new_node_id = f"{old_node_id}_{idx}" if old_node_id in original_keys else old_node_id

                            if "inputs" in node_data:
                                for input_key, input_value in node_data["inputs"].items():
                                    if isinstance(input_value, list) and len(input_value) == 2 and str(input_value[0]) in original_keys:
                                        node_data["inputs"][input_key][0] = f"{input_value[0]}_{idx}"

                            macro_workflow[new_node_id] = node_data

                    # Fire off the Single Shot Payload
                    print(f"📡 Executing Macro-Graph... ComfyUI UNet will load ONCE and persist for all batches.")
                    await self.execute_comfy_workflow(session, macro_workflow)

                    print("\n📥 Macro-Graph Generation Completed! Securing Media Outputs...")
                    for idx, scene in enumerate(batch_scenes):
                        scene_files = []
                        for root_p, _, filenames in os.walk(out_dir):
                            for name in filenames:
                                if name.startswith(f"scene_{idx}_output") and name.endswith((".mp4", ".gif", ".webm")): 
                                    scene_files.append(os.path.join(root_p, name))

                        if not scene_files:
                            print(f"❌ Error: Output missing for Scene {idx}")
                            continue

                        scene_files.sort(key=os.path.getmtime)
                        target_video_file = scene_files[-1]
                        saved_filename = os.path.basename(target_video_file)

                        target_key = f"{date_folder}/generated clips/{int(time.time())}_{scene.get('name', f'Clip_{idx}')}_{saved_filename}"
                        print(f"📤 Syncing Finished Asset for Scene {idx} to R2: {target_key}")
                        
                        await asyncio.get_event_loop().run_in_executor(None, self.s3.upload_file, target_video_file, "video-asset-files-storage-workflow", target_key)
                        
                        generated_outputs.append({
                            "scene": scene.get("name", f"Clip_{idx+1}"),
                            "status": "success",
                            "file_key": target_key,
                            "public_url": f"https://pub-4d91f4d3d0366568a54ffa32ffcb7bf4.r2.dev/{target_key}",
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
                if isinstance(result, (dict, list)): yield json.dumps(result).encode("utf-8")
                else: yield str(result).encode("utf-8")
            except HTTPException as e: yield json.dumps({"status": "error", "detail": e.detail}).encode("utf-8")
            except Exception as e: yield json.dumps({"status": "error", "detail": str(e)}).encode("utf-8")

        return StreamingResponse(stream_response(), media_type="application/json")
