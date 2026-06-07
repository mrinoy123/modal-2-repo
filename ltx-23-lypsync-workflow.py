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
    "nvidia/cuda:12.5.1-devel-ubuntu24.04",
    add_python="3.12"
).apt_install(
    "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0",
    "build-essential", "ninja-build", "cmake", "clang", "llvm",
    "libgoogle-perftools-dev" 
).env({
    "FORCE_REBUILD_INDEX": "422"  # Bumped for caching reset
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
torch_image = build_image.run_commands(
    "python3.12 -m pip install --no-cache-dir torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124",
    "python3.12 -m pip install --no-cache-dir diffusers accelerate transformers==4.48.3 torchsde numpy==1.26.4 kornia==0.7.3",
    "python3.12 -m pip install --no-cache-dir sageattention==1.0.6"
)

clone_image = torch_image.run_commands(
    "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
    "GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    "git clone --depth 1 https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git /workspace/ComfyUI/custom_nodes/WhatDreamsCost-ComfyUI",
    "git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone --depth 1 https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes",
    # ⚠️ Added Core Audio Processing Suites for Voice Cloning & RoFormer
    "git clone --depth 1 https://github.com/kijai/ComfyUI-MelBandRoFormer.git /workspace/ComfyUI/custom_nodes/ComfyUI-MelBandRoFormer",
    "git clone --depth 1 https://github.com/filliptm/ComfyUI_FL-CosyVoice3.git /workspace/ComfyUI/custom_nodes/ComfyUI_FL-CosyVoice3"
)

deps_image = clone_image.run_commands(
    "sed -i '/torch/d' /workspace/ComfyUI/requirements.txt",
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec sed -i '/torch/d' {} \;",
    "python3.12 -m pip install --no-cache-dir -r /workspace/ComfyUI/requirements.txt",
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec python3.12 -m pip install --no-cache-dir -r {} \;"
)

final_image = deps_image.run_commands(
    "python3.12 -m pip install --no-cache-dir transformers==4.48.3",
    "echo '' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py",
    "echo 'sageattn_qk_int8_pv_fp16_triton = sageattn' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py",
    env={"CUDA_HOME": "/usr/local/cuda", "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""), "FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": "8.9"}
)

# ==============================================================================
# PART 5: MODAL APP CONFIGURATION & CLOUD VOLUMES 
# ==============================================================================
app = modal.App("media-worker-ltx23-lypsync")
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
class LTX23LypsyncEngine:

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
        
        print("🎨 Injecting Smart Nodes, Caches & Lypsync Memory Protections...")
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
        return {"required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "video_latent": ("LATENT",),
            "audio_latent": ("LATENT",),
            "scene_id": ("STRING", {"default": "0"})
        }}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "write_cache"
    CATEGORY = "LTXBatch"

    def write_cache(self, positive, negative, video_latent, audio_latent, scene_id):
        global LTX_CACHE
        LTX_CACHE[str(scene_id)] = {
            "positive": positive, "negative": negative,
            "video_latent": video_latent, "audio_latent": audio_latent
        }
        print(f"\\n[Lypsync System] 💾 Saved Conditionings & Dual-Audio Latent for Scene {scene_id} into RAM\\n")
        return ()

class MemoryCacheReader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"scene_id": ("STRING", {"default": "0"})}}
    # ⚠️ CRITICAL FIX: We inject a dummy 'MODEL' output at index 0 
    # This prevents your exact Subgraph 2 JSON links [301, 1], [301, 2], etc. from misaligning!
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "video_latent", "audio_latent")
    FUNCTION = "read_cache"
    CATEGORY = "LTXBatch"

    def read_cache(self, scene_id):
        global LTX_CACHE
        data = LTX_CACHE.get(str(scene_id))
        if data is None:
            raise ValueError(f"Cache for Scene {scene_id} not found in RAM! Subgraph 1 Pass failed.")
        return (None, data["positive"], data["negative"], data["video_latent"], data["audio_latent"])

class FastVAEDecode(nodes.VAEDecode):
    def decode(self, vae, samples):
        try:
            return (vae.decode_tiled(samples["samples"], tile_x=512, tile_y=512), )
        except Exception:
            return super().decode(vae, samples)

NODE_CLASS_MAPPINGS = {
    "MemoryCacheWriter": MemoryCacheWriter,
    "MemoryCacheReader": MemoryCacheReader, 
    "VAEDecode": FastVAEDecode
}
""")

        print("🔗 Running Atomic Model Folder Linker for LTX 2.3 & CosyVoice3...")
        base_models_dir = "/workspace/ComfyUI/models"
        dirs = ["unet", "vae", "clip", "text_encoders", "checkpoints", "loras", "upscale_models", "latent_upscale_models", "cosyvoice", "melbandroformer"]
        for d in dirs: os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        # ⚠️ CRITICAL FIX: CosyVoice strictly relies on the internal architecture alongside its `.yaml` & `.onnx` configurations.
        # Direct folder symlinking is utilized here specifically so that its internal structure isn't flattened.
        cv_source = "/mnt/weights/cosyvoice3"
        cv_dest = os.path.join(base_models_dir, "cosyvoice", "Fun-CosyVoice3-0.5B")
        if os.path.exists(cv_source) and not os.path.exists(cv_dest):
            os.symlink(cv_source, cv_dest)
            print(f"🔗 Symlinked strictly structured CosyVoice3 folder to {cv_dest}")

        # Fallback flat file linker for everything else natively.
        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                # Avoid destroying the folder tree if it's already mapped
                if "cosyvoice3" in root_dir.split(os.sep): 
                    continue 
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin", ".onnx", ".yaml", ".json")): continue
                    src_path = os.path.join(root_dir, filename)
                    for target_dir in dirs:
                        if target_dir == "cosyvoice": continue # Skip flattened link for cosyvoice 
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

        print("🚀 Launching Lypsync Server Engine on L40S GPU...")
        os.makedirs("/tmp/comfy_swap", exist_ok=True)

        env_vars = os.environ.copy()
        env_vars["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
        env_vars["TORCH_NUM_THREADS"] = "1"
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
            except Exception: time.sleep(2)
                
        if not comfy_ready: os._exit(1)
        print("✅ Base pipeline active. Awaiting Dual-Subgraph Triggers.")

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
    # PART 6: LYPSYNC BATCH ENDPOINT
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
                    # 🛡️ 9:16 Aspect Ratio Enforced (Divisible by 32 required for VAE) 
                    custom_w = body.get("custom_width", 384) 
                    custom_h = body.get("custom_height", 672)

                    for idx, scene in enumerate(batch_scenes):
                        
                        # --- ASSET DOWNLOAD UTILITIES ---
                        async def download_asset(url, target_path):
                            if url:
                                try:
                                    async with session.get(url, timeout=60) as r:
                                        if r.status == 200:
                                            with open(target_path, "wb") as f: f.write(await r.read())
                                            return True
                                except Exception: pass
                            return False

                        # 1️⃣ Dual Image Preparation
                        img1_path = os.path.join(dynamic_guides_dir, f"char1_{idx}.png")
                        img2_path = os.path.join(dynamic_guides_dir, f"char2_{idx}.png")
                        
                        await download_asset(scene.get("image1_url"), img1_path)
                        await download_asset(scene.get("image2_url"), img2_path)
                        
                        from PIL import Image
                        for img_p in [img1_path, img2_path]:
                            if not os.path.exists(img_p):
                                Image.new('RGB', (custom_w, custom_h), color='black').save(img_p)
                            else:
                                i = Image.open(img_p).convert("RGB")
                                i.resize((custom_w, custom_h), Image.Resampling.LANCZOS).save(img_p)

                        # 2️⃣ Dual Audio Processing (16kHz Resampling for CosyVoice/LTX VAE compatibility)
                        spk1_path = os.path.join(dynamic_guides_dir, f"spk1_{idx}.wav")
                        spk2_path = os.path.join(dynamic_guides_dir, f"spk2_{idx}.wav")

                        await download_asset(scene.get("speaker1_audio_url"), spk1_path)
                        await download_asset(scene.get("speaker2_audio_url"), spk2_path)
                        
                        import soundfile as sf
                        import librosa
                        for aud_p in [spk1_path, spk2_path]:
                            if os.path.exists(aud_p):
                                data, _ = librosa.load(aud_p, sr=16000, mono=True)
                                sf.write(aud_p, data, 16000)

                        # Frames control
                        total_frames = scene.get("total_frames", 161) # 161 frames is ~6 seconds at 25fps

                        # ==============================================================================
                        # SUBGRAPH 1: TEXT, ZERO-SHOT VOICE CLONING & COMPRESSION
                        # ==============================================================================
                        print(f"\n[Lypsync API] 🎬 Initiating SUBGRAPH 1 (Voice & Embeddings) for Scene {idx}...")
                        sg1 = json.loads(json.dumps(subgraph_1))
                        
                        # Model Hookups (Correctly tailored against Subgraph 1 API references)
                        if "1" in sg1: 
                            sg1["1"]["inputs"]["model_version"] = "Fun-CosyVoice3-0.5B"
                        if "369" in sg1: 
                            sg1["369"]["inputs"]["model_name"] = "MelBandRoformer_fp32.safetensors"
                        if "367" in sg1: 
                            sg1["367"]["inputs"]["ckpt_name"] = "LTX23_audio_vae_bf16.safetensors"
                        if "368" in sg1:
                            sg1["368"]["inputs"]["clip_name1"] = "gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors"
                            sg1["368"]["inputs"]["clip_name2"] = "ltx-2.3_text_projection_bf16.safetensors"

                        # Text & Script Mapping
                        if "12" in sg1: sg1["12"]["inputs"]["text"] = scene.get("positive_prompt", "")
                        if "13" in sg1: sg1["13"]["inputs"]["text"] = scene.get("negative_prompt", "blurry, distorted, bad quality")
                        if "371" in sg1: sg1["371"]["inputs"]["dialog_text"] = scene.get("dialog_text", "")
                        if "374" in sg1: sg1["374"]["inputs"]["text"] = scene.get("speaker1_text", "")
                        if "375" in sg1: sg1["375"]["inputs"]["text"] = scene.get("speaker2_text", "")

                        # Audio References Input
                        if "365" in sg1: sg1["365"]["inputs"]["audio"] = f"dynamic_guides/spk1_{idx}.wav"
                        if "366" in sg1: sg1["366"]["inputs"]["audio"] = f"dynamic_guides/spk2_{idx}.wav"

                        # Canvas Resolution Link
                        if "14" in sg1:
                            sg1["14"]["inputs"]["width"] = custom_w
                            sg1["14"]["inputs"]["height"] = custom_h
                            sg1["14"]["inputs"]["length"] = total_frames

                        # Memory Writer Index Map
                        if "300" in sg1: sg1["300"]["inputs"]["scene_id"] = str(idx)

                        await self.execute_comfy_workflow(session, sg1)
                        await self.clear_comfy_memory(session, unload_models=False)

                        # ==============================================================================
                        # SUBGRAPH 2: DUAL-CHARACTER VIDEO GENERATION & UPSCALE
                        # ==============================================================================
                        print(f"\n[Lypsync API] 🚀 Initiating SUBGRAPH 2 (Video Render) for Scene {idx}...")
                        sg2 = json.loads(json.dumps(subgraph_2))
                        out_dir = "/workspace/ComfyUI/output"
                        if os.path.exists(out_dir): shutil.rmtree(out_dir)
                        os.makedirs(out_dir)

                        scene_seed = scene.get("seed", int(time.time() * 1000) % 1000000)

                        if "301" in sg2: sg2["301"]["inputs"]["scene_id"] = str(idx)

                        # Base Models & Quantization Targets (Correctly tailored against Subgraph 2 API References)
                        if "422" in sg2: sg2["422"]["inputs"]["model_name"] = "ltx-2.3-22b-distilled-fp8.safetensors"
                        if "428" in sg2: sg2["428"]["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                        if "431" in sg2: sg2["431"]["inputs"]["ckpt_name"] = "LTX23_audio_vae_bf16.safetensors"

                        # IC-LoRA Configuration
                        if "426" in sg2:
                            sg2["426"]["inputs"]["lora_1"] = "LTX2.3-IC-LORA-Dual-Character.safetensors"
                            sg2["426"]["inputs"]["strength_1"] = 1.0

                        # Crop Guides Setup
                        if "429" in sg2: sg2["429"]["inputs"]["image"] = f"dynamic_guides/char1_{idx}.png"
                        if "430" in sg2: sg2["430"]["inputs"]["image"] = f"dynamic_guides/char2_{idx}.png"

                        # Primary Rendering Loop (Fixed targeted node reference mapping for seed payload)
                        if "413" in sg2: sg2["413"]["inputs"]["noise_seed"] = scene_seed
                        if "412" in sg2: sg2["412"]["inputs"]["steps"] = 12

                        # Stage 2 Execution (Protecting the generated Lip Sync)
                        if "502" in sg2: sg2["502"]["inputs"]["noise_seed"] = scene_seed
                        if "501" in sg2:
                            sg2["501"]["inputs"]["steps"] = 8
                            # Ensuring Stage 2 denoise/terminal is low to avoid overwriting mouths
                            if "terminal" in sg2["501"]["inputs"]: sg2["501"]["inputs"]["terminal"] = 0.1 

                        await self.execute_comfy_workflow(session, sg2)

                        # Output Sync
                        output_files = []
                        for root_p, _, filenames in os.walk(out_dir):
                            for name in filenames:
                                if name.endswith((".mp4", ".gif", ".webm")): output_files.append(os.path.join(root_p, name))

                        if not output_files: raise Exception("Inference finished but no output media files were detected.")
                        
                        output_files.sort(key=os.path.getmtime)
                        target_video_file = output_files[-1]
                        saved_filename = os.path.basename(target_video_file)

                        target_key = f"{date_folder}/lypsync_clips/{int(time.time())}_{scene.get('name', 'clip')}_{saved_filename}"
                        print(f"📤 Syncing Finished Asset to R2: {target_key}")
                        
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
