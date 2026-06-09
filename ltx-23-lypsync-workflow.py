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
    "FORCE_REBUILD_INDEX": "432"  # ⚠️ Bumped for clean pipeline refresh
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
    "python3.12 -m pip install --no-cache-dir pandas numexpr pytz python-dateutil scipy matplotlib colorama torchvision librosa soundfile decord imageio scikit-image numba einops bitsandbytes rotary_embedding_torch"
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
    "python3.12 -m pip install --no-cache-dir transformers>=4.49.0",
    "echo '' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py",
    "echo 'sageattn_qk_int8_pv_fp16_triton = sageattn' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py",
    "echo 'import sys; sys.modules[\"torch\"].float8_e8m0fnu = getattr(sys.modules[\"torch\"], \"float8_e8m0fnu\", sys.modules[\"torch\"].float32)' >> /usr/local/lib/python3.12/site-packages/torch/__init__.py",
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
        
        print("🎨 Injecting Smart Nodes & RAM Caches...")
        custom_nodes_path = "/workspace/ComfyUI/custom_nodes/LTXCustomPipeline.py"
        with open(custom_nodes_path, "w") as f:
            f.write("""
import torch
import torchvision.transforms.functional as TF
import nodes
import comfy.utils
import comfy_extras.nodes_lt
import comfy.samplers

# ====================================================================
# ⚠️ CRITICAL AUDIO VAE BFLOAT16 TYPE-MATCHING HACK ⚠️
# ====================================================================
import comfy.ldm.lightricks.vae.causal_audio_autoencoder

_orig_causal_encode = comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.encode
def _patched_causal_encode(self, x):
    target_dtype = next(self.parameters()).dtype
    return _orig_causal_encode(self, x.to(target_dtype))
comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.encode = _patched_causal_encode

_orig_causal_decode = comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.decode
def _patched_causal_decode(self, z):
    target_dtype = next(self.parameters()).dtype
    return _orig_causal_decode(self, z.to(target_dtype))
comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.decode = _patched_causal_decode

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
        return ()

class MemoryCacheReader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"scene_id": ("STRING", {"default": "0"})}}
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "video_latent", "audio_latent")
    FUNCTION = "read_cache"
    CATEGORY = "LTXBatch"

    def read_cache(self, scene_id):
        global LTX_CACHE
        data = LTX_CACHE.get(str(scene_id))
        if data is None: raise ValueError(f"Cache for Scene {scene_id} not found in RAM!")
        return (None, data["positive"], data["negative"], data["video_latent"], data["audio_latent"])

# ====================================================================
# 🛡️ THE BULLETPROOF NESTED TENSOR FALLBACKS 🛡️
# Protects PyTorch 2.5 against crashes when AV Latents temporarily merge
# ====================================================================

class SafeLTXVScheduler(comfy_extras.nodes_lt.LTXVScheduler):
    def execute(self, steps, max_shift, base_shift, stretch, terminal, latent):
        s = latent["samples"]
        if getattr(s, "is_nested", False):
            vid_tensor = s.unbind()[0]
            dummy_latent = latent.copy()
            dummy_latent["samples"] = vid_tensor
            return super().execute(steps, max_shift, base_shift, stretch, terminal, dummy_latent)
        return super().execute(steps, max_shift, base_shift, stretch, terminal, latent)

_orig_generate_noise = comfy.samplers.Noise_RandomNoise.generate_noise
def _patched_generate_noise(self, latent):
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        tensors = samples.unbind()
        noises = [torch.randn_like(t) for t in tensors]
        try: return torch.nested.as_nested_tensor(noises)
        except Exception: return torch.nested.nested_tensor(noises)
    return _orig_generate_noise(self, latent)
comfy.samplers.Noise_RandomNoise.generate_noise = _patched_generate_noise

class SafeLTXVConcatAVLatent(comfy_extras.nodes_lt.LTXVConcatAVLatent):
    def execute(self, video_latent, audio_latent):
        vid_s = video_latent["samples"]
        aud_s = audio_latent["samples"]
        if getattr(vid_s, "is_nested", False): vid_s = vid_s.unbind()[0]
        if getattr(aud_s, "is_nested", False): aud_s = aud_s.unbind()[-1]
        aud_s = aud_s.to(dtype=vid_s.dtype, device=vid_s.device)
        padded_aud = aud_s
        while padded_aud.dim() < vid_s.dim(): padded_aud = padded_aud.unsqueeze(-1)
        while vid_s.dim() < padded_aud.dim(): vid_s = vid_s.unsqueeze(-1)
        try: res = torch.nested.as_nested_tensor([vid_s, padded_aud])
        except Exception: res = torch.nested.nested_tensor([vid_s, padded_aud])
        out = video_latent.copy()
        out["samples"] = res
        return (out,)

NODE_CLASS_MAPPINGS = {
    "MemoryCacheWriter": MemoryCacheWriter,
    "MemoryCacheReader": MemoryCacheReader, 
    "LTXVScheduler": SafeLTXVScheduler,
    "LTXVConcatAVLatent": SafeLTXVConcatAVLatent
}
""")

        print("🔗 Running Atomic Model Folder Linker for LTX 2.3 & CosyVoice3...")
        base_models_dir = "/workspace/ComfyUI/models"
        dirs = ["unet", "vae", "clip", "text_encoders", "checkpoints", "loras", "upscale_models", "latent_upscale_models", "cosyvoice", "melbandroformer", "diffusion_models"]
        for d in dirs: os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        cv_source = "/mnt/weights/cosyvoice3"
        cv_dest = os.path.join(base_models_dir, "cosyvoice", "Fun-CosyVoice3-0.5B")
        if os.path.exists(cv_source) and not os.path.exists(cv_dest):
            os.symlink(cv_source, cv_dest)

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                if "cosyvoice3" in root_dir.split(os.sep): 
                    continue 
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin", ".onnx", ".yaml", ".json")): continue
                    src_path = os.path.join(root_dir, filename)
                    for target_dir in dirs:
                        if target_dir == "cosyvoice": continue 
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
                    custom_w = int(body.get("custom_width", 384)) 
                    custom_h = int(body.get("custom_height", 672))

                    for idx, scene in enumerate(batch_scenes):
                        
                        async def download_asset(url, target_path):
                            if not url: return False
                            if "pub-4d91f4d3d0366568a54ffa32ffcb7bf4.r2.dev" in url:
                                key = url.split(".dev/")[-1]
                                if key.startswith("video-asset-files-storage-workflow/"):
                                    key = key.replace("video-asset-files-storage-workflow/", "", 1)
                                try:
                                    print(f"📥 Authenticated download via boto3 for private R2 asset: {key}")
                                    await asyncio.get_event_loop().run_in_executor(
                                        None, self.s3.download_file, "video-asset-files-storage-workflow", key, target_path
                                    )
                                    return True
                                except Exception as e:
                                    print(f"❌ Failed to download private R2 asset ({key}): {e}")
                                    return False
                            else:
                                try:
                                    async with session.get(url, timeout=60) as r:
                                        if r.status == 200:
                                            with open(target_path, "wb") as f: f.write(await r.read())
                                            return True
                                except Exception: pass
                            return False

                        has_two_chars = bool(scene.get("image2_url"))

                        img1_path = os.path.join(dynamic_guides_dir, f"char1_{idx}.png")
                        img2_path = os.path.join(dynamic_guides_dir, f"char2_{idx}.png")
                        mask1_path = os.path.join(dynamic_guides_dir, f"mask1_{idx}.png")
                        mask2_path = os.path.join(dynamic_guides_dir, f"mask2_{idx}.png")
                        
                        await download_asset(scene.get("image1_url"), img1_path)
                        if has_two_chars:
                            success2 = await download_asset(scene.get("image2_url"), img2_path)
                            if not success2: has_two_chars = False

                        from PIL import Image, ImageDraw
                        
                        # Process Image 1
                        if not os.path.exists(img1_path):
                            print(f"⚠️ Image 1 not found, generating safety dummy: {img1_path}")
                            Image.new('RGB', (custom_w, custom_h), color='black').save(img1_path)
                        else:
                            i = Image.open(img1_path).convert("RGB")
                            i.resize((custom_w, custom_h), Image.Resampling.LANCZOS).save(img1_path)

                        # Process Image 2
                        if not os.path.exists(img2_path):
                            Image.new('RGB', (custom_w, custom_h), color='black').save(img2_path)
                        else:
                            i = Image.open(img2_path).convert("RGB")
                            i.resize((custom_w, custom_h), Image.Resampling.LANCZOS).save(img2_path)

                        # GENERATE DYNAMIC MASKS (1 Person vs 2 Persons)
                        mask1 = Image.new('RGB', (custom_w, custom_h), color='white')
                        mask2 = Image.new('RGB', (custom_w, custom_h), color='black')

                        if has_two_chars:
                            # 2 Characters: Split the screen in half
                            draw1 = ImageDraw.Draw(mask1)
                            draw1.rectangle([custom_w // 2, 0, custom_w, custom_h], fill="black") # White left, Black right
                            
                            draw2 = ImageDraw.Draw(mask2)
                            draw2.rectangle([custom_w // 2, 0, custom_w, custom_h], fill="white") # Black left, White right

                        mask1.save(mask1_path)
                        mask2.save(mask2_path)

                        # Process Audio
                        spk1_path = os.path.join(dynamic_guides_dir, f"spk1_{idx}.wav")
                        spk2_path = os.path.join(dynamic_guides_dir, f"spk2_{idx}.wav")

                        await download_asset(scene.get("speaker1_audio_url"), spk1_path)
                        await download_asset(scene.get("speaker2_audio_url"), spk2_path)
                        
                        import soundfile as sf
                        import librosa
                        import numpy as np 
                        
                        for aud_p in [spk1_path, spk2_path]:
                            if not os.path.exists(aud_p):
                                print(f"⚠️ Audio not found, generating safety silence: {aud_p}")
                                dummy_data = np.zeros(16000, dtype=np.float32)
                                sf.write(aud_p, dummy_data, 16000)
                            else:
                                try:
                                    data, _ = librosa.load(aud_p, sr=16000, mono=True)
                                    sf.write(aud_p, data, 16000)
                                except Exception as e:
                                    print(f"❌ Corrupt audio replaced with silence: {e}")
                                    dummy_data = np.zeros(16000, dtype=np.float32)
                                    sf.write(aud_p, dummy_data, 16000)

                        total_frames = int(scene.get("total_frames", 161))

                        print(f"\n[Lypsync API] 🎬 Initiating SUBGRAPH 1 (Voice & Embeddings) for Scene {idx}...")
                        sg1 = json.loads(json.dumps(subgraph_1))
                        
                        if "1" in sg1: sg1["1"]["inputs"]["model_version"] = "Fun-CosyVoice3-0.5B"
                        if "369" in sg1: sg1["369"]["inputs"]["model_name"] = "MelBandRoformer_fp32.safetensors"
                        if "367" in sg1: sg1["367"]["inputs"]["ckpt_name"] = "ltx-2-3-22b-audio_vae.safetensors"
                        
                        if "368" in sg1:
                            sg1["368"]["inputs"]["clip_name1"] = "gemma_3_12B_it.safetensors"
                            sg1["368"]["inputs"]["clip_name2"] = "ltx-2-3-22b-text_encoder.safetensors"
                            sg1["368"]["inputs"]["type"] = "ltxv"

                        if "12" in sg1: sg1["12"]["inputs"]["text"] = scene.get("positive_prompt", "")
                        if "13" in sg1: sg1["13"]["inputs"]["text"] = scene.get("negative_prompt", "blurry, distorted, bad quality")
                        if "371" in sg1: sg1["371"]["inputs"]["dialog_text"] = scene.get("dialog_text", "")
                        if "374" in sg1: sg1["374"]["inputs"]["text"] = scene.get("speaker1_text", "Hello.")
                        if "375" in sg1: sg1["375"]["inputs"]["text"] = scene.get("speaker2_text", "Hello.")

                        if "365" in sg1: sg1["365"]["inputs"]["audio"] = f"dynamic_guides/spk1_{idx}.wav"
                        if "366" in sg1: sg1["366"]["inputs"]["audio"] = f"dynamic_guides/spk2_{idx}.wav"

                        if "14" in sg1:
                            sg1["14"]["inputs"]["width"] = custom_w
                            sg1["14"]["inputs"]["height"] = custom_h
                            sg1["14"]["inputs"]["length"] = total_frames

                        if "300" in sg1: sg1["300"]["inputs"]["scene_id"] = str(idx)

                        await self.execute_comfy_workflow(session, sg1)
                        await self.clear_comfy_memory(session, unload_models=False)

                        print(f"\n[Lypsync API] 🚀 Initiating SUBGRAPH 2 (Video Render) for Scene {idx}...")
                        sg2 = json.loads(json.dumps(subgraph_2))
                        out_dir = "/workspace/ComfyUI/output"
                        if os.path.exists(out_dir): shutil.rmtree(out_dir)
                        os.makedirs(out_dir)

                        scene_seed = scene.get("seed", int(time.time() * 1000) % 1000000)

                        if "100" in sg2: sg2["100"]["inputs"]["scene_id"] = str(idx)

                        if "103" in sg2: sg2["103"]["inputs"]["model_name"] = "ltx-2-3-22b-distilled-model.safetensors"
                        if "101" in sg2: sg2["101"]["inputs"]["vae_name"] = "ltx-2-3-22b-VAE.safetensors"
                        if "102" in sg2: sg2["102"]["inputs"]["ckpt_name"] = "ltx-2-3-22b-audio_vae.safetensors"

                        if "104" in sg2:
                            sg2["104"]["inputs"]["lora_1"] = "LTX2.3-IC-LORA-Dual-Character.safetensors"
                            sg2["104"]["inputs"]["strength_1"] = 1.0

                        if "137" in sg2:
                            sg2["137"]["inputs"]["pingpong"] = False
                            sg2["137"]["inputs"]["save_output"] = True

                        # DYNAMIC IMAGE AND MASK ROUTING
                        if "142" in sg2: sg2["142"]["inputs"]["image"] = f"dynamic_guides/char1_{idx}.png"
                        if "143" in sg2: sg2["143"]["inputs"]["image"] = f"dynamic_guides/mask1_{idx}.png"
                        if "144" in sg2: sg2["144"]["inputs"]["image"] = f"dynamic_guides/char2_{idx}.png"
                        if "145" in sg2: sg2["145"]["inputs"]["image"] = f"dynamic_guides/mask2_{idx}.png"

                        if "119" in sg2: sg2["119"]["inputs"]["noise_seed"] = scene_seed
                        if "118" in sg2: sg2["118"]["inputs"]["steps"] = 12

                        if "130" in sg2: sg2["130"]["inputs"]["noise_seed"] = scene_seed

                        await self.execute_comfy_workflow(session, sg2)

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
                yield b'{"status": "processing", "message": "Heartbeat... Keeping connection active"}\n' + (b" " * 1024)
                done, pending = await asyncio.wait([task], timeout=10.0)
                if task in done: break
            try:
                result = task.result()
                if isinstance(result, (dict, list)): yield json.dumps(result).encode("utf-8")
                else: yield str(result).encode("utf-8")
            except HTTPException as e: 
                yield json.dumps({"status": "error", "detail": str(e.detail)}).encode("utf-8")
            except Exception as e: 
                yield json.dumps({"status": "error", "detail": str(e)}).encode("utf-8")

        return StreamingResponse(stream_response(), media_type="application/json")
