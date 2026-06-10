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
import warnings
from urllib.parse import urlparse
from fastapi import Request, Response, HTTPException, Header
from fastapi.responses import StreamingResponse
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning, module="numba")

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
    "FORCE_REBUILD_INDEX": "455"  # Bumped for Memory Cache purity updates
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
    "git clone --depth 1 https://github.com/filliptm/ComfyUI_FL-CosyVoice3.git /workspace/ComfyUI/custom_nodes/ComfyUI_FL-CosyVoice3",
    "git clone --depth 1 https://github.com/kijai/ComfyUI-PromptRelay.git /workspace/ComfyUI/custom_nodes/ComfyUI-PromptRelay"
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
app = modal.App("media-worker-ltx23-lypsync-v2")
weights_volume = modal.Volume.from_name("Ltx-23-model-weights-new", create_if_missing=False)

@app.cls(
    gpu="L40S", 
    image=final_image,
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("custom-secret")],
    memory=8192, 
    scaledown_window=8,
    timeout=3600
)
class LTX23LypsyncEngineV2:

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
        
        print("🎨 Building Strictly Pointer-Pass Memory Nodes...")
        os.makedirs("/workspace/ComfyUI/custom_nodes/LTXCustomPipeline", exist_ok=True)
        custom_nodes_path = "/workspace/ComfyUI/custom_nodes/LTXCustomPipeline/__init__.py"
        with open(custom_nodes_path, "w") as f:
            f.write("""
import torch
import torchvision.transforms.functional as TF
import nodes
import comfy.utils
import comfy.samplers
import comfy.model_patcher

LTX_CACHE = {}

# ✅ FIX 1: We completely removed move_to_cpu. 
# PromptRelay uses custom object subclasses. Casting them to dicts/lists strips their identity 
# causing the UNet to read empty conditions -> resulting in pure static noise!
# Storing exact RAM pointers prevents data destruction.

class MemoryCacheWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),  
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "video_latent": ("LATENT",),
            "audio_latent": ("LATENT",),
        }, "optional": {
            "scene_id": ("STRING", {"default": "0"})
        }}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "write_cache"
    CATEGORY = "LTXBatch"

    def write_cache(self, model, positive, negative, video_latent, audio_latent, scene_id="0"):
        global LTX_CACHE
        
        # Store exact memory pointers natively (preserves all PromptRelay Temporal structures)
        LTX_CACHE[str(scene_id)] = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "video_latent": video_latent,
            "audio_latent": audio_latent
        }
        print(f"[LTX Cache] 💾 Saved Scene {scene_id}. Live Model and Conditioning Pointers Captured securely.")
        return ()

class MemoryCacheReader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {"scene_id": ("STRING", {"default": "0"})}}
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "video_latent", "audio_latent")
    FUNCTION = "read_cache"
    CATEGORY = "LTXBatch"

    def read_cache(self, scene_id="0"):
        global LTX_CACHE
        data = LTX_CACHE.get(str(scene_id))
        if data is None: raise ValueError(f"Cache for Scene {scene_id} not found in RAM!")
        print(f"[LTX Cache] 📂 Loaded Scene {scene_id}. Routing Live Pointers to Sampler.")
        return (data["model"], data["positive"], data["negative"], data["video_latent"], data["audio_latent"])

NODE_CLASS_MAPPINGS = {
    "MemoryCacheWriter": MemoryCacheWriter,
    "MemoryCacheReader": MemoryCacheReader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryCacheWriter": "Memory Cache Writer",
    "MemoryCacheReader": "Memory Cache Reader"
}

# ====================================================================
# BULLETPROOF HACKS (Fixed for CausalAudioAutoencoder)
# ====================================================================
try:
    import comfy.ldm.lightricks.vae.causal_audio_autoencoder
    
    _orig_encode = comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.encode
    def _patched_encode(self, x, **kwargs):
        target_dtype = next(self.parameters()).dtype
        return _orig_encode(self, x.to(target_dtype), **kwargs)
    comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.encode = _patched_encode

    _orig_decode = comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.decode
    def _patched_decode(self, z, **kwargs):
        target_dtype = next(self.parameters()).dtype
        return _orig_decode(self, z.to(target_dtype), **kwargs)
    comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.decode = _patched_decode
    
    print("[LTX Custom] ✅ Audio VAE CausalAutoencoder Dtype hack applied successfully.")
except Exception as e:
    print(f"[LTX Custom] ❌ Audio VAE Hack failed: {e}")

try:
    import comfy_extras.nodes_lt
    class SafeLTXVSeparateAVLatent(comfy_extras.nodes_lt.LTXVSeparateAVLatent):
        def execute(self, av_latent):
            s = av_latent["samples"]
            if getattr(s, "is_nested", False):
                tensors = s.unbind()
                vid_out = av_latent.copy()
                aud_out = av_latent.copy()
                vid_out["samples"] = tensors[0].contiguous()
                aud_out["samples"] = tensors[-1].contiguous() if len(tensors) > 1 else tensors[0].contiguous()
                return (vid_out, aud_out)
            return super().execute(av_latent)
    NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"] = SafeLTXVSeparateAVLatent
except Exception: pass
""")

        print("🔗 Running Atomic Model Folder Linker for ALL LTX 2.3 & Audio Dependencies...")
        base_models_dir = "/workspace/ComfyUI/models"
        
        dirs = [
            "unet", "vae", "clip", "text_encoders", "checkpoints", "loras", 
            "upscale_models", "latent_upscale_models", "cosyvoice", 
            "melbandroformer", "diffusion_models", "audio_separators", 
            "audio_vae", "audio_checkpoints"
        ]
        for d in dirs: os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        cv_source = "/mnt/weights/cosyvoice3"
        cv_dest = os.path.join(base_models_dir, "cosyvoice", "Fun-CosyVoice3-0.5B")
        if os.path.exists(cv_source) and not os.path.exists(cv_dest):
            os.symlink(cv_source, cv_dest)

        if os.path.exists("/mnt/weights/canonical_storage"):
            for root_dir, _, files in os.walk("/mnt/weights/canonical_storage"):
                if "cosyvoice3" in root_dir.split(os.sep): continue 
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
            
            if "error" in res:
                raise HTTPException(status_code=500, detail=f"Validation Error: {res['error']}")
            if "node_errors" in res and res["node_errors"]:
                raise HTTPException(status_code=500, detail=f"Node Errors: {res['node_errors']}")
                
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
    # PART 6: LYPSYNC FAST-BATCH ENDPOINT
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

            def inject_node_overrides(sg, idx, custom_w, custom_h, exact_audio_duration, total_frames, scene_data):
                # We iterate through the JSON nodes and inject dynamic values safely
                for node_id, node_data in list(sg.items()):
                    c_type = node_data.get("class_type")
                    if "inputs" not in node_data: continue
                    
                    if c_type == "DiffusionModelLoaderKJ":
                        node_data["inputs"]["model_name"] = "ltx-2.3-22b-distilled-fp8.safetensors"
                        node_data["inputs"]["patch_cublaslinear"] = False        
                        node_data["inputs"]["enable_fp16_accumulation"] = False 

                    elif c_type == "DualCLIPLoader":
                        node_data["inputs"]["clip_name1"] = "gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors"
                        node_data["inputs"]["clip_name2"] = "ltx-2.3_text_projection_bf16.safetensors"
                        node_data["inputs"]["type"] = "ltxv"
                    elif c_type == "MelBandRoFormerModelLoader":
                        node_data["inputs"]["model_name"] = "MelBandRoformer_fp32.safetensors"
                    elif c_type == "LTXVAudioVAELoader":
                        node_data["inputs"]["ckpt_name"] = "LTX23_audio_vae_bf16.safetensors"
                    elif c_type == "VAELoader":
                        node_data["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                    elif c_type == "FL_CosyVoice3_ModelLoader":
                        node_data["inputs"]["model_version"] = "Fun-CosyVoice3-0.5B"
                    elif c_type == "DenoLTXMultiLoraLoader":
                        # ENSURE distilled lora is loaded in slot 1 so the 12-step configuration doesn't break
                        node_data["inputs"]["lora_1"] = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
                        node_data["inputs"]["lora_2"] = "LTX2.3-IC-LORA-Dual-Character.safetensors"
                        node_data["inputs"]["lora_3"] = "VBVR-official-comfyui.safetensors"
                        node_data["inputs"]["lora_4"] = "LTX2.3rl-lora-zghhui-OmniNFT.safetensors"
                        for i in range(5, 9):
                            k = f"lora_{i}"
                            if k in node_data["inputs"] and node_data["inputs"][k] in ["", "None", None]:
                                node_data["inputs"][k] = "__none__"
                    elif c_type == "MemoryCacheWriter" or c_type == "MemoryCacheReader":
                        node_data["inputs"]["scene_id"] = str(idx)
                    elif c_type == "EmptyLTXVLatentVideo":
                        node_data["inputs"]["width"] = custom_w
                        node_data["inputs"]["height"] = custom_h
                        node_data["inputs"]["length"] = total_frames
                    elif c_type == "TrimAudioDuration":
                        node_data["inputs"]["duration"] = exact_audio_duration
                        node_data["inputs"]["start_index"] = 0
                    elif c_type == "PromptRelayEncode":
                        # ✅ FIX 2: If we pass naked text here, PromptRelay parsing completely fails, returning zero-conditioning.
                        # We must wrap user text in the syntax PromptRelay actually expects.
                        user_text = scene_data.get("dialog_text", "")
                        if "Shot 1" not in user_text:
                            # Automatically inject the strict parsing structure required by the node
                            formatted_prompt = f"[Scene] Cinematic visual.\n|\nShot 1 (Medium Shot, {exact_audio_duration}s): Character is speaking.\nCharacter: {user_text}"
                            node_data["inputs"]["local_prompts"] = formatted_prompt
                        else:
                            node_data["inputs"]["local_prompts"] = user_text
                    elif c_type == "CLIPTextEncode":
                        node_data["inputs"]["text"] = scene_data.get("negative_prompt", "blurry, out of focus, overexposed, underexposed, bad quality")
                    elif c_type == "LoadAudio":
                        if "18" in node_id: node_data["inputs"]["audio"] = f"dynamic_guides/spk1_{idx}.wav"
                        if "19" in node_id: node_data["inputs"]["audio"] = f"dynamic_guides/spk2_{idx}.wav"
                    elif c_type == "LoadImage":
                        node_data["inputs"]["image"] = f"dynamic_guides/char1_{idx}.png"
                    elif c_type == "SamplerCustom":
                        node_data["inputs"]["noise_seed"] = scene_data.get("seed", int(time.time() * 1000) % 1000000)
                    elif c_type == "VHS_VideoCombine":
                        node_data["inputs"]["save_output"] = True
                return sg

            try:
                async with aiohttp.ClientSession() as session:
                    custom_w = int(body.get("custom_width", 704)) 
                    custom_h = int(body.get("custom_height", 1248))

                    # 🌟 ROBUST DOWNLOAD ASSET FUNCTION (Extracts Boto3 Key even for Private Buckets)
                    async def download_asset(url, target_path):
                        if not url: return False
                        print(f"📥 Fetching asset from: {url}")
                        
                        if "r2.dev" in url or "cloudflarestorage" in url:
                            parsed = urlparse(url)
                            key = parsed.path.lstrip('/')
                            bucket_name = "video-asset-files-storage-workflow"
                            
                            if key.startswith(bucket_name + "/"):
                                key = key.replace(bucket_name + "/", "", 1)
                                
                            print(f"   [Auth] Using Boto3 for private bucket: {bucket_name} | Key: {key}")
                            try:
                                await asyncio.get_event_loop().run_in_executor(None, self.s3.download_file, bucket_name, key, target_path)
                                print(f"   ✅ Success (Boto3)")
                                return True
                            except Exception as e:
                                print(f"   ❌ Boto3 failed for {key}: {e}")
                                
                        print(f"   [HTTP] Attempting standard GET request...")
                        try:
                            async with session.get(url, timeout=60) as r:
                                if r.status == 200:
                                    with open(target_path, "wb") as f: f.write(await r.read())
                                    print(f"   ✅ Success (HTTP)")
                                    return True
                                else:
                                    print(f"   ❌ HTTP GET Failed. Status: {r.status}")
                        except Exception as e:
                            print(f"   ❌ HTTP GET Exception: {e}")
                        return False

                    print(f"\n[Lypsync API] 🎙️ STARTING PHASE 1: ENCODING {len(batch_scenes)} SCENES")
                    
                    for idx, scene in enumerate(batch_scenes):
                        spk1_path = os.path.join(dynamic_guides_dir, f"spk1_{idx}.wav")
                        spk2_path = os.path.join(dynamic_guides_dir, f"spk2_{idx}.wav")
                        await download_asset(scene.get("speaker1_audio_url"), spk1_path)
                        await download_asset(scene.get("speaker2_audio_url"), spk2_path)
                        
                        import soundfile as sf
                        import numpy as np 
                        for aud_p in [spk1_path, spk2_path]:
                            if not os.path.exists(aud_p):
                                sf.write(aud_p, np.zeros(16000, dtype=np.float32), 16000)

                        dialog_text = scene.get("dialog_text", f"{scene.get('speaker1_text', '')} {scene.get('speaker2_text', '')}")
                        clean_text = re.sub(r'SPEAKER\s+[a-zA-Z0-9]+:', '', dialog_text)
                        word_count = max(len(clean_text.split()), 1)
                        pauses = len(re.findall(r'[.,!?]', clean_text))
                        estimated_seconds = max(min((word_count / 2.0) + (pauses * 0.4) + 1.5, 20.0), 3.0)
                        
                        total_frames = (math.ceil(int(estimated_seconds * 25) / 8) * 8) + 1
                        exact_audio_duration = float(total_frames) / 25.0

                        sg1 = json.loads(json.dumps(subgraph_1))
                        sg1 = inject_node_overrides(sg1, idx, custom_w, custom_h, exact_audio_duration, total_frames, scene)

                        print(f"🎬 Processing Audio & Text Cache for Scene {idx}...")
                        await self.execute_comfy_workflow(session, sg1)

                        # 🚀 CRITICAL: DO NOT UNLOAD THE MODEL. Keep the UNet pointer alive in VRAM.
                        await self.clear_comfy_memory(session, unload_models=False)

                    print(f"\n[Lypsync API] 🎥 STARTING PHASE 2: RENDERING {len(batch_scenes)} VIDEOS")
                    out_dir = "/workspace/ComfyUI/output"
                    if os.path.exists(out_dir): shutil.rmtree(out_dir)
                    os.makedirs(out_dir)

                    for idx, scene in enumerate(batch_scenes):
                        img1_path = os.path.join(dynamic_guides_dir, f"char1_{idx}.png")
                        await download_asset(scene.get("image1_url"), img1_path)
                        from PIL import Image
                        if not os.path.exists(img1_path):
                            Image.new('RGB', (custom_w, custom_h), color='black').save(img1_path)
                        else:
                            Image.open(img1_path).convert("RGB").resize((custom_w, custom_h), Image.Resampling.LANCZOS).save(img1_path)
                        
                        scene["seed"] = scene.get("seed", int(time.time() * 1000) % 1000000)

                        sg2 = json.loads(json.dumps(subgraph_2))
                        sg2 = inject_node_overrides(sg2, idx, custom_w, custom_h, 0, 0, scene)

                        print(f"🎬 Rendering Video for Scene {idx} (Executing using Live Memory Pointer)...")
                        await self.execute_comfy_workflow(session, sg2)

                        # 🚀 Keep the model alive for the next batch iteration
                        await self.clear_comfy_memory(session, unload_models=False)

                        output_files = []
                        for root_p, _, filenames in os.walk(out_dir):
                            for name in filenames:
                                if name.endswith((".mp4", ".gif", ".webm")): output_files.append(os.path.join(root_p, name))

                        if not output_files: raise Exception(f"Inference for Scene {idx} finished but no output media files detected.")
                        
                        output_files.sort(key=os.path.getmtime)
                        target_video_file = output_files[-1]
                        saved_filename = os.path.basename(target_video_file)

                        target_key = f"{date_folder}/lypsync_clips/{int(time.time())}_{scene.get('name', 'clip')}_{saved_filename}"
                        print(f"📤 Syncing Finished Asset {idx} to R2...")
                        await asyncio.get_event_loop().run_in_executor(None, self.s3.upload_file, target_video_file, "video-asset-files-storage-workflow", target_key)
                        
                        generated_outputs.append({
                            "scene": scene.get("name", f"Clip_{idx+1}"),
                            "status": "success",
                            "file_key": target_key,
                            "public_url": f"https://pub-4d91f4d3d0366568a54ffa32ffcb7bf4.r2.dev/{target_key}",
                            "filename": saved_filename
                        })
                        os.remove(target_video_file)

                    print(f"\n[Lypsync API] 🎉 All Scenes Rendered. Pipeline Finished. Full Purge.")
                    await self.clear_comfy_memory(session, unload_models=True)
                    
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
