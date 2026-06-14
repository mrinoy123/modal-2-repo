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
    "FORCE_REBUILD_INDEX": "506"  # Updated for Director Workflow
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
    "git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
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
app = modal.App("media-worker-ltx23-director-lypsync")
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
class LTX23DirectorLypsyncEngine:

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

def move_to_cuda(item):
    if isinstance(item, torch.Tensor): 
        return item.to("cuda").contiguous()
    if isinstance(item, dict): 
        new_dict = type(item)()
        for k, v in item.items(): new_dict[k] = move_to_cuda(v)
        return new_dict
    if isinstance(item, list): 
        return type(item)(move_to_cuda(v) for v in item)
    if isinstance(item, tuple): 
        return type(item)(move_to_cuda(v) for v in item)
    return item

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
            "frame_rate": ("INT",)
        }, "optional": {
            "scene_id": ("STRING", {"default": "0"})
        }}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "write_cache"
    CATEGORY = "LTXBatch"

    def write_cache(self, model, positive, negative, video_latent, audio_latent, guide_data, frame_rate, scene_id="0"):
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
        print(f"[LTX Cache] 💾 Saved Scene {scene_id} pointers securely.")
        return ()

class MemoryCacheReader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {"scene_id": ("STRING", {"default": "0"})}}
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "LATENT", "GUIDE_DATA", "INT")
    RETURN_NAMES = ("model", "positive", "negative", "video_latent", "audio_latent", "guide_data", "frame_rate")
    FUNCTION = "read_cache"
    CATEGORY = "LTXBatch"

    def read_cache(self, scene_id="0"):
        global LTX_CACHE
        data = LTX_CACHE.get(str(scene_id))
        if data is None: raise ValueError(f"Cache for Scene {scene_id} not found in RAM!")
        print(f"[LTX Cache] 📂 Loaded Scene {scene_id}.")
        return (
            data["model"], 
            move_to_cuda(data["positive"]), 
            move_to_cuda(data["negative"]), 
            move_to_cuda(data["video_latent"]), 
            move_to_cuda(data["audio_latent"]),
            data["guide_data"],
            data["frame_rate"]
        )

NODE_CLASS_MAPPINGS = {
    "MemoryCacheWriter": MemoryCacheWriter,
    "MemoryCacheReader": MemoryCacheReader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryCacheWriter": "Memory Cache Writer",
    "MemoryCacheReader": "Memory Cache Reader"
}

# HACKS TO PREVENT CausalAudioAutoencoder DTYPE CRASHES 
try:
    import comfy.ldm.lightricks.vae.causal_audio_autoencoder
    _orig_encode = comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.encode
    def _patched_encode(self, x, **kwargs):
        return _orig_encode(self, x.to(next(self.parameters()).dtype), **kwargs)
    comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.encode = _patched_encode

    _orig_decode = comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.decode
    def _patched_decode(self, z, **kwargs):
        return _orig_decode(self, z.to(next(self.parameters()).dtype), **kwargs)
    comfy.ldm.lightricks.vae.causal_audio_autoencoder.CausalAudioAutoencoder.decode = _patched_decode
except Exception: pass
""")

        print("🔗 Running Atomic Model Folder Linker for ALL LTX 2.3 & Audio Dependencies...")
        base_models_dir = "/workspace/ComfyUI/models"
        
        # Ensure upscale_models and latent_upscale_models are properly mapped
        dirs = [
            "unet", "vae", "clip", "text_encoders", "checkpoints", "loras", 
            "upscale_models", "latent_upscale_models", "cosyvoice", 
            "melbandroformer", "diffusion_models", "audio_separators", 
            "audio_vae", "audio_checkpoints"
        ]
        for d in dirs: os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        # Mapping canonical storage to ComfyUI hierarchy
        if os.path.exists("/mnt/weights/canonical_storage"):
            for root_dir, _, files in os.walk("/mnt/weights/canonical_storage"):
                if "cosyvoice3" in root_dir.split(os.sep): continue 
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin", ".onnx", ".yaml", ".json")): continue
                    src_path = os.path.join(root_dir, filename)
                    
                    # Direct mapping logic for Upscalers / Checkpoints / Lora
                    if "spatial-upscaler" in filename.lower():
                        dest = os.path.join(base_models_dir, "latent_upscale_models", filename)
                    elif "lora" in filename.lower() or "talking_head" in filename.lower():
                        dest = os.path.join(base_models_dir, "loras", filename)
                    else:
                        dest = os.path.join(base_models_dir, "checkpoints", filename) # fallback
                        
                    for target_dir in dirs:
                        if target_dir == "cosyvoice": continue 
                        symlink_dest = os.path.join(base_models_dir, target_dir, filename)
                        if not os.path.exists(symlink_dest):
                            try: os.symlink(src_path, symlink_dest)
                            except FileExistsError: pass

        self.s3 = boto3.client(
            service_name='s3', 
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", 
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], 
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], 
            region_name="auto"
        )

        print("🚀 Launching LTX-Director Server Engine on L40S GPU...")
        os.makedirs("/tmp/comfy_swap", exist_ok=True)

        env_vars = os.environ.copy()
        env_vars["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
        env_vars["TORCH_NUM_THREADS"] = "1"
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
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
    # PART 6: LYPSYNC FAST-BATCH ENDPOINT (DYNAMIC 1 & 2 CHARACTER SUPPORT)
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

            # 🔥 NEW INJECTION LOGIC DYNAMICALLY ROUTES 1 OR 2 IMAGES/CHARACTERS
            def inject_node_overrides(sg, idx, custom_w, custom_h, exact_audio_duration, total_frames, scene_data):
                audio_node_counter = 0
                image_node_counter = 0

                for node_id, node_data in list(sg.items()):
                    c_type = node_data.get("class_type")
                    if "inputs" not in node_data: continue
                    
                    if c_type == "DiffusionModelLoaderKJ":
                        node_data["inputs"]["model_name"] = "ltx-2.3-22b-distilled-fp8.safetensors"
                    elif c_type == "DenoLTXMultiLoraLoader":
                        node_data["inputs"]["lora_1"] = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors" 
                        node_data["inputs"]["lora_2"] = "LTX_2.3_ID_LoRA_TalkVid_3K.safetensors" 
                        for i in range(3, 9):
                            k = f"lora_{i}"
                            if k in node_data["inputs"]: node_data["inputs"][k] = "__none__"
                    elif c_type == "LTXAVTextEncoderLoader":
                        node_data["inputs"]["text_encoder"] = "gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors"
                        node_data["inputs"]["ckpt_name"] = "ltx-2.3_text_projection_bf16.safetensors"
                    elif c_type == "MelBandRoFormerModelLoader":
                        node_data["inputs"]["model_name"] = "MelBandRoformer_fp32.safetensors"
                    elif c_type == "LTXVAudioVAELoader":
                        node_data["inputs"]["ckpt_name"] = "LTX23_audio_vae_bf16.safetensors"
                    elif c_type == "VAELoader":
                        node_data["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                    elif c_type == "LowVRAMLatentUpscaleModelLoader":
                        node_data["inputs"]["model_name"] = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
                    elif c_type == "FL_CosyVoice3_ModelLoader":
                        node_data["inputs"]["model_version"] = "Fun-CosyVoice3-0.5B"
                    elif c_type in ["MemoryCacheWriter", "MemoryCacheReader"]:
                        node_data["inputs"]["scene_id"] = str(idx)
                        
                    elif c_type == "LTXDirector":
                        node_data["inputs"]["custom_width"] = custom_w
                        node_data["inputs"]["custom_height"] = custom_h
                        if total_frames > 0:  
                            node_data["inputs"]["duration_frames"] = total_frames
                            node_data["inputs"]["duration_seconds"] = exact_audio_duration
                            
                            # 🔥 DYNAMIC 1 vs 2 CHARACTER PROMPT BUILDER
                            user_text = scene_data.get("dialog_text", "")
                            has_char_2 = bool(scene_data.get("image2_url"))
                            
                            if "Shot 1" not in user_text: 
                                if has_char_2:
                                    # Formats for 2 character conversation - splitting the time equally
                                    half_time = exact_audio_duration / 2.0
                                    spk1_txt = scene_data.get("speaker1_text", "I have told you everything I know.")
                                    spk2_txt = scene_data.get("speaker2_text", "You are holding back.")
                                    formatted_prompt = (
                                        f"[Characters]\nCharA: Person one.\nCharB: Person two.\n|\n"
                                        f"Shot 1 (Medium Shot, {half_time:.2f}s):\nCharA: {spk1_txt}\n|\n"
                                        f"Shot 2 (Medium Shot, {half_time:.2f}s):\nCharB: {spk2_txt}"
                                    )
                                else:
                                    # Formats for single character monologue
                                    formatted_prompt = f"[Scene] Cinematic talking portrait.\n|\nShot 1 (Medium Shot, {exact_audio_duration:.2f}s):\nCharacter: {user_text}"
                                node_data["inputs"]["global_prompt"] = formatted_prompt
                            else:
                                # Uses exactly what you pass if it's pre-formatted
                                node_data["inputs"]["global_prompt"] = user_text
                                
                    elif c_type == "FL_CosyVoice3_Dialog":
                        node_data["inputs"]["dialog_text"] = scene_data.get("dialog_text", "SPEAKER A: Hello.")
                        node_data["inputs"]["seed"] = scene_data.get("seed", int(time.time() * 1000) % 1000000)
                        
                    elif c_type == "LoadAudio":
                        audio_node_counter += 1
                        if audio_node_counter == 1: 
                            node_data["inputs"]["audio"] = f"dynamic_guides/spk1_{idx}.wav"
                        else: 
                            node_data["inputs"]["audio"] = f"dynamic_guides/spk2_{idx}.wav"
                            
                    elif c_type == "LoadImage":
                        # Sequentially pairs images: Char 1 -> First Node, Char 2 -> Second Node
                        image_node_counter += 1
                        if image_node_counter == 1: 
                            node_data["inputs"]["image"] = f"dynamic_guides/char1_{idx}.png"
                        else: 
                            node_data["inputs"]["image"] = f"dynamic_guides/char2_{idx}.png"
                            
                    elif c_type == "RandomNoise":
                        node_data["inputs"]["noise_seed"] = scene_data.get("seed", int(time.time() * 1000) % 1000000)
                    elif c_type == "VHS_VideoCombine":
                        node_data["inputs"]["save_output"] = True
                return sg

            try:
                async with aiohttp.ClientSession() as session:
                    custom_w = 448 
                    custom_h = 768

                    async def download_asset(url, target_path):
                        if not url: return False
                        if "r2.dev" in url or "cloudflarestorage" in url:
                            parsed = urlparse(url)
                            key = parsed.path.lstrip('/')
                            bucket_name = "video-asset-files-storage-workflow"
                            if key.startswith(bucket_name + "/"): key = key.replace(bucket_name + "/", "", 1)
                            try:
                                await asyncio.get_event_loop().run_in_executor(None, self.s3.download_file, bucket_name, key, target_path)
                                return True
                            except Exception: pass
                        try:
                            async with session.get(url, timeout=60) as r:
                                if r.status == 200:
                                    with open(target_path, "wb") as f: f.write(await r.read())
                                    return True
                        except Exception: pass
                        return False

                    print(f"\n[Lypsync API] 🎙️ STARTING PHASE 1: DIRECTING & ENCODING {len(batch_scenes)} SCENES")
                    
                    for idx, scene in enumerate(batch_scenes):
                        spk1_path = os.path.join(dynamic_guides_dir, f"spk1_{idx}.wav")
                        spk2_path = os.path.join(dynamic_guides_dir, f"spk2_{idx}.wav")
                        await download_asset(scene.get("speaker1_audio_url"), spk1_path)
                        await download_asset(scene.get("speaker2_audio_url"), spk2_path)
                        
                        import soundfile as sf
                        import numpy as np 
                        for aud_p in [spk1_path, spk2_path]:
                            if not os.path.exists(aud_p): sf.write(aud_p, np.zeros(16000, dtype=np.float32), 16000)

                        dialog_text = scene.get("dialog_text", f"{scene.get('speaker1_text', '')} {scene.get('speaker2_text', '')}")
                        clean_text = re.sub(r'SPEAKER\s+[a-zA-Z0-9]+:', '', dialog_text)
                        
                        word_count = max(len(clean_text.split()), 1)
                        pauses = len(re.findall(r'[.,!?]', clean_text))
                        estimated_seconds = max((word_count / 2.5) + (pauses * 0.4) + 1.0, 2.0)
                        
                        total_frames = (math.ceil(int(estimated_seconds * 25) / 8) * 8) + 1
                        if total_frames > 257: total_frames = 257 
                            
                        exact_audio_duration = float(total_frames - 1) / 25.0

                        sg1 = json.loads(json.dumps(subgraph_1))
                        sg1 = inject_node_overrides(sg1, idx, custom_w, custom_h, exact_audio_duration, total_frames, scene)

                        print(f"🎬 Processing Audio & Text Cache for Scene {idx} (Frames: {total_frames}, Seconds: {exact_audio_duration:.2f})...")
                        await self.execute_comfy_workflow(session, sg1)

                        await self.clear_comfy_memory(session, unload_models=False)

                    print("\n🧹 Phase 1 Batch Complete. Clearing Address Spaces...")
                    await self.clear_comfy_memory(session, unload_models=True)

                    print(f"\n[Lypsync API] 🎥 STARTING PHASE 2: BASE SAMPLING & UPSCALING {len(batch_scenes)} VIDEOS")
                    out_dir = "/workspace/ComfyUI/output"
                    if os.path.exists(out_dir): shutil.rmtree(out_dir)
                    os.makedirs(out_dir)

                    for idx, scene in enumerate(batch_scenes):
                        # 🔥 DOWNLOAD BOTH IMAGES DYNAMICALLY
                        img1_path = os.path.join(dynamic_guides_dir, f"char1_{idx}.png")
                        img2_path = os.path.join(dynamic_guides_dir, f"char2_{idx}.png")
                        
                        await download_asset(scene.get("image1_url"), img1_path)
                        await download_asset(scene.get("image2_url"), img2_path) # Fails silently if not provided
                        
                        from PIL import Image
                        # Standardize Image 1 (Character A)
                        if not os.path.exists(img1_path):
                            Image.new('RGB', (custom_w, custom_h), color='black').save(img1_path)
                        else:
                            Image.open(img1_path).convert("RGB").resize((custom_w, custom_h), Image.Resampling.LANCZOS).save(img1_path)
                            
                        # Standardize Image 2 (Character B) -> Creates a blank fail-safe image if left empty by n8n
                        if os.path.exists(img2_path):
                            Image.open(img2_path).convert("RGB").resize((custom_w, custom_h), Image.Resampling.LANCZOS).save(img2_path)
                        else:
                            Image.new('RGB', (custom_w, custom_h), color='black').save(img2_path)
                        
                        scene["seed"] = scene.get("seed", int(time.time() * 1000) % 1000000)

                        sg2 = json.loads(json.dumps(subgraph_2))
                        sg2 = inject_node_overrides(sg2, idx, custom_w, custom_h, 0, 0, scene)

                        print(f"🎬 Rendering Video for Scene {idx} (Executing using Live Memory Pointer)...")
                        await self.execute_comfy_workflow(session, sg2)
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
