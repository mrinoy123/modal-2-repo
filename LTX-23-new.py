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
    "FORCE_REBUILD_INDEX": "310"  # Bumped to force image rebuild
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
    "python3.12 -m pip install --no-cache-dir diffusers accelerate transformers torchsde numpy==1.26.4 kornia==0.7.3",
    "python3.12 -m pip install --no-cache-dir sageattention==1.0.6"
)

clone_image = torch_image.run_commands(
    "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
    "GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    "git clone --depth 1 https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git /workspace/ComfyUI/custom_nodes/WhatDreamsCost-ComfyUI",
    "git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    # Fetching multi lora loader deno node for proper LTX 2.3 loading
    "git clone --depth 1 https://github.com/kijai/ComfyUI-LTXVideo-Wrappers.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo-Wrappers"
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
        
        # 🔥 CUSTOM NODES: LTXColorFixer, Two-Pass Cache Writers & VAE Armor Patch 🔥
        # UPDATED: Added negative conditioning to the Cache Reader/Writer
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

# High-Speed Python Dictionary serving as VRAM Cache Buffer
LTX_CACHE = {}

class MemoryCacheWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",), # NEW: Capture Negative Prompts
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
            "negative": negative, # Save negative data
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "guide_data": guide_data,
            "frame_rate": frame_rate
        }
        print(f"\\n[Two-Pass System] 💾 Encoded & Saved Conditionings (Pos+Neg) for Scene {scene_id} into RAM\\n")
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
        print(f"\\n[Two-Pass System] 🚀 Bypassing Loaders: Loaded Pre-Cached Conditionings (Pos+Neg) for Scene {scene_id}\\n")
        return (data["model"], data["positive"], data["negative"], data["video_latent"], data["audio_latent"], data["guide_data"], data["frame_rate"])

# VAE Memory Armor Patch: Prevents massive allocations from dumping the UNet
class FastVAEDecode(nodes.VAEDecode):
    def decode(self, vae, samples):
        print("\\n[Two-Pass System] 🛡️ Auto-Routing to Tiled VAE Decoding to protect 22B UNet VRAM state.\\n")
        try:
            return (vae.decode_tiled(samples["samples"], tile_x=512, tile_y=512), )
        except Exception as e:
            print(f"\\n[Two-Pass System] Tiled decode warning: {e}. Falling back to standard.\\n")
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
        for d in dirs: 
            os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin")): 
                        continue
                    src_path = os.path.join(root_dir, filename)
                    for target_dir in dirs:
                        dest = os.path.join(base_models_dir, target_dir, filename)
                        if not os.path.exists(dest):
                            try: 
                                os.symlink(src_path, dest)
                            except FileExistsError: 
                                pass

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
            "--bf16-vae", "--use-sage-attention"
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        start_time = time.time()
        comfy_ready = False
        while time.time() - start_time < 300:
            if self.process.poll() is not None: 
                os._exit(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1) as response:
                    if response.status == 200: 
                        comfy_ready = True
                        break
            except Exception: 
                time.sleep(2)
                
        if not comfy_ready:
            os._exit(1)

        print("✅ Base pipeline active. Awaiting Two-Pass API Batch triggers.")

    async def clear_comfy_memory(self, session, unload_models=False):
        try:
            async with session.post("http://127.0.0.1:8188/free", json={"unload_models": unload_models, "free_memory": True}) as r:
                await r.read()
        except Exception: pass
        
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
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

    def merge_overrides(self, base_graph, override_graph):
        if not override_graph: return base_graph
        if isinstance(override_graph, str):
            try: override_graph = json.loads(override_graph)
            except Exception: return base_graph
        for node_id, node_data in override_graph.items():
            if node_id in base_graph:
                if "inputs" in node_data and "inputs" in base_graph[node_id]:
                    base_graph[node_id]["inputs"].update(node_data["inputs"])
                else: base_graph[node_id].update(node_data)
            else: base_graph[node_id] = node_data
        return base_graph

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
            
            if not batch_scenes:
                raise HTTPException(status_code=400, detail="Missing batch_scenes array.")

            dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
            if os.path.exists(dynamic_guides_dir): shutil.rmtree(dynamic_guides_dir)
            os.makedirs(dynamic_guides_dir, exist_ok=True)

            ram_task = asyncio.create_task(self._ram_squeezer())
            generated_outputs = []

            try:
                async with aiohttp.ClientSession() as session:
                    # 1. Base Workflow Extraction
                    base_workflow = body.get("workflow_json")
                    if isinstance(base_workflow, str):
                        base_workflow = json.loads(base_workflow)

                    custom_w = 576
                    custom_h = 1024
                    overrides = body.get("workflow_override", {})
                    if "46" in overrides and "inputs" in overrides["46"]:
                        custom_w = overrides["46"]["inputs"].get("custom_width", 576)
                        custom_h = overrides["46"]["inputs"].get("custom_height", 1024)

                    # Dynamic Target Injector
                    def inject_model_paths(workflow):
                        if "98" in workflow: workflow["98"]["inputs"]["unet_name"] = "ltx-2.3-22b-distilled-fp8.safetensors"
                        if "97" in workflow: workflow["97"]["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                        if "103" in workflow: workflow["103"]["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                        if "102" in workflow: workflow["102"]["inputs"]["vae_name"] = "LTX23_audio_vae_bf16.safetensors"
                        if "94:105" in workflow: workflow["94:105"]["inputs"]["model_name"] = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
                        if "101" in workflow: 
                            workflow["101"]["inputs"]["clip_name1"] = "gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors"
                            workflow["101"]["inputs"]["clip_name2"] = "ltx-2.3_text_projection_bf16.safetensors"
                        return workflow

                    # ==============================================================================
                    # PRE-COMPUTE: Download Images & Calculate Dimensions
                    # ==============================================================================
                    for idx, scene in enumerate(batch_scenes):
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

                        style = scene.get("style", "")
                        subject = scene.get("subject", "")
                        bg = scene.get("background", "")
                        light = scene.get("lighting", "")
                        cam = scene.get("camera", "")
                        
                        # Generate the negative prompt dynamically
                        negative_prompt = scene.get("negative_prompt", "")
                        scene["_negative_prompt"] = negative_prompt
                        
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
                        scene["_seed"] = scene.get("seed", int(time.time() * 1000) % 1000000)

                    # ==============================================================================
                    # PASS 1: THE TEXT ENCODING MARATHON & LORA MATRIX INJECTION
                    # ==============================================================================
                    print("\n[Two-Pass System] 🎬 PASS 1 START: Initiating Text Encoding & Multi-LoRA Structure...")
                    pass1_workflow = json.loads(json.dumps(base_workflow))
                    pass1_workflow = self.merge_overrides(pass1_workflow, overrides)
                    pass1_workflow = inject_model_paths(pass1_workflow)

                    # Strip heavy execution layers to make Pass 1 exclusively a text/motion compiler
                    keys_to_delete = []
                    for node_id, node_data in pass1_workflow.items():
                        c_type = node_data.get("class_type", "")
                        if c_type in ["KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced", 
                                      "VHS_VideoCombine", "SaveVideo", "VAEDecode", "LatentUpscale", 
                                      "LTXVCropGuides", "CFGGuider", "BasicScheduler", "BasicGuider", "LatentInterpolate"]:
                            keys_to_delete.append(node_id)
                    for k in keys_to_delete: del pass1_workflow[k]

                    orig_46 = pass1_workflow.pop("46", None)
                    if not orig_46: raise Exception("LTXDirector node '46' not found in base workflow!")
                    
                    orig_multi_lora_id = None
                    for nid, ndata in pass1_workflow.items():
                        if ndata.get("class_type") == "LTXMultiLoRALoader":
                            orig_multi_lora_id = nid
                            break
                            
                    if orig_multi_lora_id:
                        del pass1_workflow[orig_multi_lora_id]

                    # Map Director loops into RAM Caches with the New Multi-LoRA Logic
                    for idx, scene in enumerate(batch_scenes):
                        scene_46 = json.loads(json.dumps(orig_46))
                        scene_46["inputs"]["duration_frames"] = scene["_total_frames"]
                        scene_46["inputs"]["local_prompts"] = scene["_local_prompts_str"]
                        scene_46["inputs"]["timeline_data"] = scene["_timeline_data_str"]
                        scene_46["inputs"]["frame_rate"] = 24
                        
                        # NEW: Inject dynamic negative prompt handling into Director
                        # Instead of static string, bind it if available
                        if "94:8" in pass1_workflow: # Update negative clip encoder with prompt
                            neg_encoder_id = f"neg_encoder_{idx}"
                            pass1_workflow[neg_encoder_id] = json.loads(json.dumps(pass1_workflow["94:8"]))
                            pass1_workflow[neg_encoder_id]["inputs"]["text"] = scene.get("_negative_prompt", "")
                            # We will route this new negative encode directly to the Writer
                            
                      
# DYNAMIC MULTI-LORA (DENO) STACK BUILDING LOGIC 
# ----------------------------------------------------------------------
current_model_link = orig_46["inputs"].get("model", ["98", 0])

# Set default values for the 10 slots with correct calibrated weights
lora_inputs = {
    "model": current_model_link,
    # Fixed Style LoRAs (Permanent on 8, 9, 10)
    "lora_8_name": "VBVR-official-comfyui.safetensors",
    "lora_8_strength": 1.0,  # Restored structural baseline logic
    "lora_9_name": "LTX_2.3_Soft_Enhance_Style_LoRa.safetensors",
    "lora_9_strength": 0.4,  # Keeps background cloud contrast clean
    "lora_10_name": "LTX_2.3_Crisp_Enhance_Style_LoRa.safetensors",
    "lora_10_strength": 0.4, # Sharpens mechanical lines and surfaces
    
    # Camera Control Dictionary mapping
    "lora_1_name": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "lora_2_name": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "lora_3_name": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "lora_4_name": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "lora_5_name": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "lora_6_name": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "lora_7_name": "ltx-2-19b-lora-camera-control-static.safetensors"
}

# Initialize camera strengths to 0.0
for i in range(1, 8):
    lora_inputs[f"lora_{i}_strength"] = 0.0
    
# Toggle camera control dynamically based on parsed scene request
cam_string = scene.get("camera", "").lower()
active_slots = 0

if "in" in cam_string and "dolly" in cam_string: lora_inputs["lora_1_strength"] = 1.0; active_slots+=1
if "left" in cam_string and "dolly" in cam_string: lora_inputs["lora_2_strength"] = 1.0; active_slots+=1
if "out" in cam_string and "dolly" in cam_string: lora_inputs["lora_3_strength"] = 1.0; active_slots+=1
if "right" in cam_string and "dolly" in cam_string: lora_inputs["lora_4_strength"] = 1.0; active_slots+=1
if "down" in cam_string and "jib" in cam_string: lora_inputs["lora_5_strength"] = 1.0; active_slots+=1
if "up" in cam_string and "jib" in cam_string: lora_inputs["lora_6_strength"] = 1.0; active_slots+=1
if "static" in cam_string or active_slots == 0: lora_inputs["lora_7_strength"] = 1.0

# Inject the configured Multi-LoRA Deno node cleanly into the pipeline graph
multi_lora_id = f"9000_multi_lora_{idx}"
pass1_workflow[multi_lora_id] = {
    "class_type": "MultiLoraLoaderDeno",
    "inputs": lora_inputs
}
                        
                        # Initialize camera strengths to 0.0
                        for i in range(1, 8):
                            lora_inputs[f"lora_{i}_strength"] = 0.0
                            
                        # Toggle camera control dynamically based on parsed scene request
                        cam_string = scene.get("camera", "").lower()
                        active_slots = 0
                        
                        if "in" in cam_string and "dolly" in cam_string: lora_inputs["lora_1_strength"] = 1.0; active_slots+=1
                        if "left" in cam_string and "dolly" in cam_string: lora_inputs["lora_2_strength"] = 1.0; active_slots+=1
                        if "out" in cam_string and "dolly" in cam_string: lora_inputs["lora_3_strength"] = 1.0; active_slots+=1
                        if "right" in cam_string and "dolly" in cam_string: lora_inputs["lora_4_strength"] = 1.0; active_slots+=1
                        if "down" in cam_string and "jib" in cam_string: lora_inputs["lora_5_strength"] = 1.0; active_slots+=1
                        if "up" in cam_string and "jib" in cam_string: lora_inputs["lora_6_strength"] = 1.0; active_slots+=1
                        if "static" in cam_string or active_slots == 0: lora_inputs["lora_7_strength"] = 1.0
                        
                        # Inject the configured Multi-LoRA node
                        multi_lora_id = f"9000_multi_lora_{idx}"
                        pass1_workflow[multi_lora_id] = {
                            "class_type": "LTXMultiLoRALoader",
                            "inputs": lora_inputs
                        }
                        
                        # Snap the fully stacked chain back into the Director node
                        scene_46["inputs"]["model"] = [multi_lora_id, 0]
                        # ----------------------------------------------------------------------
                        
                        pass1_workflow[f"46_{idx}"] = scene_46
                        
                        pass1_workflow[f"writer_{idx}"] = {
                            "class_type": "MemoryCacheWriter",
                            "inputs": {
                                "model": [f"46_{idx}", 0],
                                "positive": [f"46_{idx}", 1],
                                "negative": [f"neg_encoder_{idx}" if "94:8" in pass1_workflow else "46_{idx}", 1], # Pull from dynamic encoder if available
                                "video_latent": [f"46_{idx}", 2],
                                "audio_latent": [f"46_{idx}", 3], # Audio conditioning is routed properly to phase 2
                                "guide_data": [f"46_{idx}", 4],
                                "frame_rate": [f"46_{idx}", 5],
                                "scene_id": str(idx)
                            }
                        }

                    print(f"🚀 Queuing Math Encoding Pass for {len(batch_scenes)} Scenes simultaneously...")
                    await self.execute_comfy_workflow(session, pass1_workflow)
                    await self.clear_comfy_memory(session, unload_models=False)

                    # ==============================================================================
                    # PASS 2: THE SAMPLING BLAST
                    # ==============================================================================
                    print("\n[Two-Pass System] 🚀 PASS 2 START: Initiating Pure Sampling Blast...")
                    
                    for idx, scene in enumerate(batch_scenes):
                        print(f"\n🎬 Rendering Native Scene [{idx+1}/{len(batch_scenes)}]: {scene.get('name', 'Clip')}")
                        
                        pass2_workflow = json.loads(json.dumps(base_workflow))
                        pass2_workflow = self.merge_overrides(pass2_workflow, overrides)
                        pass2_workflow = inject_model_paths(pass2_workflow)

                        out_dir = "/workspace/ComfyUI/output"
                        if os.path.exists(out_dir): shutil.rmtree(out_dir)
                        os.makedirs(out_dir)

                        if "101" in pass2_workflow: del pass2_workflow["101"]
                        if "46" in pass2_workflow: del pass2_workflow["46"]
                        
                        # Remove text encoders in pass 2 since they are cached
                        if "94:8" in pass2_workflow: del pass2_workflow["94:8"] 
                        if orig_multi_lora_id and orig_multi_lora_id in pass2_workflow: del pass2_workflow[orig_multi_lora_id]

                        pass2_workflow[f"reader_{idx}"] = {
                            "class_type": "MemoryCacheReader",
                            "inputs": {"scene_id": str(idx)}
                        }

                        # Dynamically reroute execution nodes directly to RAM output ports
                        for node_id, node_data in pass2_workflow.items():
                            if "inputs" in node_data:
                                for input_name, input_val in node_data["inputs"].items():
                                    if isinstance(input_val, list):
                                        if len(input_val) == 2 and input_val[0] == "46":
                                            # Route video/audio latents and positive conditioning
                                            pass2_workflow[node_id]["inputs"][input_name] = [f"reader_{idx}", input_val[1]]
                                        elif len(input_val) == 2 and input_val[0] == "94:8" and input_name == "negative":
                                            # Route the negative conditioning from the cache reader pin 2
                                            pass2_workflow[node_id]["inputs"][input_name] = [f"reader_{idx}", 2]

                        if "94:28" in pass2_workflow: pass2_workflow["94:28"]["inputs"]["noise_seed"] = scene["_seed"]
                        
                        keys = list(pass2_workflow.keys())
                        for node_id in keys:
                            node_info = pass2_workflow[node_id]
                            c_type = node_info.get("class_type", "")
                            
                            # Update VHS Video Combine for proper output formatting and FPS
                            if c_type == "VHS_VideoCombine":
                                if "inputs" in node_info:
                                    node_info["inputs"]["frame_rate"] = 24
                                    node_info["inputs"]["format"] = "video/mp4"
                                    if "pingpong" in node_info["inputs"]: node_info["inputs"]["pingpong"] = False

                            # Color Fixer Injection
                            if c_type == "VHS_VideoCombine":
                                if "inputs" in node_info and "images" in node_info["inputs"]:
                                    original_image_source = node_info["inputs"]["images"]
                                    fixer_id = f"9999_color_fixer_{idx}"
                                    pass2_workflow[fixer_id] = {
                                        "class_type": "LTXColorFixer",
                                        "inputs": {"image": original_image_source, "target_brightness": 0.40, "max_boost": 2.0}
                                    }
                                    node_info["inputs"]["images"] = [fixer_id, 0]

                        await self.execute_comfy_workflow(session, pass2_workflow)

                        # Capture & Upload Final Processed Video
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

                        # BATCH SPEED RULE: Clean out active rendering cache but keep heavy UNET loaded.
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
