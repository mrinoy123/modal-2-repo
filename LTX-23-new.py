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
    "FORCE_REBUILD_INDEX": "222"  # Bumped to ensure a completely fresh image build layer
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
    "git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes"
)

deps_image = clone_image.run_commands(
    "sed -i '/torch/d' /workspace/ComfyUI/requirements.txt",
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec sed -i '/torch/d' {} \;",
    "python3.12 -m pip install --no-cache-dir -r /workspace/ComfyUI/requirements.txt",
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec python3.12 -m pip install --no-cache-dir -r {} \;"
)

final_image = deps_image.run_commands(
    "wget -qO /workspace/ComfyUI/ltxDirector_v10_api.json 'https://raw.githubusercontent.com/WhatDreamsCost/WhatDreamsCost-ComfyUI/main/workflows/LTX%20Director%20Example%20Workflow%20(Fixed).json' || true",
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
        
        # 🔥 CUSTOM NODES: LTXColorFixer & L40S Batch SubGraphIsolator 🔥
        print("🎨 Injecting Smart Auto-Exposure & Batch SubGraph Nodes...")
        custom_nodes_path = "/workspace/ComfyUI/custom_nodes/LTXCustomPipeline.py"
        with open(custom_nodes_path, "w") as f:
            f.write("""
import torch
import torchvision.transforms.functional as TF

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

class SubGraphIsolator:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",)
        }}
    
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "isolate_and_flush"
    CATEGORY = "utils"

    def isolate_and_flush(self, model, positive, negative):
        import gc
        import comfy.model_management
        print("\\n[SubGraph System] 🛑 Pass 1 (Text Encoding) Complete. Isolating Conditionings.")
        
        # L40S Optimization: We clear PyTorch activation cache but KEEP models loaded in VRAM for speed
        comfy.model_management.soft_empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        
        print("[SubGraph System] 🟢 Activation memory cleared. Handing Patched Data to KSampler.\\n")
        return (model, positive, negative)

NODE_CLASS_MAPPINGS = {
    "LTXColorFixer": LTXColorFixer,
    "SubGraphIsolator": SubGraphIsolator
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

        print("🚀 Launching Optimized LTX Server Engine on L40S GPU (48GB Native)...")
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

        print("✅ Base pipeline active. Awaiting API Batch triggers.")

    async def clear_comfy_memory(self, session, unload_models=False):
        # Optimized for Batching: Clears cache but KEEPS weights loaded between sequences
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

        print(f"⌛ Queued workflow. prompt_id: {prompt_id}. Polling state...")
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
    # PART 6: MAIN HIGH-SPEED INTERNAL BATCH ENDPOINT
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
            workflow_url = body.get("workflow_url") 

            # Normalize incoming payload to handle both flat fallback and advanced batch formats
            batch_scenes = body.get("batch_scenes", [])
            if not batch_scenes:
                print("⚠️ No batch_scenes array found. Falling back to legacy single-item format mapping.")
                batch_scenes = [{
                    "name": body.get("filename", "clip_output"),
                    "image_url": body.get("image_url", ""),
                    "kinetic_actions": [p for p in body.get("prompts", {}).values()] if isinstance(body.get("prompts"), dict) else body.get("prompts", []),
                    "style": body.get("style_profile", ""),
                    "subject": body.get("subject", ""),
                    "background": body.get("background", ""),
                    "lighting": body.get("lighting", ""),
                    "camera": body.get("camera", "")
                }]

            dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
            if os.path.exists(dynamic_guides_dir): shutil.rmtree(dynamic_guides_dir)
            os.makedirs(dynamic_guides_dir, exist_ok=True)

            ram_task = asyncio.create_task(self._ram_squeezer())
            generated_outputs = []

            try:
                async with aiohttp.ClientSession() as session:
                    # 1. Fetch Master Workflow Template Once
                    workflow_raw = body.get("workflow_json")
                    if workflow_raw:
                        base_workflow = json.loads(workflow_raw) if isinstance(workflow_raw, str) else workflow_raw
                    elif workflow_url:
                        async with session.get(workflow_url) as resp: base_workflow = await resp.json()
                    else:
                        try:
                            with open("/workspace/ComfyUI/ltxDirector_v10_api.json", "r") as f: base_workflow = json.load(f)
                        except FileNotFoundError:
                            with open("ltxDirector_v10(modified-own)api.json", "r") as f: base_workflow = json.load(f)

                    # DYNAMIC RESOLUTION EXTRACTION: Defaults to 9:16 aspect ratio
                    custom_w = 576
                    custom_h = 1024
                    overrides = body.get("workflow_override", {})
                    if "46" in overrides and "inputs" in overrides["46"]:
                        custom_w = overrides["46"]["inputs"].get("custom_width", 576)
                        custom_h = overrides["46"]["inputs"].get("custom_height", 1024)

                    # ==============================================================================
                    # 🚀 HIGH SPEED BATCH LOOP BEGINS HERE
                    # ==============================================================================
                    for idx, scene in enumerate(batch_scenes):
                        print(f"\\n🎬 Preparing Batch Loop Sequence [{idx+1}/{len(batch_scenes)}]: {scene.get('name', 'Clip')}")
                        
                        workflow = json.loads(json.dumps(base_workflow)) # Fresh copy per loop
                        workflow = self.merge_overrides(workflow, overrides)

                        # A. Ensure Out Dir is fresh for this specific video execution
                        out_dir = "/workspace/ComfyUI/output"
                        if os.path.exists(out_dir): shutil.rmtree(out_dir)
                        os.makedirs(out_dir)

                        # B. Single-Image Downloader & Initializer
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
                            
                        # Resize precisely to LTX Grid Constraints
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

                        # C. ⚙️ AUTO FRAME CALCULATOR & 6-PART PROMPT FUSION ⚙️
                        actions = scene.get("kinetic_actions", [])
                        if not actions: actions = ["The subject moves dynamically across the cinematic scene."]
                        
                        # Math Calculation Phase (130 WPM Pacing) -> Frames -> 8n+1 Grid Formatter
                        total_words = sum(len(str(a).split()) for a in actions)
                        seconds = max(total_words / (130 / 60.0), 2.5)  # Strict minimum 2.5 sec buffer
                        raw_frames = seconds * 24
                        total_frames = int(math.ceil((raw_frames - 1) / 8) * 8 + 1)
                        total_frames = max(33, min(total_frames, 257))  # Hardware sanity clamp (1.3s to 10.7s)
                        
                        num_actions = len(actions)
                        keyframe_steps = [int(i * (total_frames - 1) / max(1, num_actions - 1)) for i in range(num_actions)]

                        style = scene.get("style", "")
                        subject = scene.get("subject", "")
                        bg = scene.get("background", "")
                        light = scene.get("lighting", "")
                        cam = scene.get("camera", "")
                        
                        # Unify the static parameters
                        static_env = f"{subject} {style} {bg} {light}".strip()
                        
                        local_prompts_list = []
                        for step_frame, action_text in zip(keyframe_steps, actions):
                            # Master Grammar Alignment Equation
                            action_cam = f"{action_text} {cam}".strip()
                            fused_prompt = f"{action_cam}. Cinematic environment and styling: {static_env}"
                            local_prompts_list.append(f"{step_frame}: {fused_prompt}")
                            
                        local_prompts_str = "\n".join(local_prompts_list)
                        print(f"📊 Auto-Calculated Math: {total_words} Words -> {seconds:.1f} Sec -> Locked exactly to {total_frames} Frames.")

                        # D. Core Parameter Injection
                        if "46" in workflow:
                            if "inputs" not in workflow["46"]: workflow["46"]["inputs"] = {}
                            workflow["46"]["inputs"]["duration_frames"] = total_frames
                            workflow["46"]["inputs"]["local_prompts"] = local_prompts_str
                            workflow["46"]["inputs"]["timeline_data"] = timeline_data_str
                            workflow["46"]["inputs"]["frame_rate"] = 24

                        if "98" in workflow: 
                            workflow["98"]["inputs"]["unet_name"] = "LTX-2.3-22B-Distilled-FP4ME.safetensors"
                        if "100" in workflow: workflow["100"]["inputs"]["lora_name"] = "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"
                        if "101" in workflow:
                            workflow["101"]["inputs"]["clip_name1"] = "gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors"
                            workflow["101"]["inputs"]["clip_name2"] = "ltx-2.3_text_projection_bf16.safetensors"
                        if "97" in workflow: workflow["97"]["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                        if "103" in workflow: workflow["103"]["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                        if "102" in workflow: workflow["102"]["inputs"]["vae_name"] = "LTX23_audio_vae_bf16.safetensors"
                        if "94:105" in workflow: workflow["94:105"]["inputs"]["model_name"] = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
                        if "94:28" in workflow and "seed" in scene: workflow["94:28"]["inputs"]["noise_seed"] = scene["seed"]

                        # E. Dynamic SubGraph & Color Fixer Network Hooking
                        keys = list(workflow.keys())
                        for node_id in keys:
                            node_info = workflow[node_id]
                            
                            # Wire Color Fixer
                            if node_info.get("class_type") in ["VHS_VideoCombine", "SaveVideo"]:
                                if "inputs" in node_info and "images" in node_info["inputs"]:
                                    original_image_source = node_info["inputs"]["images"]
                                    fixer_id = f"9999_color_fixer_{idx}"
                                    workflow[fixer_id] = {
                                        "class_type": "LTXColorFixer",
                                        "inputs": {"image": original_image_source, "target_brightness": 0.40, "max_boost": 2.0}
                                    }
                                    node_info["inputs"]["images"] = [fixer_id, 0]

                            # Wire SubGraph Isolator inline before the primary Guider/Sampler kicks off UNet calculation
                            if node_info.get("class_type") in ["KSampler", "KSamplerAdvanced", "SamplerCustom", "CFGGuider", "BasicGuider"]:
                                if "model" in node_info.get("inputs", {}) and "positive" in node_info.get("inputs", {}):
                                    orig_model = node_info["inputs"]["model"]
                                    orig_pos = node_info["inputs"]["positive"]
                                    orig_neg = node_info["inputs"]["negative"]
                                    
                                    isolator_id = f"9998_subgraph_isolator_{node_id}_{idx}"
                                    workflow[isolator_id] = {
                                        "class_type": "SubGraphIsolator",
                                        "inputs": {
                                            "model": orig_model,
                                            "positive": orig_pos,
                                            "negative": orig_neg
                                        }
                                    }
                                    
                                    node_info["inputs"]["model"] = [isolator_id, 0]
                                    node_info["inputs"]["positive"] = [isolator_id, 1]
                                    node_info["inputs"]["negative"] = [isolator_id, 2]

                        print(f"🚀 Processing Sequence via L40S Server Matrix...")
                        await self.execute_comfy_workflow(session, workflow)

                        # F. File Capture and Upload Phase
                        output_files = []
                        for root_p, _, filenames in os.walk(out_dir):
                            for name in filenames:
                                if name.endswith((".mp4", ".gif", ".webm")):
                                    output_files.append(os.path.join(root_p, name))

                        if not output_files:
                            raise Exception("Inference finished but no output media files were detected.")
                        
                        output_files.sort(key=os.path.getmtime)
                        target_video_file = output_files[-1]
                        saved_filename = os.path.basename(target_video_file)

                        target_key = f"{date_folder}/generated clips/{int(time.time())}_{scene.get('name', 'clip')}_{saved_filename}"
                        print(f"📤 Uploading Segment to R2: {target_key}")
                        
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

                        # BATCH SPEED RULE: Unload ONLY the activation footprint, KEEP the models loaded inside L40S RAM for the next loop.
                        await self.clear_comfy_memory(session, unload_models=False)
                    
                    # Yield final output block
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
