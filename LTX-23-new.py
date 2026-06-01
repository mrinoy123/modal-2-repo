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
    "FORCE_REBUILD_INDEX": "212"  # Bumped to ensure a completely fresh image build layer
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
# PART 5: MODAL APP CONFIGURATION & CLOUD VOLUMES (8GB SYSTEM RAM LIMIT)
# ==============================================================================
app = modal.App("media-worker-ltx23")
weights_volume = modal.Volume.from_name("Ltx-23-model-weights-new", create_if_missing=False)

@app.cls(
    gpu="L40S", 
    image=final_image,
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("custom-secret")],
    memory=8192, 
    scaledown_window=30,
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
        
        # 🔥 SMART AUTO-EXPOSURE & COLOR FIXER NODE 🔥
        print("🎨 Injecting Smart Auto-Exposure LTX Color Fixer Node...")
        color_node_path = "/workspace/ComfyUI/custom_nodes/LTXColorFixer.py"
        with open(color_node_path, "w") as f:
            f.write("""
import torch
import torchvision.transforms.functional as TF

class LTXColorFixer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "target_brightness": ("FLOAT", {"default": 0.40, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Target average luminance (0.4 is good daylight)"}),
            "max_boost": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "Maximum multiplier so dark scenes don't blow out completely"}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"

    def process(self, image, target_brightness, max_boost):
        # Image shape from ComfyUI is (Batch/Frames, Height, Width, Channels RGB) [0.0 to 1.0]
        
        # 1. Calculate the real mean luminance (brightness) of the entire video batch
        # ITU-R BT.601 formula for relative luminance
        r = image[:, :, :, 0]
        g = image[:, :, :, 1]
        b = image[:, :, :, 2]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        
        mean_lum = torch.mean(luminance).item()
        
        # 2. Dynamic Logic Gate: Only trigger if the video is too dark
        if mean_lum < target_brightness and mean_lum > 0.01:
            # Calculate exactly how much to multiply the pixels to reach the target
            boost = target_brightness / mean_lum
            
            # Clamp it to the safe maximum so we don't destroy contrast
            boost = min(boost, max_boost)
            print(f"[LTX Smart Exposure] Video is dark (Luma: {mean_lum:.3f}). Applying Auto-Boost: {boost:.2f}x")
            
            # TF functions expect (B, C, H, W)
            img_t = image.permute(0, 3, 1, 2)
            
            # Apply dynamic brightness
            img_t = TF.adjust_brightness(img_t, boost)
            
            # Auto-compensate saturation (brighter pixels wash out, so we boost saturation slightly based on the brightness boost)
            sat_boost = 1.0 + ((boost - 1.0) * 0.4) 
            img_t = TF.adjust_saturation(img_t, sat_boost)
            
            # Convert back to ComfyUI standard shape (B, H, W, C)
            image = img_t.permute(0, 2, 3, 1)
        else:
            print(f"[LTX Smart Exposure] Video brightness is optimal (Luma: {mean_lum:.3f}). Bypassing adjustments.")
            
        return (image,)

NODE_CLASS_MAPPINGS = {"LTXColorFixer": LTXColorFixer}
""")
        # =====================================================================

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

        print("🚀 Launching Monolithic LTX 2.3 Server Engine...")
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

        print("🔥 Running Ghost Load to build mmap pointers and stream weights directly to GPU...")
        try:
            # FIX: Added `"device": "default"` into the VAELoaderKJ inputs so it successfully passes ComfyUI Validation!
            prewarm_payload = {
                "prompt": {
                    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors", "weight_dtype": "fp8_e4m3fn"}},
                    "2": {"class_type": "VAELoaderKJ", "inputs": {"vae_name": "LTX23_video_vae_bf16.safetensors", "weight_dtype": "bf16", "device": "default"}},
                    "3": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors", "clip_name2": "ltx-2.3_text_projection_bf16.safetensors", "type": "ltxv", "device": "default"}},
                    "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "prewarm system", "clip": ["3", 0]}},
                    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 128, "height": 128, "batch_size": 1}},
                    "6": {"class_type": "KSampler", "inputs": {
                        "seed": 0, "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                        "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 0], "latent_image": ["5", 0]
                    }},
                    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
                    "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ghost_load", "images": ["7", 0]}}
                }
            }
            req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps(prewarm_payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=300) 
            time.sleep(10)
            print("✅ Ghost Load complete. System is prepared for n8n API.")
        except Exception as e:
            print(f"⚠️ Ghost Load skipped: {e}")

    async def clear_comfy_memory(self, session):
        try:
            async with session.post("http://127.0.0.1:8188/free", json={"unload_models": True, "free_memory": True}) as r:
                await r.read()
        except Exception:
            pass
        
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        await asyncio.sleep(2)

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
    # PART 6: MAIN FASTAPI ENDPOINT & DYNAMIC TIMELINE / LATENT INJECTION
    # ==============================================================================
    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != "testing-modal-workflow-2": 
            raise HTTPException(status_code=403, detail="Unauthorized Account 2 Pipeline Request")
        
        body = await request.json()
        if isinstance(body, dict):
            if "json" in body: body = body["json"]
            elif "body" in body: body = body["body"]

        async def process_pipeline():
            incoming_image_urls = body.get("image_url")
            requested_length = int(body.get("length", 480)) 
            prompts_dict = body.get("prompts", {})
            date_folder = body.get("date_folder", time.strftime('%Y-%m-%d'))
            workflow_url = body.get("workflow_url") 

            if isinstance(prompts_dict, list):
                prompts_list = [str(p).strip() for p in prompts_dict if str(p).strip()]
            elif isinstance(prompts_dict, dict):
                try:
                    sorted_keys = sorted(prompts_dict.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
                    prompts_list = [str(prompts_dict[k]).strip() for k in sorted_keys if str(prompts_dict[k]).strip()]
                except Exception:
                    prompts_list = [str(v).strip() for v in prompts_dict.values()]
            else:
                prompts_list = [p.strip() for p in str(prompts_dict).split("\n") if p.strip()]

            urls_to_download = []
            if incoming_image_urls:
                if isinstance(incoming_image_urls, list): 
                    urls_to_download = [str(u).strip() for u in incoming_image_urls if str(u).strip()]
                elif isinstance(incoming_image_urls, str) and incoming_image_urls.strip():
                    urls_to_download = [u.strip() for u in incoming_image_urls.split(",") if u.strip()]

            dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
            if os.path.exists(dynamic_guides_dir): shutil.rmtree(dynamic_guides_dir)
            os.makedirs(dynamic_guides_dir, exist_ok=True)

            async def download_one(session, url_str, target_dest):
                try:
                    async with session.get(url_str, timeout=120) as r:
                        if r.status == 200:
                            with open(target_dest, "wb") as f: f.write(await r.read())
                except Exception: pass
                
                if not os.path.exists(target_dest):
                    from PIL import Image
                    img = Image.new('RGB', (1280, 704), color='black') 
                    img.save(target_dest)

            image_filenames = []
            if urls_to_download:
                async with aiohttp.ClientSession() as download_session:
                    tasks = [download_one(download_session, url, os.path.join(dynamic_guides_dir, f"guide_{i:04d}.png")) for i, url in enumerate(urls_to_download)]
                    await asyncio.gather(*tasks)
                image_filenames = [os.path.join(dynamic_guides_dir, f"guide_{i:04d}.png") for i in range(len(urls_to_download))]

            out_dir = "/workspace/ComfyUI/output"
            if os.path.exists(out_dir): shutil.rmtree(out_dir)
            os.makedirs(out_dir)

            ram_task = asyncio.create_task(self._ram_squeezer())

            try:
                async with aiohttp.ClientSession() as session:
                    workflow_raw = body.get("workflow_json")
                    if workflow_raw:
                        workflow = json.loads(workflow_raw) if isinstance(workflow_raw, str) else workflow_raw
                    elif workflow_url:
                        async with session.get(workflow_url) as resp:
                            workflow = await resp.json()
                    else:
                        try:
                            with open("/workspace/ComfyUI/ltxDirector_v10_api.json", "r") as f: 
                                workflow = json.load(f)
                        except FileNotFoundError:
                            with open("ltxDirector_v10(modified-own)api.json", "r") as f: 
                                workflow = json.load(f)
                    
                    workflow = self.merge_overrides(workflow, body.get("workflow_override"))

                    num_imgs = len(image_filenames)
                    num_prompts = len(prompts_list)

                    prompt_frames = [int(i * (requested_length - 1) / max(1, num_prompts - 1)) for i in range(num_prompts)]
                    local_prompts_str = "\n".join([f"{frame}: {prompt}" for frame, prompt in zip(prompt_frames, prompts_list)])

                    if num_imgs == 1:
                        segments = []
                        for frame in [0, 1]:
                            try:
                                with open(image_filenames[0], "rb") as f:
                                    b64_img = base64.b64encode(f.read()).decode("utf-8")
                                    segments.append({
                                        "frame": frame,
                                        "image": f"data:image/png;base64,{b64_img}"
                                    })
                            except Exception:
                                continue
                        timeline_data_str = json.dumps({"segments": segments, "audioSegments": []})
                            
                    elif num_imgs > 1:
                        img_frames = [int(i * (requested_length - 1) / max(1, num_imgs - 1)) for i in range(num_imgs)]
                        segments = []
                        for frame, img_path in zip(img_frames, image_filenames):
                            try:
                                with open(img_path, "rb") as f:
                                    b64_img = base64.b64encode(f.read()).decode("utf-8")
                                    segments.append({
                                        "frame": frame,
                                        "image": f"data:image/png;base64,{b64_img}"
                                    })
                            except Exception:
                                continue
                        timeline_data_str = json.dumps({"segments": segments, "audioSegments": []})

                    if "46" in workflow:
                        if "inputs" not in workflow["46"]: workflow["46"]["inputs"] = {}
                        workflow["46"]["inputs"]["duration_frames"] = requested_length
                        workflow["46"]["inputs"]["local_prompts"] = local_prompts_str
                        workflow["46"]["inputs"]["timeline_data"] = timeline_data_str
                        workflow["46"]["inputs"]["frame_rate"] = 24

                    if "98" in workflow: workflow["98"]["inputs"]["unet_name"] = "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"
                    if "100" in workflow: workflow["100"]["inputs"]["lora_name"] = "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"
                    if "101" in workflow:
                        workflow["101"]["inputs"]["clip_name1"] = "gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors"
                        workflow["101"]["inputs"]["clip_name2"] = "ltx-2.3_text_projection_bf16.safetensors"
                    if "97" in workflow: workflow["97"]["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                    if "103" in workflow: workflow["103"]["inputs"]["vae_name"] = "LTX23_video_vae_bf16.safetensors"
                    if "102" in workflow: workflow["102"]["inputs"]["vae_name"] = "LTX23_audio_vae_bf16.safetensors"
                    if "94:105" in workflow: workflow["94:105"]["inputs"]["model_name"] = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

                    # ==============================================================================
                    # FIX 2: 🎨 DYNAMICALLY WIRE THE BRIGHTNESS / COLOR FIXER INTO THE WORKFLOW
                    # ==============================================================================
                    keys = list(workflow.keys())
                    for node_id in keys:
                        node_info = workflow[node_id]
                        if node_info.get("class_type") in ["VHS_VideoCombine", "SaveVideo"]:
                            if "inputs" in node_info and "images" in node_info["inputs"]:
                                original_image_source = node_info["inputs"]["images"]
                                
                                fixer_id = "9999_color_fixer"
                                workflow[fixer_id] = {
                                    "class_type": "LTXColorFixer",
                                    "inputs": {
                                        "image": original_image_source,
                                        "target_brightness": 0.40,
                                        "max_boost": 2.0
                                    }
                                }
                                node_info["inputs"]["images"] = [fixer_id, 0]
                                print(f"🔗 Successfully wired LTXColorFixer before node {node_id}")
                    # ==============================================================================

                    print(f"🚀 Executing Monolithic LTX 2.3 Director Generation ({num_imgs} Images, {num_prompts} Prompts)...")
                    await self.execute_comfy_workflow(session, workflow)
                    await self.clear_comfy_memory(session)

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

                    target_key = f"{date_folder}/generated clips/{int(time.time())}_{saved_filename}"
                    print(f"📤 Uploading LTX 2.3 Output Video to R2: {target_key}")
                    
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.s3.upload_file, target_video_file, "video-asset-files-storage-workflow", target_key
                    )

                    public_path_url = f"https://pub-4d91f4d3d0366568a54ffa32ffcb7bf4.r2.dev/{target_key}" 
                    
                    return {
                        "status": "success",
                        "file_key": target_key,
                        "public_url": public_path_url,
                        "filename": saved_filename
                    }

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
