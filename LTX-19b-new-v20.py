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
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# Setup optimized development container image with required compilation tooling
base_image = modal.Image.from_registry(
    "nvidia/cuda:12.5.1-devel-ubuntu24.04", 
    add_python="3.12"
).apt_install(
    "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0", 
    "build-essential", "ninja-build", "cmake", "clang", "llvm"
)

# Lock dependencies globally and map environment
build_image = base_image.env({
    "CUDA_HOME": "/usr/local/cuda",
    "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
    "FORCE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": "8.9", 
    "MAX_JOBS": "1",
    "CC": "gcc",
    "CXX": "g++"
}).pip_install(
    "fastapi", "aiohttp", "boto3", "triton>=3.1.0", 
    "ninja", "setuptools>=70.0.0", "wheel", "pip>=24.0"
).pip_install(
    "pandas", "numexpr", "pytz", "python-dateutil", 
    "scipy", "matplotlib", "colorama", "librosa", "soundfile", 
    "decord", "imageio", "scikit-image", "numba", "einops", 
    "transformers", "diffusers", "accelerate", "bitsandbytes"
)

# =========================================================================================
# 🔥 FIX 1: Install Exact Torch Stack FIRST (Prevents deployment freezing from cu13 bloat)
# =========================================================================================
torch_image = build_image.run_commands(
    "pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124",
    "pip install --no-cache-dir numpy==1.26.4 kornia==0.7.3 sageattention==1.0.6 diffusers accelerate transformers"
)

# Clone ComfyUI and install required custom nodes
clone_image = torch_image.run_commands(
    "git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
    "cd /workspace/ComfyUI && git checkout $(git rev-list -n 1 --before=\"2026-03-01\" HEAD)",
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    "cd /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo && git checkout $(git rev-list -n 1 --before=\"2026-03-01\" HEAD)",
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
    "git clone https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes",
    "git clone https://github.com/cubiq/ComfyUI_essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_essentials",
    "git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes",
    "git clone https://github.com/SquirrelRat/MultiString-Prompts.git /workspace/ComfyUI/custom_nodes/MultiString-Prompts",
    "git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts",
    "git clone https://github.com/IvanRybakov/comfyui-node-int-to-string-convertor.git /workspace/ComfyUI/custom_nodes/comfyui-node-int-to-string-convertor"
)

# Isolate requirements processing
deps_image = clone_image.run_commands(
    "sed -i '/torch/d' /workspace/ComfyUI/requirements.txt",
    "pip install --no-cache-dir -r /workspace/ComfyUI/requirements.txt",
    "sed -i '/torch/d' /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt",
    "pip install --no-cache-dir -r /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt",
    "sed -i '/torch/d' /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt",
    "pip install --no-cache-dir -r /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt"
)

final_image = deps_image.run_commands(
    # Fix FizzNodes NoneType crash
    "sed -i 's/final_pooled_output = torch.cat(pooled_out, dim=0)/final_pooled_output = torch.cat([p for p in pooled_out if p is not None], dim=0) if any(p is not None for p in pooled_out) else None/g' /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes/BatchFuncs.py",
    
    # =========================================================================================
    # 🔥 FIX 2: Resolves "list index out of range" crash inside the LTXVideo Looping Sampler
    # By forcibly intercepting ComfyUI's standard set_conds to retain unparsed raw_conds
    # =========================================================================================
    "python3 -c \"filepath = '/workspace/ComfyUI/comfy/samplers.py'; code = open(filepath).read(); code = code.replace('def set_conds(self, positive, negative):', 'def set_conds(self, positive, negative):\\n        self.raw_conds = (positive, negative)'); open(filepath, 'w').write(code)\"",
    
    # Global SageAttention API remap
    "echo '' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py",
    "echo 'sageattn_qk_int8_pv_fp16_triton = sageattn' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py"
)

app = modal.App("ltx-2-19b-v20-api")
weights_volume = modal.Volume.from_name("ltx-new-version20-weights")

@app.cls(
    gpu="L4", 
    image=final_image, 
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("video-generator-workflow")], 
    memory=8192, 
    scaledown_window=60,
    timeout=3600 
)
class LTXEngine:
    def _log_reader(self):
        for line in iter(self.process.stdout.readline, ""):
            if line: print(f"[ComfyUI] {line.strip()}")

    async def _ram_squeezer(self):
        print("🛡️ RAM Watchdog Active. Forcing Linux to drop page cache...")
        while True:
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1\n')
            except Exception:
                try: ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception: pass
            await asyncio.sleep(2)

    # ==============================================================================
    # ORCHESTRATION METHODS (Memory Clear & Subgraph Execution)
    # ==============================================================================
    async def clear_comfy_memory(self, session):
        try:
            async with session.post("http://127.0.0.1:8188/free", json={"unload_models": True, "free_memory": True}) as r:
                await r.read()
        except Exception as e:
            print(f"[Warning] Memory unload endpoint failed: {e}")
        
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    async def execute_comfy_workflow(self, session, workflow_json):
        async with session.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow_json}) as r:
            res = await r.json()
            if "error" in res: raise HTTPException(status_code=400, detail=f"Invalid JSON: {res['error']}")
            prompt_id = res["prompt_id"]

        print(f"⌛ Queued workflow step. prompt_id: {prompt_id}. Polling state...")
        start_time = time.time()
        while True:
            if self.process.poll() is not None:
                raise HTTPException(status_code=500, detail="Backend server execution failure.")
            async with session.get(f"http://127.0.0.1:8188/history/{prompt_id}") as hist_resp:
                if hist_resp.status == 200:
                    history = await hist_resp.json()
                    if prompt_id in history:
                        step_data = history[prompt_id]
                        if "status" in step_data and "messages" in step_data["status"]:
                            for msg in step_data["status"]["messages"]:
                                if msg[0] == "execution_error":
                                    raise HTTPException(status_code=500, detail=f"ComfyUI Error: {msg[1]}")
                        return step_data
            if time.time() - start_time > 2400:
                raise HTTPException(status_code=540, detail="Execution timeout reached.")
            await asyncio.sleep(2)

    def merge_overrides(self, base_graph, override_graph):
        if not override_graph:
            return base_graph
        if isinstance(override_graph, str):
            try: override_graph = json.loads(override_graph)
            except Exception: return base_graph
        
        for node_id, node_data in override_graph.items():
            if node_id in base_graph:
                if "inputs" in node_data and "inputs" in base_graph[node_id]:
                    base_graph[node_id]["inputs"].update(node_data["inputs"])
                else:
                    base_graph[node_id].update(node_data)
            else:
                base_graph[node_id] = node_data
        return base_graph

    @modal.enter()
    def start_comfy(self):
        import boto3
        print("🔗 Running Atomic Model Folder Linker...")
        base_models_dir = "/workspace/ComfyUI/models"
        
        dirs = ["unet", "vae", "clip", "text_encoders", "text_encoder", "checkpoints", "diffusion_models", "gguf", "loras"]
        for d in dirs:
            os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        # Explicit target-specific file mapper
        exact_mapping = {
            "gemma-3-12b-it-FP8.safetensors": ["text_encoders", "text_encoder"],
            "ltx-2-19b-embeddings_connector_dev_bf16.safetensors": ["checkpoints"],
            "ltx-2-19b-distilled-fp8.safetensors": ["unet", "diffusion_models"],
            "ltx-2-19b-ic-lora-detailer.safetensors": ["loras"],
            "ltx-2-19b-dev_audio_vae.safetensors": ["checkpoints"],
            "ltx-2-19b-dev_video_vae.safetensors": ["vae"]
        }

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                for filename in files:
                    if filename in exact_mapping:
                        src_path = os.path.join(root_dir, filename)
                        target_dirs = exact_mapping[filename]
                        for target_dir in target_dirs:
                            dest = os.path.join(base_models_dir, target_dir, filename)
                            if not os.path.exists(dest):
                                try: 
                                    os.symlink(src_path, dest)
                                    print(f"🔗 Linked target weight: {filename} -> models/{target_dir}")
                                except FileExistsError: pass

        self.s3 = boto3.client(
            service_name='s3', 
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", 
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], 
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], 
            region_name="auto"
        )

        # ==============================================================================
        # INJECT SAVER/LOADER CUSTOM NODES FOR SUBGRAPH INTERCONNECTIVITY
        # ==============================================================================
        saver_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/conditioning_saver.py"
        with open(saver_path, "w") as f:
            f.write('''import torch\nimport os\nimport folder_paths\n
class LTXVSaveConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING",)}, "optional": {"file_name": ("STRING", {"default": "conditioning.pt"}), "filename": ("STRING", {"default": "conditioning.pt"}), "dtype": ("STRING", {"default": "float16"})}}
    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "Lightricks/LTXVideo"
    OUTPUT_NODE = True\n
    def execute(self, conditioning, file_name="conditioning.pt", filename="conditioning.pt", dtype="float16"):
        output_dir = folder_paths.get_output_directory()
        fname = filename if filename != "conditioning.pt" else file_name
        if not fname.endswith(".pt"): fname += ".pt"
        torch.save(conditioning, os.path.join(output_dir, fname))
        return ()
''')
                
        loader_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/conditioning_loader.py"
        with open(loader_path, "w") as f:
            f.write('''import torch\nimport os\nimport folder_paths\n
class LTXVLoadConditioning:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_output_directory()
        files = [f for f in os.listdir(input_dir) if f.endswith(".pt") or f.endswith(".safetensors")] if os.path.exists(input_dir) else []
        return {"required": {"file_name": (files + ["(POSITIVE)conditioning.pt", "(NEGATIVE)conditioning.pt"],)}, "optional": {"filename": ("STRING", {"default": ""}), "device": ("STRING", {"default": "cpu"})}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "Lightricks/LTXVideo"\n
    def execute(self, file_name, filename="", device="cpu"):
        input_dir = folder_paths.get_output_directory()
        fname = filename if filename else file_name
        if not fname.endswith(".pt"): fname += ".pt"
        conditioning = torch.load(os.path.join(input_dir, fname), weights_only=False)
        return (conditioning,)
''')

        print("🚀 Launching Clean LTX Server Engine...")
        os.makedirs("/tmp/comfy_swap", exist_ok=True)
        os.makedirs("/tmp/hf_offload", exist_ok=True)

        env_vars = os.environ.copy()
        env_vars["TORCH_NUM_THREADS"] = "1"
        env_vars["OMP_NUM_THREADS"] = "1"
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
        env_vars["CUDA_MODULE_LOADING"] = "LAZY" 
        env_vars["MALLOC_TRIM_THRESHOLD_"] = "65536" 
        env_vars["HF_HUB_OFFLOAD_DIR"] = "/tmp/hf_offload"
        
        self.process = subprocess.Popen([
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--mmap-torch-files", "--cache-none", "--temp-directory", "/tmp/comfy_swap", 
            "--bf16-vae", "--disable-xformers", "--fp8_e4m3fn-text-enc"        
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        start_time = time.time()
        while time.time() - start_time < 300:
            if self.process.poll() is not None:
                print("❌ Startup Crash!")
                os._exit(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1) as response:
                    if response.status == 200:
                        print("⚡ LTX-2 API ONLINE!")
                        return
            except Exception:
                time.sleep(2)
        os._exit(1)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != os.environ.get("API_KEY"): 
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        body = await request.json()
        if isinstance(body, dict):
            if "json" in body: body = body["json"]
            elif "body" in body: body = body["body"]

        image_url = body.get("image_url") if isinstance(body, dict) else None
        requested_length = body.get("length") if isinstance(body, dict) else None

        # =========================================================================
        # 🔗 FUZZYLINKER: EXACT NODAL SPECIFIC FILE FORCING (Based on provided JSONs)
        # =========================================================================
        target_unet = "ltx-2-19b-distilled-fp8.safetensors"
        target_gemma = "gemma-3-12b-it-FP8.safetensors"
        target_connector = "ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
        target_video_vae = "ltx-2-19b-dev_video_vae.safetensors"
        target_audio_vae = "ltx-2-19b-dev_audio_vae.safetensors"
        target_detailer_lora = "ltx-2-19b-ic-lora-detailer.safetensors"

        def fuzzy_linker(wf_data):
            for node_id, node in wf_data.items():
                if not isinstance(node, dict) or "inputs" not in node:
                    continue
                
                cls = node.get("class_type", "")
                inputs = node["inputs"]
                
                # Model Forcing based on accurate Class Types from your JSON
                if cls == "UNETLoader":
                    inputs["unet_name"] = target_unet
                    inputs["weight_dtype"] = "fp8_e4m3fn"
                elif cls == "LTXAVTextEncoderLoader":
                    inputs["text_encoder"] = target_gemma
                    inputs["ckpt_name"] = target_connector
                elif cls in ["VAELoaderKJ", "VAELoader"]:
                    inputs["vae_name"] = target_video_vae
                elif cls == "LTXVAudioVAELoader":
                    inputs["ckpt_name"] = target_audio_vae
                elif cls == "LoraLoaderModelOnly":
                    inputs["lora_name"] = target_detailer_lora
                elif cls == "DenoMultiImageLoader":
                    inputs["image_paths"] = "input/dynamic_guides"

            # Frame length auto-sync
            if requested_length is not None:
                try:
                    tgt_len = int(requested_length)
                    if (tgt_len - 1) % 8 != 0:
                        tgt_len = ((tgt_len - 1) // 8) * 8 + 1
                        if tgt_len < 9: tgt_len = 9
                    
                    for node_id, node in wf_data.items():
                        if not isinstance(node, dict) or "inputs" not in node: continue
                        cls = node.get("class_type", "")
                        
                        # Subgraph 2 Frame Input
                        if cls == "EmptyLTXVLatentVideo": 
                            node["inputs"]["length"] = tgt_len
                        # Subgraph 3 Frame Input
                        if cls == "LTXVEmptyLatentAudio": 
                            node["inputs"]["frames_number"] = tgt_len
                            # Maintain the frame_rate configured by N8N overrides (or default)
                except Exception as e:
                    print(f"⚠️ Dynamic framing error: {e}")

            return wf_data

        dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
        if os.path.exists(dynamic_guides_dir):
            shutil.rmtree(dynamic_guides_dir)
        os.makedirs(dynamic_guides_dir, exist_ok=True)

        urls_to_download = []
        if image_url:
            if isinstance(image_url, list): urls_to_download = [str(u).strip() for u in image_url if str(u).strip()]
            elif isinstance(image_url, str) and image_url.strip():
                urls_to_download = [u.strip() for u in image_url.split(",") if u.strip()]

        if not urls_to_download:
            from PIL import Image
            img = Image.new('RGB', (1024, 1024), color='black')
            img.save(os.path.join(dynamic_guides_dir, "guide_0.png"))
        else:
            async def download_one(session, url_str, target_dest):
                from urllib.parse import urlparse
                parsed = urlparse(url_str)
                is_r2_storage = ("r2.cloudflarestorage.com" in url_str or "pub-" in url_str or parsed.netloc == "" or not parsed.scheme)
                
                if is_r2_storage:
                    file_key = parsed.path.lstrip('/')
                    while "//" in file_key: file_key = file_key.replace("//", "/")
                    await asyncio.get_event_loop().run_in_executor(None, self.s3.download_file, "video-asset-files-storage-workflow", file_key, target_dest)
                else:
                    try:
                        async with session.get(url_str, timeout=120) as r:
                            if r.status == 200:
                                f_content = await r.read()
                                with open(target_dest, "wb") as f: f.write(f_content)
                            else: raise Exception(f"HTTP code {r.status}")
                    except Exception:
                        await asyncio.get_event_loop().run_in_executor(None, urllib.request.urlretrieve, url_str, target_dest)

            async with aiohttp.ClientSession() as session:
                tasks = [download_one(session, url, os.path.join(dynamic_guides_dir, f"guide_{idx}.png")) for idx, url in enumerate(urls_to_download)]
                await asyncio.gather(*tasks)

        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        ram_task = asyncio.create_task(self._ram_squeezer())

        try:
            async with aiohttp.ClientSession() as session:
                
                # ==============================================================================
                # SUB-GRAPH 1: Text Conditioning
                # ==============================================================================
                sg1_raw = body.get("subgraph_1")
                if sg1_raw:
                    sg1 = json.loads(sg1_raw) if isinstance(sg1_raw, str) else sg1_raw
                    sg1 = self.merge_overrides(sg1, body.get("subgraph_1_override"))
                    sg1 = fuzzy_linker(sg1)
                    
                    print("🚀 Executing Sub-Graph 1 (Text Conditioning)...")
                    await self.execute_comfy_workflow(session, sg1)
                    print("💾 Phase 1 Complete. Purging VRAM...")
                    await self.clear_comfy_memory(session)

                # ==============================================================================
                # SUB-GRAPH 2: Main Video Latent Generation
                # ==============================================================================
                sg2_raw = body.get("subgraph_2") or body.get("workflow")
                if sg2_raw:
                    sg2 = json.loads(sg2_raw) if isinstance(sg2_raw, str) else sg2_raw
                    sg2 = self.merge_overrides(sg2, body.get("subgraph_2_override"))
                    sg2 = fuzzy_linker(sg2)

                    print("🚀 Executing Sub-Graph 2 (Main Video Generation)...")
                    await self.execute_comfy_workflow(session, sg2)
                    print("💾 Phase 2 Complete. Purging VRAM...")
                    await self.clear_comfy_memory(session)

                # ==============================================================================
                # SUB-GRAPH 3: Audio Co-Generation & Decoders
                # ==============================================================================
                sg3_raw = body.get("subgraph_3")
                if sg3_raw:
                    sg3 = json.loads(sg3_raw) if isinstance(sg3_raw, str) else sg3_raw
                    sg3 = self.merge_overrides(sg3, body.get("subgraph_3_override"))
                    sg3 = fuzzy_linker(sg3)

                    # ⚡ DYNAMIC LATENT FILE INTERCEPTOR
                    # ComfyUI's `SaveLatent` node appends dynamic suffixes (like `_00001`). 
                    # This safely grabs the exact file generated by Subgraph 2 and forces Node 232 to load it.
                    latent_files = [f for f in os.listdir(out_dir) if f.endswith('.latent')]
                    if latent_files:
                        latent_files.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)))
                        latest_latent = latent_files[-1]
                        
                        for node_id, node in sg3.items():
                            if node.get("class_type") == "LoadLatent":
                                node["inputs"]["latent"] = latest_latent
                                print(f"🔗 Dynamically linked Subgraph 3 LoadLatent node to: {latest_latent}")

                    print("🚀 Executing Sub-Graph 3 (Audio Generation & Combining)...")
                    await self.execute_comfy_workflow(session, sg3)
                    print("💾 Phase 3 Complete. Unloading VRAM...")
                    await self.clear_comfy_memory(session)

                # Find generated final video output
                videos = [v for v in os.listdir(out_dir) if v.endswith((".mp4", ".mkv", ".webm"))]
                if not videos:
                    raise HTTPException(status_code=500, detail="Output generation target missing. Did Subgraph 3 complete successfully?")
                    
                videos.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
                
                print("🧹 Generation Complete. Sending response stream...")
                with open(os.path.join(out_dir, videos[0]), "rb") as f:
                    return Response(content=f.read(), media_type="video/mp4")
        finally:
            ram_task.cancel()
