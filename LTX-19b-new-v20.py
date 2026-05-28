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
from fastapi import Request, Response, HTTPException, Header
from typing import Optional

# ==============================================================================
# PART 2: BASE IMAGE & OS CONFIGURATION
# ==============================================================================
base_image = modal.Image.from_registry(
    "nvidia/cuda:12.5.1-devel-ubuntu24.04", 
    add_python="3.12"
).apt_install(
    "git", "wget", "ffmpeg", "libgl1", "libglib2.0-0", 
    "build-essential", "ninja-build", "cmake", "clang", "llvm"
).env({
    "FORCE_REBUILD_INDEX": "121"  # Bumped to force the new Recursive Flattening Patch
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
    "python3.12 -m pip install --no-cache-dir fastapi aiohttp boto3 triton>=3.1.0 ninja setuptools>=70.0.0 wheel pip>=24.0",
    "python3.12 -m pip install --no-cache-dir pandas numexpr pytz python-dateutil scipy matplotlib colorama librosa soundfile decord imageio scikit-image numba einops bitsandbytes"
)

# ==============================================================================
# PART 4: COMFYUI & CUSTOM NODES CLONING + DEPENDENCY ISOLATION
# ==============================================================================

# 1. Install Exact Torch & SageAttention First
torch_image = build_image.run_commands(
    "python3.12 -m pip install --no-cache-dir torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124",
    "python3.12 -m pip install --no-cache-dir diffusers accelerate transformers torchsde numpy==1.26.4 kornia==0.7.3",
    "python3.12 -m pip install --no-cache-dir sageattention==1.0.6"
)

# 2. Clone Repositories
clone_image = torch_image.run_commands(
    "git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI",
    "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
    "git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo",
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use",
    "git clone https://github.com/Deno2026/comfyui-deno-custom-nodes.git /workspace/ComfyUI/custom_nodes/comfyui-deno-custom-nodes",
    "git clone https://github.com/cubiq/ComfyUI_essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_essentials",
    "git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes",
    "git clone https://github.com/SquirrelRat/MultiString-Prompts.git /workspace/ComfyUI/custom_nodes/MultiString-Prompts",
    "git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts",
    "git clone https://github.com/IvanRybakov/comfyui-node-int-to-string-convertor.git /workspace/ComfyUI/custom_nodes/comfyui-node-int-to-string-convertor",
    "git clone https://github.com/siraxe/ComfyUI-LTX-FDG.git /workspace/ComfyUI/custom_nodes/ComfyUI-LTX-FDG"
)

# 3. Strip torch identifiers from node requirements, and install the rest.
deps_image = clone_image.run_commands(
    "sed -i '/torch/d' /workspace/ComfyUI/requirements.txt",
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec sed -i '/torch/d' {} \;",
    "python3.12 -m pip install --no-cache-dir -r /workspace/ComfyUI/requirements.txt",
    r"find /workspace/ComfyUI/custom_nodes -name 'requirements.txt' -exec python3.12 -m pip install --no-cache-dir -r {} \;"
)

# 4. Inject Patches to bypass Core crashes
final_image = deps_image.run_commands(
    # 🔥 PATCH 1: Fix FizzNodes NoneType crash
    "sed -i 's/final_pooled_output = torch.cat(pooled_out, dim=0)/final_pooled_output = torch.cat([p for p in pooled_out if p is not None], dim=0) if any(p is not None for p in pooled_out) else None/g' /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes/BatchFuncs.py",
    
    # 🔥 PATCH 2: Safely bypass LTXVideo looping_sampler missing raw_conds
    "python3 -c \"filepath = '/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/looping_sampler.py'; code = open(filepath).read(); code = code.replace('positive, negative = guider.raw_conds', 'positive, negative = getattr(guider, \\'raw_conds\\', None) or (getattr(guider, \\'original_conds\\', {}).get(\\'positive\\'), getattr(guider, \\'original_conds\\', {}).get(\\'negative\\'))'); open(filepath, 'w').write(code)\"",
    
    # 🔥 PATCH 3: Global SageAttention API remap
    "echo '' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py",
    "echo 'sageattn_qk_int8_pv_fp16_triton = sageattn' >> /usr/local/lib/python3.12/site-packages/sageattention/__init__.py",

    # 🔥 PATCH 4: FORCE RESOLVE Nested Conditioning Arrays from DenoLTXSequencer
    "python3 -c \"filepath = '/workspace/ComfyUI/comfy/sampler_helpers.py'; code = open(filepath).read(); replacement = 'def convert_cond(cond):\\n    def flatten(x):\\n        res = []\\n        for c in x:\\n            if isinstance(c, list) and len(c) > 0 and isinstance(c[0], list): res.extend(flatten(c))\\n            else: res.append(c)\\n        return res\\n    if isinstance(cond, list): cond = flatten(cond)\\n'; code = code.replace('def convert_cond(cond):', replacement); open(filepath, 'w').write(code)\"",
    
    env={
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", ""),
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.9"
    }
)

# ==============================================================================
# PART 5: MODAL APP CONFIGURATION
# ==============================================================================
app = modal.App("media-worker")

weights_volume = modal.Volume.from_name("ltx-new-version20-weights", create_if_missing=False)

@app.cls(
    gpu="L4", 
    image=final_image, 
    volumes={"/mnt/weights": weights_volume},
    secrets=[modal.Secret.from_name("custom-secret")],
    memory=8192, 
    scaledown_window=30,
    timeout=3600 
)
class LTXEngine:
    
    def _log_reader(self):
        for line in iter(self.process.stdout.readline, ""):
            if line: print(f"[ComfyUI] {line.strip()}")

    async def _ram_squeezer(self):
        while True:
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1\n')
            except Exception:
                try: ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception: pass
            await asyncio.sleep(10)

    @modal.enter()
    def start_comfy(self):
        import boto3
        print("🔗 Running Atomic Model Folder Linker...")
        base_models_dir = "/workspace/ComfyUI/models"
        
        dirs = ["unet", "vae", "clip", "text_encoders", "text_encoder", "checkpoints", "diffusion_models", "gguf", "loras"]
        for d in dirs: os.makedirs(os.path.join(base_models_dir, d), exist_ok=True)

        if os.path.exists("/mnt/weights"):
            for root_dir, _, files in os.walk("/mnt/weights"):
                for filename in files:
                    if not filename.endswith((".safetensors", ".gguf", ".pth", ".pt", ".bin")): continue
                    src_path = os.path.join(root_dir, filename)
                    for target_dir in ["unet", "vae", "clip", "text_encoders", "text_encoder", "checkpoints", "diffusion_models", "loras"]:
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

        saver_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/conditioning_saver.py"
        if os.path.exists(saver_path):
            with open(saver_path, "w") as f:
                f.write('''import torch
import os
import folder_paths

class LTXVSaveConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "file_name": ("STRING", {"default": "conditioning.pt"}),
                "filename": ("STRING", {"default": "conditioning.pt"}),
                "dtype": ("STRING", {"default": "float16"}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "Lightricks/LTXVideo"
    OUTPUT_NODE = True

    def execute(self, conditioning, file_name="conditioning.pt", filename="conditioning.pt", dtype="float16"):
        output_dir = folder_paths.get_output_directory()
        fname = filename if filename != "conditioning.pt" else file_name
        if not fname.endswith(".pt"):
            fname += ".pt"
        file_path = os.path.join(output_dir, fname)
        torch.save(conditioning, file_path)
        return ()
''')
                
        loader_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/conditioning_loader.py"
        if os.path.exists(loader_path):
            with open(loader_path, "w") as f:
                f.write('''import torch
import os
import folder_paths

class LTXVLoadConditioning:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_output_directory()
        files = [f for f in os.listdir(input_dir) if f.endswith(".pt") or f.endswith(".safetensors")] if os.path.exists(input_dir) else []
        return {
            "required": {
                "file_name": (files + ["(POSITIVE)conditioning.pt", "(NEGATIVE)conditioning.pt"],),
            },
            "optional": {
                "filename": ("STRING", {"default": ""}),
                "device": ("STRING", {"default": "cpu"}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "Lightricks/LTXVideo"

    def execute(self, file_name, filename="", device="cpu"):
        input_dir = folder_paths.get_output_directory()
        fname = filename if filename else file_name
        if not fname.endswith(".pt"):
            fname += ".pt"
        file_path = os.path.join(input_dir, fname)
        conditioning = torch.load(file_path, weights_only=False)
        return (conditioning,)
''')

        print("🚀 Launching Hybrid-Memory LTX Server Engine with FP8 and SageAttention configurations...")
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
            "python3.12", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--mmap-torch-files", "--cache-none", "--temp-directory", "/tmp/comfy_swap", 
            "--bf16-vae", "--use-sage-attention", "--fp8_e4m3fn-unet", "--fp8_e4m3fn-text-enc"
        ], cwd="/workspace/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env_vars)
        
        self.t = threading.Thread(target=self._log_reader, daemon=True)
        self.t.start()

        start_time = time.time()
        while time.time() - start_time < 300:
            if self.process.poll() is not None: os._exit(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=1) as response:
                    if response.status == 200: return
            except Exception: time.sleep(2)
        os._exit(1)

    async def clear_comfy_memory(self, session):
        try:
            async with session.post("http://127.0.0.1:8188/free", json={"unload_models": True, "free_memory": True}) as r:
                await r.read()
        except Exception as e:
            print(f"[Warning] Memory unload endpoint failed to execute: {e}")
        
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
            if r.status != 200:
                err_text = await r.text()
                raise HTTPException(status_code=500, detail=f"Failed to queue sub-graph prompt: {r.status} - {err_text}")
            res = await r.json()
            prompt_id = res["prompt_id"]

        print(f"⌛ Queued workflow step. prompt_id: {prompt_id}. Polling state...")
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
                raise HTTPException(status_code=500, detail="ComfyUI server process crashed during workflow execution.")
                
            await asyncio.sleep(1)

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

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request, x_api_key: Optional[str] = Header(None)):
        if x_api_key != "testing-modal-workflow-2": 
            raise HTTPException(status_code=403, detail="Unauthorized Account 2 Pipeline Request")
        
        body = await request.json()
        if isinstance(body, dict):
            if "json" in body: body = body["json"]
            elif "body" in body: body = body["body"]

        incoming_image_urls = body.get("image_url")
        requested_length = int(body.get("length", 73))
        prompts_dict = body.get("prompts", {})
        negative_prompt = body.get("negative", "worst quality, blurry, low resolution, artifacts, watermarks")

        if isinstance(prompts_dict, dict):
            try:
                sorted_keys = sorted(prompts_dict.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
                prompts_list = [str(prompts_dict[k]).strip() for k in sorted_keys if str(prompts_dict[k]).strip()]
                prompts_timeline_str = "|".join(prompts_list)
            except Exception as e:
                print(f"[Warning] Failed to parse prompts dict numerically: {e}")
                prompts_timeline_str = "|".join([str(v).strip() for v in prompts_dict.values()])
        else:
            prompts_timeline_str = str(prompts_dict)

        target_unet = "ltx-2-19b-distilled-fp8.safetensors"
        target_gemma = "gemma-3-12b-it-FP8.safetensors"
        target_connector = "ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
        target_video_vae = "ltx-2-19b-dev_video_vae.safetensors"
        target_audio_vae = "ltx-2-19b-dev_audio_vae.safetensors"
        target_detailer_lora = "ltx-2-19b-ic-lora-detailer.safetensors"

        # ==============================================================================
        # [FIX 2] DYNAMIC IMAGE DOWNLOADING TO ABSOLUTE PATHS
        # ==============================================================================
        dynamic_guides_dir = "/workspace/ComfyUI/input/dynamic_guides"
        if os.path.exists(dynamic_guides_dir): shutil.rmtree(dynamic_guides_dir)
        os.makedirs(dynamic_guides_dir, exist_ok=True)

        urls_to_download = []
        if incoming_image_urls:
            if isinstance(incoming_image_urls, list): urls_to_download = [str(u).strip() for u in incoming_image_urls if str(u).strip()]
            elif isinstance(incoming_image_urls, str) and incoming_image_urls.strip():
                urls_to_download = [u.strip() for u in incoming_image_urls.split(",") if u.strip()]

        image_filenames = []
        if not urls_to_download:
            fallback_path = os.path.join(dynamic_guides_dir, "guide_0.png")
            from PIL import Image
            img = Image.new('RGB', (384, 480), color='black')
            img.save(fallback_path)
            image_filenames = [fallback_path] # Use Absolute path
        else:
            async def download_one(session, url_str, target_dest):
                from urllib.parse import urlparse
                import botocore.exceptions
                try:
                    parsed = urlparse(url_str)
                    if "r2.cloudflarestorage.com" in url_str or "pub-" in url_str or parsed.netloc == "" or not parsed.scheme:
                        file_key = parsed.path.lstrip('/')
                        while "//" in file_key: file_key = file_key.replace("//", "/")
                        print(f"📥 Downloading dynamic key from R2: {file_key}")
                        await asyncio.get_event_loop().run_in_executor(
                            None, 
                            self.s3.download_file, 
                            "video-asset-files-storage-workflow", 
                            file_key, 
                            target_dest
                        )
                    else:
                        print(f"📥 Downloading dynamic public URL: {url_str}")
                        async with session.get(url_str, timeout=120) as r:
                            if r.status == 200:
                                with open(target_dest, "wb") as f: f.write(await r.read())
                            else:
                                print(f"[Warning] HTTP status {r.status} for URL: {url_str}")
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "404":
                        print(f"[Warning] Key not found in R2: {url_str}")
                    else:
                        print(f"[Warning] R2 ClientError downloading {url_str}: {e}")
                except Exception as e:
                    print(f"[Warning] Unexpected error downloading {url_str}: {e}")
                
                if not os.path.exists(target_dest):
                    from PIL import Image
                    img = Image.new('RGB', (384, 480), color='black')
                    img.save(target_dest)
                    print(f"[Fallback] Generated blank placeholder guide image at: {target_dest}")

            async with aiohttp.ClientSession() as download_session:
                tasks = [download_one(download_session, url, os.path.join(dynamic_guides_dir, f"guide_{i}.png")) for i, url in enumerate(urls_to_download)]
                await asyncio.gather(*tasks)
            
            # Map directly to Absolute Path
            image_filenames = [os.path.join(dynamic_guides_dir, f"guide_{i}.png") for i in range(len(urls_to_download))]

        out_dir = "/workspace/ComfyUI/output"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        ram_task = asyncio.create_task(self._ram_squeezer())

        try:
            async with aiohttp.ClientSession() as session:
                
                # ==============================================================================
                # SUB-GRAPH 1
                # ==============================================================================
                sg1_raw = body.get("subgraph_1")
                if sg1_raw:
                    sg1 = json.loads(sg1_raw) if isinstance(sg1_raw, str) else sg1_raw
                else:
                    with open("comfyui-ltx-20-subgraph-1(api).json", "r") as f:
                        sg1 = json.load(f)
                
                sg1 = self.merge_overrides(sg1, body.get("subgraph_1_override"))

                if "243" in sg1:
                    sg1["243"]["inputs"]["text_encoder"] = target_gemma
                    sg1["243"]["inputs"]["ckpt_name"] = target_connector
                    sg1["243"]["inputs"]["device"] = "default" 
                if "246" in sg1:
                    sg1["246"]["inputs"]["prompts"] = prompts_timeline_str
                if "112" in sg1:
                    sg1["112"]["inputs"]["text"] = negative_prompt
                if "242" in sg1:
                    sg1["242"]["inputs"]["filename"] = "(NEGATIVE)conditioning"
                if "244" in sg1:
                    sg1["244"]["inputs"]["filename"] = "(POSITIVE)conditioning"

                print("🚀 Executing Sub-Graph 1 (Text Conditioning)...")
                await self.execute_comfy_workflow(session, sg1)
                
                print("💾 Phase 1 Complete. Purging VRAM...")
                await self.clear_comfy_memory(session)

                # ==============================================================================
                # SUB-GRAPH 2
                # ==============================================================================
                sg2_raw = body.get("subgraph_2")
                if sg2_raw:
                    sg2 = json.loads(sg2_raw) if isinstance(sg2_raw, str) else sg2_raw
                else:
                    with open("comfyui-ltx-20-subgraph-2(api).json", "r") as f:
                        sg2 = json.load(f)

                sg2 = self.merge_overrides(sg2, body.get("subgraph_2_override"))

                if "194" in sg2:
                    sg2["194"]["inputs"]["length"] = requested_length
                if "238" in sg2:
                    sg2["238"]["inputs"]["unet_name"] = target_unet
                    sg2["238"]["inputs"]["weight_dtype"] = "fp8_e4m3fn" 
                if "241" in sg2:
                    sg2["241"]["inputs"]["vae_name"] = target_video_vae
                if "245" in sg2:
                    sg2["245"]["inputs"]["file_name"] = "(POSITIVE)conditioning.pt"
                if "246" in sg2:
                    sg2["246"]["inputs"]["file_name"] = "(NEGATIVE)conditioning.pt"
                if "237" in sg2:
                    sg2["237"]["inputs"]["image_paths"] = "\n".join(image_filenames) 
                
                # ======================================================================================
                # [FIX 1]: Dynamically inject the missing "frame_0" keys to sync images correctly
                # ======================================================================================
                if "235" in sg2:
                    num_imgs = len(image_filenames)
                    sg2["235"]["inputs"]["num_images"] = num_imgs
                    for i in range(num_imgs):
                        if f"frame_{i}" not in sg2["235"]["inputs"]:
                            sg2["235"]["inputs"][f"frame_{i}"] = 0 if num_imgs == 1 else int(i * (requested_length - 1) / (num_imgs - 1))

                if "248" in sg2:
                    sg2["248"]["inputs"]["lora_name"] = target_detailer_lora

                print("🚀 Executing Sub-Graph 2 (Main Video Generation)...")
                await self.execute_comfy_workflow(session, sg2)

                print("💾 Phase 2 Complete. Purging VRAM...")
                await self.clear_comfy_memory(session)

                # ======================================================================================
                # [FIX 3]: Copy Latent output to Latent input with strict names so LoadLatent finds it
                # ======================================================================================
                generated_latents = [f for f in os.listdir(out_dir) if f.startswith("video_latent_output") and f.endswith(".latent")]
                if generated_latents:
                    os.makedirs("/workspace/ComfyUI/input", exist_ok=True)
                    generated_latents.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)))
                    latest_latent = generated_latents[-1]
                    latent_out_path = os.path.join(out_dir, latest_latent)
                    latent_in_path = "/workspace/ComfyUI/input/video_latent_output.latent"
                    shutil.copy(latent_out_path, latent_in_path)
                    print(f"🔗 Shifted saved latent {latest_latent} into input path for Subgraph 3.")

                # ==============================================================================
                # SUB-GRAPH 3
                # ==============================================================================
                sg3_raw = body.get("subgraph_3")
                if sg3_raw:
                    sg3 = json.loads(sg3_raw) if isinstance(sg3_raw, str) else sg3_raw
                else:
                    with open("comfyui-ltx-20-Subgraph-3(api).json", "r") as f:
                        sg3 = json.load(f)

                sg3 = self.merge_overrides(sg3, body.get("subgraph_3_override"))

                if "232" in sg3:
                    sg3["232"]["inputs"]["latent"] = "video_latent_output.latent"
                if "278" in sg3:
                    sg3["278"]["inputs"]["unet_name"] = target_unet
                    sg3["278"]["inputs"]["weight_dtype"] = "fp8_e4m3fn" 
                if "282" in sg3:
                    sg3["282"]["inputs"]["file_name"] = "(POSITIVE)conditioning.pt"
                if "283" in sg3:
                    sg3["283"]["inputs"]["file_name"] = "(NEGATIVE)conditioning.pt"
                if "295" in sg3:
                    sg3["295"]["inputs"]["ckpt_name"] = target_audio_vae
                if "296" in sg3:
                    sg3["296"]["inputs"]["vae_name"] = target_video_vae
                
                # ======================================================================================
                # [FIX 4]: Change Audio latent frames calculation to EXACT dimensional representation formula
                # ======================================================================================
                if "290" in sg3:
                    sg3["290"]["inputs"]["frames_number"] = (requested_length * 2) - 1 
                
                if "298" in sg3:
                    sg3["298"]["inputs"]["format"] = "video/h264-mp4"
                    sg3["298"]["inputs"]["frame_rate"] = 24
                if "302" in sg3:
                    sg3["302"]["inputs"]["lora_name"] = target_detailer_lora

                print("🚀 Executing Sub-Graph 3 (Audio Generation & Combine Decoders)...")
                await self.execute_comfy_workflow(session, sg3)
                print("💾 Phase 3 Complete. Unloading VRAM...")
                await self.clear_comfy_memory(session)

                output_files = []
                for root_p, _, filenames in os.walk(out_dir):
                    for name in filenames:
                        if name.endswith((".mp4", ".gif", ".webm")):
                            output_files.append(os.path.join(root_p, name))

                if not output_files:
                    raise HTTPException(status_code=500, detail="Inference finished but no combined output media files were detected in ComfyUI workspace.")
                
                output_files.sort(key=os.path.getmtime)
                target_video_file = output_files[-1]
                saved_filename = os.path.basename(target_video_file)

                target_key = f"outputs/{int(time.time())}_{saved_filename}"
                print(f"📤 Uploading compiled video containing audio track to R2: {target_key}")
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.s3.upload_file,
                    "video-asset-files-storage-workflow",
                    target_video_file,
                    target_key
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
