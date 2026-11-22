# Reproduce the evaluation

On an x86-64 Linux workstation with an NVIDIA GPU, create a Python 3.11 environment. The benchmark used PyTorch 2.7.1 with CUDA 12.8, Transformers 4.51.3, LeRobot v0.4.4 source, ONNX 1.19.1, MuJoCo 2.3.7 and Robosuite 1.4.1. The pinned LeRobot source is loaded over the LeRobot 0.3.3 package environment to select the v0.4.4 policy and processor code while retaining the tested Transformers release.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[export,cpu]' lerobot==0.3.3 transformers==4.51.3 \
  numpy==1.26.4 accelerate==1.7.0 huggingface-hub==0.34.4 num2words==0.5.14 \
  pyarrow==19.0.1 pandas==2.2.3 diffusers==0.33.1
python -m pip install mujoco==2.3.7 robosuite==1.4.1 bddl==1.0.1 easydict gym==0.25.2

git clone https://github.com/huggingface/lerobot.git upstream/lerobot
git -C upstream/lerobot checkout 8fff0fde7c79f23a93d845d1a50e985de01f8b8a
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git upstream/LIBERO
git -C upstream/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
export PYTHONPATH="$PWD/upstream/lerobot/src:$PWD/upstream/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
```

EGL needs access to the NVIDIA graphics driver. If you use an NVIDIA container, expose the GPU and include `graphics` in `NVIDIA_DRIVER_CAPABILITIES`. LIBERO's task definition and initial state files come from its Git checkout. The benchmark runs against these assets directly.

## Export and generate fixtures

```bash
python -m smolvla_iq9075.export --output bundles/libero
python -m smolvla_iq9075.libero --engine native \
  --libero-root upstream/LIBERO/libero/libero --device cuda \
  --execute 10 --inits 1 --save-fixtures runs/fixtures --output runs/native.jsonl
```

Both commands fetch `lerobot/smolvla_libero` at revision `a24f1f3` and the pinned SmolVLM2 config and tokenizer. If Hub Xet downloads stall on your network, run the commands with `HF_HUB_DISABLE_XET=1`. For an offline export or a compatible local checkpoint, supply `--checkpoint DIR --vlm DIR` with both snapshots already present. Each export uses about 1.5 GiB of bundle storage and writes a SHA-256 manifest. Choose an empty output path for each export. [Checkpoint](https://huggingface.co/lerobot/smolvla_libero/tree/a24f1f380461b665158d52acd140152c79e8b36d).

```bash
scp -r bundles/libero runs/fixtures user@iq9075:~/
```

## Latency on IQ-9075

Use the board's Qualcomm Linux image and a Python environment with QNN support. The evaluation used Python 3.12, ONNX Runtime 1.30.0 and ONNX Runtime QNN 2.6.0. Follow the provider's [Linux ARM64 installation instructions](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md) to match your board image. With a compatible image and wheel:

```bash
python -m pip install -e . onnxruntime==1.30.0 onnxruntime-qnn==2.6.0
ls /dev/fastrpc-cdsp /dev/fastrpc-cdsp1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m smolvla_iq9075.bench --bundle ~/libero --fixtures ~/fixtures \
  --output runs/iq9075.json --engines cpu htp dual --repeats 1 --warmup 3 --limit 20 --execute 10
```

Run with an idle board. `cpu` uses the serialized FP32 graphs with ORT CPU. `htp` runs the graph stages on HTP 0. `dual` runs one camera on each HTP concurrently and places successive language partitions on alternating HTPs. QNN enables its FP16 path and rejects CPU fallback inside these graph sessions. Preprocessing, cross-attention K/V projections, and output scaling are explicit host operations in all three engines.

The timers start with RGB arrays already available and end after action scaling. Warm-ups and graph compilation are excluded from the latency distribution. The report includes session initialization time, per-stage samples, action error against the native output for the full chunk and first 10 actions, and the HTP device files opened. The timers exclude physical cameras, transport, robot drivers and motor control.

### Adreno GPU

On the workstation, export the same checkpoint with rank-four prefill K/V outputs and copy the bundle to the board.

```bash
python -m smolvla_iq9075.export --adreno-compatible --output bundles/libero-adreno
scp -r bundles/libero-adreno user@iq9075:~/
```

On IQ-9075, use the QNN package's `libQnnGpu.so` and give the benchmark process access to the Adreno device node. Use the same fixtures and thread settings as the CPU and HTP runs.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m smolvla_iq9075.bench --bundle ~/libero-adreno --fixtures ~/fixtures \
  --output runs/iq9075-adreno.json --engines gpu --repeats 1 --warmup 3 --limit 20 --execute 10
```

The runtime restores the singleton batch dimension before passing the K/V arrays to the Action Expert. It selects the QNN GPU library explicitly, disables CPU fallback for all graph sessions, and records the opened GPU device nodes in the report. [QNN GPU backend documentation](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md#running-a-model-with-qnn-eps-gpu-backend).

For the FP16 pass, convert the FP32 Adreno bundle on the workstation. This keeps the checkpoint and ten Euler steps. The converter expresses reciprocal as division by an explicit FP16 constant for QNN GPU. It rescales the RMSNorm calculation so both the epsilon constant and its contribution to zero rows are normal FP16 values. It writes a manifest that identifies the source FP32 bundle.

```bash
python -m pip install onnxconverter-common==1.16.0
python -m smolvla_iq9075.adreno_fp16 --bundle bundles/libero-adreno \
  --output bundles/libero-adreno-fp16
scp -r bundles/libero-adreno-fp16 user@iq9075:~/
```

Use the same board command, with `--bundle ~/libero-adreno-fp16 --output runs/iq9075-adreno-fp16.json`. The runtime feeds FP16 tensors to the graph sessions and uses FP32 for the host K/V projections and action scaling.

## Native PyTorch latency on IQ-9075

Use a separate CPU-only Python environment for the LeRobot reference. On the board, install the matching ARM64 wheels from PyPI and use the same pinned LeRobot source checkout described above.

```bash
python3 -m venv .venv-native
source .venv-native/bin/activate
python -m pip install -e . torch==2.7.1 torchvision==0.22.1 \
  lerobot==0.3.3 transformers==4.51.3 numpy==1.26.4 accelerate==1.7.0 \
  huggingface-hub==0.34.4 num2words==0.5.14 pyarrow==19.0.1 pandas==2.2.3 diffusers==0.33.1
export PYTHONPATH="$PWD/upstream/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m smolvla_iq9075.bench --bundle ~/libero --fixtures ~/fixtures \
  --output runs/iq9075-native.json --engines native --repeats 1 --warmup 3 --limit 20 --execute 10
```

The runner uses four PyTorch threads and verifies the native checkpoint SHA-256 against the ONNX bundle. Use the same fixtures and run the CPU and HTP timing passes separately.

## Run closed-loop LIBERO with the board

The [full native LeRobot benchmark](libero-full.md) provides the batched, 2,000-episode evaluation commands.

On IQ-9075, start the inference service. It listens on loopback. Add `--serve-port 18075` to the timing command to serve from the final engine and reuse the loaded sessions.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m smolvla_iq9075.remote --bundle ~/libero --engine dual --port 18075
```

Forward the service to the workstation from a second shell.

```bash
ssh -N -L 127.0.0.1:18075:127.0.0.1:18075 user@iq9075
```

Run the paired episodes in the original evaluation environment.

```bash
python -m smolvla_iq9075.libero --engine remote --url http://127.0.0.1:18075 \
  --libero-root upstream/LIBERO/libero/libero \
  --execute 10 --inits 1 --output runs/dual.jsonl
python -m smolvla_iq9075.compare runs/native.jsonl runs/dual.jsonl
```

Both rollouts predict 50 actions with 10 flow steps and apply the first 10 actions before observing again. The published checkpoint saves `n_action_steps=50`. Pass `--execute 50` to both runners to use that cadence. The 40-episode protocol uses one fixed published initial state for each task across LIBERO Spatial, Object, Goal and Long. Increase `--inits` in both commands for additional initial states. The comparison validates seed, initial observation hash, checkpoint hash, task assets and execution length for every episode before computing success rates. The episode log records simulator wall time, including RPC for remote runs. The board latency benchmark measures inference locally.
