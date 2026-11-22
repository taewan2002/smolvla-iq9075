"""LeRobot reference backend and pinned Hub snapshots."""
import hashlib
from pathlib import Path

MODEL = "HuggingFaceVLA/smolvla_libero"
REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
VLM = "HuggingFaceTB/SmolVLM2-500M-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshots():
    from huggingface_hub import snapshot_download

    checkpoint = snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors"])
    vlm = snapshot_download(
        VLM, revision=VLM_REVISION,
        allow_patterns=["*.json", "merges.txt"],
    )
    return Path(checkpoint), Path(vlm)


class NativePolicy:
    def __init__(self, checkpoint, vlm, device):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.vlm_model_name = str(vlm)
        config.load_vlm_weights = False
        config.device = device
        config.push_to_hub = False
        config.pretrained_path = Path(checkpoint)
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": device},
                "tokenizer_processor": {"tokenizer_name": str(vlm)},
            },
        )
        self.action_shape = (1, config.chunk_size, config.max_action_dim)
        self.image_keys = tuple(config.image_features)

    def __call__(self, rgb, state, task, noise):
        torch = self.torch
        batch = {
            key: torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255)[None]
            for key, image in zip(self.image_keys, rgb, strict=True)
        }
        batch["observation.state"] = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        batch["task"] = [task]
        with torch.inference_mode():
            action = self.policy.predict_action_chunk(
                self.pre(batch), noise=torch.from_numpy(noise).to(self.device)
            )
            return self.post(action).cpu().numpy()

    def predict_batch(self, rgb, state, task, noise):
        import numpy as np

        torch = self.torch
        inputs = {
            key: torch.from_numpy(np.stack([views[index] for views in rgb])).permute(0, 3, 1, 2).float().div_(255)
            for index, key in enumerate(self.image_keys)
        }
        inputs["observation.state"] = torch.as_tensor(state, dtype=torch.float32)
        inputs["task"] = task
        with torch.inference_mode():
            action = self.policy.predict_action_chunk(
                self.pre(inputs), noise=torch.from_numpy(noise).to(self.device)
            )
            return self.post(action).cpu().numpy()
