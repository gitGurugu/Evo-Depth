"""CPU 上把一张图送进 InternVL3-1B 视觉塔，打印每块的输入和输出。

python trace_vit_blocks_cpu.py [image.jpg]
没给图就用一张合成图。大张量只打 shape 和前 8 个数。
"""
import sys
from PIL import Image, ImageDraw

import torch
from torchvision import transforms as T
from transformers import AutoModel

MODEL = "OpenGVLab/InternVL3-1B"


def peek(t):
    nums = " ".join(f"{v:.4f}" for v in t.detach().float().cpu().flatten()[:8])
    return f"shape={tuple(t.shape)}  前8个: {nums}"


def capture(name, note, saved):
    def hook(module, inputs, output):
        inp = inputs[0]
        out = output[0] if isinstance(output, tuple) else output
        print(f"\n{name}")
        print(f"  {note}")
        print("  输入", peek(inp))
        print("  输出", peek(out))
        saved[name] = out.detach()
    return hook


def main():
    if len(sys.argv) > 1:
        image = Image.open(sys.argv[1]).convert("RGB")
        print(f"图片: {sys.argv[1]}  {image.size}")
    else:
        image = Image.new("RGB", (640, 480), (180, 40, 40))
        ImageDraw.Draw(image).rectangle([40, 40, 280, 420], fill=(240, 220, 40))
        print("没给图片，用合成图 640x480")

    print("加载", MODEL, "CPU float32")
    model = AutoModel.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    ).eval()

    x = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])(image).unsqueeze(0)
    print("\n模型输入 pixel_values  (1张图, RGB, 448, 448)")
    print(" ", peek(x))

    saved = {}
    layer0 = model.vision_model.encoder.layers[0]
    targets = [
        ("Patch Embedding", "卷积把图切成 patch", model.vision_model.embeddings.patch_embedding),
        ("Embeddings", "patch 拉成 token，拼上 CLS，加上位置编码", model.vision_model.embeddings),
        ("第1层 QKV", "这一层注意力的输入，线性投影成 Q、K、V", layer0.attn.qkv),
        ("第1层 Attention", "同一层，QKV 算完注意力后的向量", layer0.attn),
        ("第1个 Block", "注意力残差加 MLP 残差，形状和输入相同", layer0),
        ("Projector mlp1", "去掉 CLS 并 pixel shuffle 之后，投到语言模型宽度", model.mlp1),
    ]
    hooks = [m.register_forward_hook(capture(n, note, saved)) for n, note, m in targets]

    with torch.inference_mode():
        features = model.extract_feature(x)

    qkv = saved["第1层 QKV"]
    heads = model.config.vision_config.num_attention_heads
    head_dim = qkv.shape[-1] // (3 * heads)
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, heads, head_dim)
    print("\nQKV 重排  (batch, token, 3=Q/K/V, head, head_dim)")
    print(" ", peek(qkv))
    q0 = " ".join(f"{v:.4f}" for v in qkv[0, 0, 0, 0, :8].float())
    print(" Token0 Head0 的 Q 前8个:", q0)

    print("\nextract_feature 返回值")
    print(" ", peek(features))
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
