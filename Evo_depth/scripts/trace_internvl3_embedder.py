"""两张图走过 internvl3_embedder，打印每步张量，存一张对照图。

python trace_internvl3_embedder.py [a.jpg b.jpg] [--model]
没给图就用两张合成图。不加 --model 不加载 InternVL3-1B。
"""
import os
import sys
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision.transforms.functional import to_pil_image
from model.internvl3.internvl3_embedder import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    InternVL3Embedder,
    build_transform,
    dynamic_preprocess,
)

SIZE = 448
# 名字按 build_transform 里的顺序，方便对着打印看
STEPS = ("to_rgb", "resize_448", "to_tensor", "imagenet_norm")


def brief(x):
    if isinstance(x, Image.Image):
        return f"PIL {x.mode} {x.size[0]}x{x.size[1]}"
    t = x.detach().float().cpu()
    return (
        f"shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={t.min():.3f} max={t.max():.3f} mean={t.mean():.3f}"
    )


def denorm(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    view = (t.detach().float().cpu() * std + mean).clamp(0, 1)
    return to_pil_image(view)


def load_images(paths):
    if not paths:
        a = Image.new("RGB", (640, 480), (180, 40, 40))
        b = Image.new("RGB", (800, 300), (40, 80, 180))
        ImageDraw.Draw(a).rectangle([40, 40, 280, 420], fill=(240, 220, 40))
        ImageDraw.Draw(b).ellipse([80, 30, 520, 260], fill=(240, 240, 240))
        print("没给路径，用两张合成图：640x480 和 800x300")
        return [a, b]
    if len(paths) != 2:
        raise SystemExit("用法: python trace_internvl3_embedder.py a.jpg b.jpg [--model]")
    return [Image.open(p).convert("RGB") for p in paths]


def save_sheet(panels, path):
    label_h = 28
    canvas = Image.new("RGB", (SIZE * len(panels), SIZE + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (im, title) in enumerate(panels):
        canvas.paste(im.resize((SIZE, SIZE)), (i * SIZE, label_h))
        draw.text((i * SIZE + 8, 6), title, fill=(0, 0, 0))
    canvas.save(path)
    print("对照图:", path)


def walk_images(images):
    """对应 InternVL3Embedder._preprocess_images，只走到 pixel_values。"""
    transform = build_transform(SIZE)
    panels, tensors = [], []
    for i, im in enumerate(images):
        print(f"\n[{i}] 输入: {brief(im)}")
        tiles = dynamic_preprocess(im, image_size=SIZE)
        print(f"[{i}] dynamic_preprocess: {len(tiles)} 块, 每块 {tiles[0].size}")
        for tile in tiles:
            x = tile
            for name, step in zip(STEPS, transform.transforms):
                x = step(x)
                print(f"[{i}] {name}: {brief(x)}")
            tensors.append(x)
        panels += [
            (im, f"img{i} raw {im.size[0]}x{im.size[1]}"),
            (denorm(x), f"img{i} norm->rgb"),
        ]
    pixel_values = torch.stack(tensors).bfloat16()
    print(f"\npixel_values: {brief(pixel_values)}")
    return panels


def walk_model(images):
    """对应 get_fused_image_text_embedding_from_tensor_images 里模型那一段。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n加载 InternVL3-1B, device={device}")
    embedder = InternVL3Embedder(device=device)
    mask = torch.ones(len(images), dtype=torch.bool)

    pixel_values, num_tiles = embedder._preprocess_images(images)
    print("pixel_values:", brief(pixel_values), "num_tiles", num_tiles)

    vit = embedder.model.extract_feature(pixel_values)
    print("vit_embeds:", brief(vit))

    prompt = embedder._build_multimodal_prompt(num_tiles, "pick up the block")
    print(f"prompt 字符={len(prompt)}  <IMG_CONTEXT>={prompt.count('<IMG_CONTEXT>')}")

    embeds, attn = embedder._prepare_and_fuse_embeddings(prompt, vit, mask, num_tiles)
    print("input_embeds:", brief(embeds))
    print(f"attention_mask: shape={tuple(attn.shape)} 有效token={int(attn.sum())}")

    hidden = embedder.model.language_model(
        inputs_embeds=embeds,
        attention_mask=attn,
        output_hidden_states=True,
        return_dict=True,
    ).hidden_states[-1].float()
    print("最后一层:", brief(hidden))
    print("返回的 cls:", brief(hidden[:, 0]))


if __name__ == "__main__":
    args = sys.argv[1:]
    images = load_images([a for a in args if a != "--model"])
    out = os.path.join(os.path.dirname(__file__), "trace_internvl3_embedder.png")
    save_sheet(walk_images(images), out)
    if "--model" in args:
        walk_model(images)
    else:
        print("没加 --model，视觉塔和语言模型没跑。")
