import argparse
import os
import time
import traceback

import numpy as np
import open3d as o3d

try:
    import cadquery as cq
    import yaml

    import torch
    from torch import nn
    from transformers import (
        AutoTokenizer, Qwen2ForCausalLM, Qwen2Model, PreTrainedModel)
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from pytorch3d.ops import sample_farthest_points
    from huggingface_hub import try_to_load_from_cache

    class FourierPointEncoder(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
            self.register_buffer('frequencies', frequencies, persistent=False)
            self.projection = nn.Linear(51, hidden_size)

        def forward(self, points):
            x = points
            x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
            x = torch.cat((points, x.sin(), x.cos()), dim=-1)
            x = self.projection(x)
            return x

    class CADRecode(Qwen2ForCausalLM):
        def __init__(self, config):
            PreTrainedModel.__init__(self, config)
            self.model = Qwen2Model(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            torch.set_default_dtype(torch.float32)
            self.point_encoder = FourierPointEncoder(config.hidden_size)
            torch.set_default_dtype(torch.bfloat16)

        def forward(self,
                    input_ids=None,
                    attention_mask=None,
                    point_cloud=None,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=None,
                    labels=None,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,
                    cache_position=None):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if past_key_values is None or past_key_values.get_seq_length() == 0:
                assert inputs_embeds is None
                inputs_embeds = self.model.embed_tokens(input_ids)
                point_embeds = self.point_encoder(point_cloud).bfloat16()
                inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
                attention_mask[attention_mask == -1] = 1
                input_ids = None
                position_ids = None

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position)

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
            model_inputs['point_cloud'] = kwargs['point_cloud']
            return model_inputs

    _GEN_DEPS_OK = True
    _GEN_IMPORT_ERR = None
except ImportError as _e:
    _GEN_DEPS_OK = False
    _GEN_IMPORT_ERR = _e


def run_name_for(input_path):
    stem = os.path.splitext(os.path.basename(input_path))[0]
    if stem.isdigit():
        parent = os.path.basename(os.path.dirname(os.path.abspath(input_path)))
        return f"{parent}_{stem}"
    return stem


def normalize_for_cadrecode(pts):
    pts_min = pts.min(axis=0)
    pts_max = pts.max(axis=0)
    center = (pts_min + pts_max) / 2.0
    extent = float((pts_max - pts_min).max())
    return ((pts - center) * (2.0 / extent)).astype(np.float32)


def fps_downsample(pts, n=256):
    pts_t = torch.tensor(pts).unsqueeze(0)
    _, ids = sample_farthest_points(pts_t, K=n, random_start_point=True)
    return pts[ids[0].numpy()].astype(np.float32)


def visualize_clouds(pts_full, pts_fps):
    W, H, top = 600, 600, 50
    point_size = 3.0

    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(pts_full)
    pcd_full.paint_uniform_color([0.2, 0.5, 0.8])

    pcd_fps = o3d.geometry.PointCloud()
    pcd_fps.points = o3d.utility.Vector3dVector(pts_fps)
    pcd_fps.paint_uniform_color([0.9, 0.3, 0.2])

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window("Input cloud (full, normalized)", width=W, height=H, left=0, top=top)
    vis1.add_geometry(pcd_full)
    vis1.get_render_option().point_size = point_size

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window("FPS 256 (normalized)", width=W, height=H, left=W, top=top)
    vis2.add_geometry(pcd_fps)
    vis2.get_render_option().point_size = point_size

    vises = [vis1, vis2]
    running = [True] * len(vises)
    while all(running):
        for k, vis in enumerate(vises):
            if running[k]:
                running[k] = vis.poll_events()
                vis.update_renderer()
        time.sleep(0.01)
    for v in vises:
        v.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="sample_clouds/abc_00949/0.xyzc",
                        help="Point cloud to reconstruct: any whitespace-separated "
                             "text file whose first three columns are x y z. A "
                             "segmentation column, if present, is ignored.")
    parser.add_argument("--visualize", action="store_true",
                        help="Skip generation; load saved input_full.ply / "
                             "input_fps256.ply and show side-by-side. Takes the "
                             "same --input_path as the generation run. "
                             "Needs a GUI build of Open3D, which the cadrecode image does not have.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for FPS random-start. Different seeds give "
                             "different 256-point subsets and thus different generations.")
    args = parser.parse_args()

    OUT_DIR = os.path.join("output_cadrecode", run_name_for(args.input_path))

    if args.visualize:
        full_path = os.path.join(OUT_DIR, "input_full.ply")
        fps_path = os.path.join(OUT_DIR, "input_fps256.ply")
        if not (os.path.exists(full_path) and os.path.exists(fps_path)):
            raise FileNotFoundError(
                f"Need {full_path} and {fps_path}. Run generation in Docker first.")
        full_pcd = o3d.io.read_point_cloud(full_path)
        fps_pcd = o3d.io.read_point_cloud(fps_path)
        visualize_clouds(np.asarray(full_pcd.points), np.asarray(fps_pcd.points))
    else:
        if not _GEN_DEPS_OK:
            raise RuntimeError(
                f"Generation deps missing ({_GEN_IMPORT_ERR}). "
                f"Run inside the cadrecode Docker container, or use --visualize.")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        with open("secrets.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            HF_TOKEN = data["HF_TOKEN"]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        attn_implementation = 'flash_attention_2' if torch.cuda.is_available() else None

        for repo, probe in (('Qwen/Qwen2-1.5B', 'tokenizer_config.json'),
                            ('filapro/cad-recode-v1.5', 'config.json')):
            cached = try_to_load_from_cache(repo, probe)
            if isinstance(cached, str):
                print(f"[cad-recode] cache hit: {repo}")
            else:
                print(f"[cad-recode] cache miss: {repo}, downloading from the Hugging Face Hub")

        tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2-1.5B',
            pad_token='<|im_end|>',
            padding_side='left',
            token=HF_TOKEN)

        model = CADRecode.from_pretrained(
            'filapro/cad-recode-v1.5',
            torch_dtype='auto',
            attn_implementation=attn_implementation,
            token=HF_TOKEN).eval().to(device)

        INPUT_PATH = args.input_path
        if os.path.isdir(OUT_DIR):
            print(f"[cad-recode] {OUT_DIR} already exists, overwriting its contents")
        os.makedirs(OUT_DIR, exist_ok=True)

        pts_full_world = np.loadtxt(INPUT_PATH).astype(np.float32)[:, :3]
        np.save(os.path.join(OUT_DIR, "input_full.npy"), pts_full_world)
        print(f"[cad-recode] full point cloud: N={len(pts_full_world)}")

        pts_full_norm = normalize_for_cadrecode(pts_full_world)
        pts_fps_norm = fps_downsample(pts_full_norm, 256)
        np.save(os.path.join(OUT_DIR, "input_fps256.npy"), pts_fps_norm)
        print(f"[cad-recode] fps (farthest point sampling): 256  range=[{pts_fps_norm.min():.3f}, {pts_fps_norm.max():.3f}]")

        full_pcd_save = o3d.geometry.PointCloud()
        full_pcd_save.points = o3d.utility.Vector3dVector(pts_full_norm)
        o3d.io.write_point_cloud(os.path.join(OUT_DIR, "input_full.ply"), full_pcd_save)
        fps_pcd_save = o3d.geometry.PointCloud()
        fps_pcd_save.points = o3d.utility.Vector3dVector(pts_fps_norm)
        o3d.io.write_point_cloud(os.path.join(OUT_DIR, "input_fps256.ply"), fps_pcd_save)
        print("[cad-recode] saved input_full.ply + input_fps256.ply (view with --visualize)")

        start_token_id = tokenizer('<|im_start|>')['input_ids'][0]
        input_ids = [tokenizer.pad_token_id] * 256 + [start_token_id]
        attention_mask = [-1] * 256 + [1]

        print("[cad-recode] processing the input point cloud...")
        with torch.no_grad():
            out = model.generate(
                input_ids=torch.tensor([input_ids], dtype=torch.long, device=device),
                attention_mask=torch.tensor([attention_mask], dtype=torch.long, device=device),
                point_cloud=torch.tensor(pts_fps_norm, dtype=torch.float32, device=device).unsqueeze(0),
                max_new_tokens=768,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = out[0, len(input_ids):]
        py_source = tokenizer.decode(generated_ids, skip_special_tokens=True)

        py_path = os.path.join(OUT_DIR, "generated.py")
        with open(py_path, "w") as f:
            f.write(py_source)
        print(f"[cad-recode] generated tokens: {len(generated_ids)}  source length: {len(py_source)} chars")
        print(f"[cad-recode] saved generated source: {py_path}")

        try:
            exec(py_source, globals())
            r = globals().get("r")
            if r is None:
                raise RuntimeError("generated code did not bind a Workplane to `r`")
            step_path = os.path.join(OUT_DIR, "generated.step")
            cq.exporters.export(r, step_path)
            print(f"[cad-recode] outcome: STEP exported -> {step_path}")
        except Exception:
            err_path = os.path.join(OUT_DIR, "compile_error.txt")
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
            print(f"[cad-recode] outcome: compile FAILED -> {err_path}")
