"""fp32 vs bf16-autocast: loss/gradient agreement and speed, on synthetic batches (no data loaded)."""
import json, os, sys, time
os.environ["SCDFM_SDPA"] = "1"
import numpy as np, torch
import vcc.train as T
if os.environ.get("TF32"):
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
F = sys.argv[1]; B = int(sys.argv[2]); cond = sys.argv[3] if len(sys.argv) > 3 else "token"
vocab = json.load(open(f"{F}/vocab.json")); G = [g.strip() for g in open(f"{F}/genes.txt")][:1000]
gt = torch.tensor([vocab[g] for g in G]).cuda().unsqueeze(0).repeat(B, 1)
torch.manual_seed(0)
vf = T.build_model(F, vocab, 512, cond).cuda().train()
g = torch.Generator(device="cuda").manual_seed(1)
src = torch.log1p(torch.poisson(torch.rand(B, 1000, device="cuda", generator=g) * 0.4)).float()
tgt = torch.log1p(torch.poisson(torch.rand(B, 1000, device="cuda", generator=g) * 0.45)).float()
c = (torch.randn(B, 1000, device="cuda", generator=g) * 0.3) if cond == "effect" else torch.tensor([[vocab[G[0]], vocab["control"]]]).cuda().repeat(B, 1)
GAMMA = float(os.environ.get("GAMMA", 0.5))
def run(dtype, seed=5):
    T.AMP["dtype"] = dtype; torch.manual_seed(seed); vf.zero_grad(set_to_none=True)
    loss = T.loss_fn(vf, src, tgt, gt, c, GAMMA); loss.backward()
    return loss.item(), torch.cat([p.grad.flatten() for p in vf.parameters() if p.grad is not None]).float()
for seed in (5, 6, 7):
    a, ga = run(None, seed); b, gb = run(torch.bfloat16, seed)
    print(f"[equiv-seed{seed}] gamma={GAMMA} loss {a:.4f} vs {b:.4f}; grad cosine {torch.nn.functional.cosine_similarity(ga, gb, dim=0).item():.4f}")
l32, g32 = run(None); l16, g16 = run(torch.bfloat16)
cos = torch.nn.functional.cosine_similarity(g32, g16, dim=0).item()
print(f"[equiv] B={B} loss fp32 {l32:.5f} bf16 {l16:.5f} (rel {abs(l16-l32)/l32:.2e}); grad cosine {cos:.5f}; |g| ratio {g16.norm()/g32.norm():.4f}")
if os.environ.get("SKIPSPEED"): sys.exit()
for name, dt in (("fp32", None), ("bf16", torch.bfloat16)):
    run(dt); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    for i in range(6): run(dt, i)
    torch.cuda.synchronize(); print(f"[speed] {name}: {(time.time()-t)/6:.3f} s/step, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
