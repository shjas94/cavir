# nccl_test.py
import os, torch, torch.distributed as dist, time
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(0)
for numel in [1<<20, 1<<22, 1<<23, 12582912, 1<<24]:
    x = torch.ones(numel, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(numel // w, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize(); t0 = time.time()
    dist.reduce_scatter_tensor(out, x)
    torch.cuda.synchronize()
    if r == 0:
        print(f"numel={numel} {(time.time()-t0)*1000:.1f} ms")
dist.destroy_process_group()