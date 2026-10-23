# Inisialisasi package src PUSDIKZI RAG

# Mitigasi crash level-zero Intel GPU driver pada sistem tanpa hardware aktif XPU
try:
    import torch
    if hasattr(torch, "xpu"):
        torch.xpu.is_available = lambda: False
except Exception:
    pass

try:
    import transformers.utils.import_utils
    transformers.utils.import_utils.is_torch_xpu_available = lambda check_device=False: False
except Exception:
    pass
