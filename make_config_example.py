# make_config_example.py
# 用法: python make_config_example.py config.json config.example.json
import json, sys, re
from copy import deepcopy

IN = sys.argv[1] if len(sys.argv) > 1 else "config.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "config.example.json"

# 认为是“敏感信息”的键名片段（大小写不敏感）
SECRET_KEYS = re.compile(
    r"(api[_-]?key|apikey|bearer|token|secret|password|passphrase|"
    r"private[_-]?key|mnemonic|seed|rpc[_-]?url|endpoint|webhook|auth|cookie)",
    re.I
)

PLACEHOLDER_FOR = {
    "rpc_url": "<YOUR_RPC_URL>",
    "api_key": "<YOUR_API_KEY>",
    "private_key": "<YOUR_PRIVATE_KEY>",
    "secret": "<YOUR_SECRET>",
    "token": "<YOUR_TOKEN>",
    "endpoint": "<YOUR_ENDPOINT>",
    "webhook": "<YOUR_WEBHOOK>",
}

def scrub(obj, parent_key=None):
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if SECRET_KEYS.search(k):
                # 为常见键给出更友好的占位符
                key_norm = k.lower().replace("-", "_")
                ph = None
                for known, tmpl in PLACEHOLDER_FOR.items():
                    if known in key_norm:
                        ph = tmpl
                        break
                if ph is None:
                    ph = f"<FILL_{k.upper()}>"
                new[k] = ph
            else:
                new[k] = scrub(v, k)
        return new
    elif isinstance(obj, list):
        return [scrub(x, parent_key) for x in obj]
    else:
        return obj

with open(IN, "r", encoding="utf-8") as f:
    cfg = json.load(f)

example = scrub(deepcopy(cfg))

# 可选：提示性修改，比如确保没有真实私钥/令牌残留
# （已经通过 scrub 处理过，通常不需要额外操作）

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(example, f, ensure_ascii=False, indent=2)

print(f"Created {OUT} from {IN}. (secrets replaced with placeholders)")
