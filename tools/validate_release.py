"""Dependency-free release integrity and runtime compatibility checks."""
from pathlib import Path
import ast
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]
def main():
    errors = []
    manifest = json.loads((ROOT / "SHA256SUMS.json").read_text())
    for name, expected in manifest.items():
        p = ROOT / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            errors.append("Release checksum mismatch: " + name)
    for p in (ROOT / "infantry_rl").glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
    for p in (ROOT / "pretrained").glob("*/run.json"):
        meta = json.loads(p.read_text())
        for name, expected in meta["source_hashes"].items():
            if name.endswith(("_train.py", "_audit.py", "_play.py")):
                continue
            q = ROOT / "infantry_rl" / name
            if not q.exists() or hashlib.sha256(q.read_bytes()).hexdigest() != expected:
                errors.append(p.parent.name + ": runtime mismatch " + name)
        for key, filename in [("model_sha256", "training_scene_fix.xml"), ("physics_model_sha256", "training_scene_fix_gas_trial.xml")]:
            if key in meta:
                q = ROOT / "infantry_rl/robot/xmls" / filename
                if hashlib.sha256(q.read_bytes()).hexdigest() != meta[key]:
                    errors.append(p.parent.name + ": model mismatch")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(manifest)} published files; Python syntax; checkpoint runtime/model contracts")
if __name__ == "__main__":
    main()
