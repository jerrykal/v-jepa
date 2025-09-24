#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fix_typo_keys.py
Rename all checkpoint keys containing 'attenion' -> 'attention'.

Examples
--------
# 乾跑（不輸出檔案，只列出將會改的鍵）
python fix_typo_keys.py -i /path/to/ckpt.pth.tar --dry-run --limit 20

# 輸出新檔（自動命名：ckpt.fixed.pth.tar）
python fix_typo_keys.py -i /path/to/ckpt.pth.tar

# 指定輸出檔
python fix_typo_keys.py -i /path/to/ckpt.pth.tar -o /path/to/ckpt.attention.pth.tar

# 若真的有 key 碰撞，要選擇覆蓋
python fix_typo_keys.py -i in.pth.tar -o out.pth.tar --allow-overwrite
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import torch


MATCH = "attenion"
REPLACE = "attention"


def _rename_container(obj: Any,
                      match: str,
                      replace: str,
                      renamed: List[Tuple[str, str]],
                      allow_overwrite: bool) -> Any:
    """
    Recursively rename dict keys containing `match` -> `replace`.
    Works for nested dict/list/tuple. Non-container objects are returned as-is.
    """
    if isinstance(obj, dict):
        new_d: Dict[Any, Any] = {}
        for k, v in obj.items():
            new_k = k
            if isinstance(k, str) and match in k:
                candidate = k.replace(match, replace)
                # collision check
                if (candidate in new_d) and (candidate != k) and not allow_overwrite:
                    raise RuntimeError(
                        f"Key collision after rename: '{k}' -> '{candidate}', "
                        f"but '{candidate}' already exists in this dict. "
                        f"Run again with --allow-overwrite if you are sure."
                    )
                new_k = candidate
                renamed.append((k, new_k))
            # recurse
            new_d[new_k] = _rename_container(v, match, replace, renamed, allow_overwrite)
        return new_d
    elif isinstance(obj, list):
        return [_rename_container(x, match, replace, renamed, allow_overwrite) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_rename_container(x, match, replace, renamed, allow_overwrite) for x in obj)
    else:
        return obj


def _auto_out_path(in_path: str) -> str:
    base, ext = os.path.splitext(in_path)
    # handle double ext like .pth.tar
    if ext == ".tar":
        base2, ext2 = os.path.splitext(base)
        if ext2 in (".pth", ".pt"):
            return f"{base2}.fixed{ext2}{ext}"
    return f"{base}.fixed{ext}"


def main():
    ap = argparse.ArgumentParser(description=f"Rename checkpoint keys: '{MATCH}' -> '{REPLACE}'")
    ap.add_argument("-i", "--input", required=True, help="Input checkpoint path (.pth/.pth.tar)")
    ap.add_argument("-o", "--output", default=None, help="Output path (default: auto add .fixed)")
    ap.add_argument("--match", default=MATCH, help="Substring to match (default: attenion)")
    ap.add_argument("--replace", default=REPLACE, help="Replacement (default: attention)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write file; just report changes")
    ap.add_argument("--limit", type=int, default=50, help="Max number of rename pairs to print")
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="Allow overwriting when a renamed key collides with an existing key")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output or _auto_out_path(in_path)

    if not os.path.exists(in_path):
        print(f"[ERR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading: {in_path}")
    ckpt = torch.load(in_path, map_location="cpu")

    renamed: List[Tuple[str, str]] = []
    ckpt_new = _rename_container(
        ckpt, match=args.match, replace=args.replace,
        renamed=renamed, allow_overwrite=args.allow_overwrite
    )

    if not renamed:
        print("[INFO] No keys contained the match substring; nothing to change.")
        if not args.dry_run and in_path != out_path:
            # still allow writing identical copy if user insisted
            torch.save(ckpt_new, out_path)
            print(f"[INFO] Wrote (identical) checkpoint to: {out_path}")
        return

    print(f"[INFO] Renamed {len(renamed)} keys (showing up to {args.limit}):")
    for old, new in renamed[: args.limit]:
        print(f"  - {old}  -->  {new}")
    if len(renamed) > args.limit:
        print(f"  ... (+{len(renamed) - args.limit} more)")

    if args.dry_run:
        print("[INFO] Dry-run complete. No file written.")
        return

    # ensure output dir
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    torch.save(ckpt_new, out_path)
    print(f"[OK] Wrote fixed checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
