#!/usr/bin/env python3
"""Validate spec/examples/*.json against spec/map-spec.schema.json.

Usage: python3 spec/validate_example.py
Exit 0 when every example validates, 1 otherwise.
"""
import json
import os
import sys

try:
    import jsonschema
except ImportError:
    sys.exit("jsonschema not installed: pip install -r requirements.txt")

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "map-spec.schema.json"), "r", encoding="utf-8") as f:
    schema = json.load(f)

ok = True
exdir = os.path.join(HERE, "examples")
for fn in sorted(os.listdir(exdir)):
    if not fn.endswith(".json"):
        continue
    with open(os.path.join(exdir, fn), "r", encoding="utf-8") as f:
        spec = json.load(f)
    try:
        jsonschema.validate(instance=spec, schema=schema)
        print("OK   " + fn)
    except Exception as e:
        ok = False
        print("FAIL " + fn + ": " + str(getattr(e, "message", e)))
sys.exit(0 if ok else 1)
