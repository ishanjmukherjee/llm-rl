#!/usr/bin/env python3
# wandb_dump.py
import glob
import sys, json
import typer
from google.protobuf.json_format import MessageToDict
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

def msg_to_dict(msg):
    # Support older/newer google.protobuf versions
    try:
        return MessageToDict(
            msg,
            preserving_proto_field_name=True,
            including_default_value_fields=False,
        )
    except TypeError:
        # Older versions do not support including_default_value_fields
        return MessageToDict(
            msg,
            preserving_proto_field_name=True,
        )

def decode_history_dict(hd):
    """
    hd is dict version of rec.history, typically:
      { "item": [ {"key": "...", "value_json": "..."} , ... ] }
    but older schemas may nest under "value": {"json_value": "..."} or numeric.
    """
    out = {}
    items = hd.get("item", [])
    for it in items:
        k = it.get("key")
        if not k:
            continue
        if "value_json" in it:
            try:
                out[k] = json.loads(it["value_json"])
            except Exception:
                out[k] = it["value_json"]
        elif "value" in it:
            v = it["value"]
            # Try json_value first
            if "json_value" in v:
                try:
                    out[k] = json.loads(v["json_value"])
                except Exception:
                    out[k] = v["json_value"]
            # Fallbacks for numeric/string/bool
            elif "num" in v:
                out[k] = v["num"]
            elif "str" in v:
                out[k] = v["str"]
            elif "bool" in v:
                out[k] = v["bool"]
            else:
                out[k] = v
        else:
            # Last resort: keep whatever is there
            out[k] = {kk: vv for kk, vv in it.items() if kk != "key"}
    return out

def decode_kv_update(msg_dict):
    """
    Handles config/summary updates across schema shifts.
    Expect shapes like:
      {"update":{"item":[{"key":"...", "value_json":"..."}]}}
    or
      {"update":{"item":[{"key":"...","value":{"json_value":"..."}}]}}
    """
    def collect_items(container):
        collected = []
        if isinstance(container, dict):
            if "item" in container:
                raw = container.get("item", [])
                if isinstance(raw, dict):
                    collected.append(raw)
                elif isinstance(raw, list):
                    collected.extend(raw)
            # Some versions put items at the top level without an "update" key
            elif "key" in container and ("value" in container or "value_json" in container):
                collected.append(container)
        elif isinstance(container, list):
            for element in container:
                collected.extend(collect_items(element))
        return collected

    items = []
    if isinstance(msg_dict, dict):
        if "update" in msg_dict:
            items.extend(collect_items(msg_dict["update"]))
        # Fallback: sometimes items are at top-level
        items.extend(collect_items(msg_dict))
    elif isinstance(msg_dict, list):
        items.extend(collect_items(msg_dict))
    out = {}
    for it in items:
        k = it.get("key")
        if not k:
            continue
        if "value_json" in it:
            try:
                out[k] = json.loads(it["value_json"])
            except Exception:
                out[k] = it["value_json"]
        elif "value" in it:
            v = it["value"]
            if "json_value" in v:
                try:
                    out[k] = json.loads(v["json_value"])
                except Exception:
                    out[k] = v["json_value"]
            elif "num" in v:
                out[k] = v["num"]
            elif "str" in v:
                out[k] = v["str"]
            elif "bool" in v:
                out[k] = v["bool"]
            else:
                out[k] = v
    return out

def extract_output_line(out_dict):
    # Try the common possibilities across versions
    for key in ("stdout_line", "stderr_line", "log_line", "line", "data", "text"):
        val = out_dict.get(key)
        if val:
            return val
    return None

def extract_output_line_msg(msg):
    # Prefer reading directly from the proto to avoid base64 from MessageToDict
    for attr in ("stdout_line", "stderr_line", "log_line", "line", "text", "data"):
        if hasattr(msg, attr):
            val = getattr(msg, attr)
            if val:
                if isinstance(val, (bytes, bytearray)):
                    try:
                        return val.decode("utf-8", errors="replace")
                    except Exception:
                        return None
                if isinstance(val, str):
                    return val
    return None

app = typer.Typer(add_completion=False)

path = glob.glob("wandb/latest-run/*.wandb")[0]

@app.command()
def dump():
    ds = datastore.DataStore()
    ds.open_for_scan(path)

    def next_record_tuple():
        # Prefer scan_data to avoid CRC bookkeeping issues in some SDK builds
        if hasattr(ds, "scan_data"):
            try:
                data = ds.scan_data()
            except Exception:
                return None
            return None if data is None else (None, data)
        if hasattr(ds, "scan_record"):
            try:
                t = ds.scan_record()
            except Exception:
                return None
            return t            # (offset:int, data:bytes) or None
        raise RuntimeError("Unsupported wandb SDK version")

    while True:
        t = next_record_tuple()
        if t is None:
            break
        _, raw = t
        rec = pb.Record()
        try:
            rec.ParseFromString(raw)
        except Exception:
            # Tolerate decode errors (e.g., truncated/corrupt trailing bytes)
            continue
        kind = rec.WhichOneof("record_type")

        with open('wandb/latest-run/log.txt', 'a') as f:
            # Output: print lines as-is (stdout+stderr)
            if kind == "output":
                line = extract_output_line_msg(rec.output)
                if line is None:
                    out_d = msg_to_dict(rec.output)
                    line = extract_output_line(out_d)
                if line is not None:
                    if line and not line.endswith("\n"):
                        line += "\n"
                    f.write(line)
                continue

            # Output v2: output_raw
            if kind == "output_raw":
                line = extract_output_line_msg(rec.output_raw)
                if line is None:
                    out_d = msg_to_dict(rec.output_raw)
                    line = extract_output_line(out_d)
                if line is not None:
                    # Ensure trailing newline (W&B lines usually include it already)
                    if line and not line.endswith("\n"):
                        line += "\n"
                    f.write(line)
                continue

            # Skip JSON outputs (history/config/summary) for plain text mode
            if kind in ("history", "config", "summary"):
                continue

    # Optional: files, telemetry, system stats, etc. (ignored by default)

if __name__ == "__main__":
    app()