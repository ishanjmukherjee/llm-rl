#!/usr/bin/env python3
import os
import sys
import time
import json
import glob
import logging
import base64
import mimetypes
import subprocess
from typing import List, Optional, Tuple

from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from twilio.twiml.voice_response import VoiceResponse, Gather

POLL_INTERVAL_SECS = 600  # 10 minutes
BASE_WANDB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "wandb"))
MAX_COMPLETION_ROWS = 5

# Twilio configuration
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")
TWILIO_TO_NUMBER = os.environ.get("TWILIO_TO_NUMBER")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

_model_cache = {"model": None, "tokenizer": None}
_vision_cache = {"model": None, "tokenizer": None, "model_name": None}

# Default local VLM to use when --use_local_vision is set
DEFAULT_VISION_MODEL = "unsloth/Llama-3.2-11B-Vision-Instruct"
SMALL_MODEL_NAME = "unsloth/gemma-2-9b-bnb-4bit"


def make_emergency_call(analysis: str, run_dir_name: str) -> bool:
    """Place an emergency call via Twilio with a synthesized summary."""
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER]):
        logging.error("Twilio credentials not configured. Please set environment variables.")
        return False

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        response = VoiceResponse()
        response.say(
            "Alert! Alert! Critical error detected in your reinforcement learning training run.",
            voice="alice",
            language="en-US",
        )
        response.pause(length=1)
        response.say(
            f"The model supervisor has detected an error condition that requires immediate attention. Run directory: "
            f"{run_dir_name.split('/')[-1] if '/' in run_dir_name else run_dir_name}",
            voice="alice",
            language="en-US",
        )
        response.pause(length=1)
        error_summary = analysis[:100] if analysis else ""
        response.say(f"Error summary: {error_summary}", voice="alice", language="en-US")
        response.pause(length=1)
        gather = Gather(num_digits=1, action="/handle-key", method="POST")
        gather.say("Press 1 to acknowledge this alert, or press 2 to repeat this message.", voice="alice")
        response.append(gather)
        response.say("No input received. The message will now repeat.", voice="alice")
        response.redirect("/")

        call = client.calls.create(
            to=TWILIO_TO_NUMBER,
            from_=TWILIO_FROM_NUMBER,
            twiml=str(response),
            status_callback=f"https://your-webhook-url.com/call-status?run={run_dir_name}",
            status_callback_method="POST",
        )
        logging.info(f"Emergency call initiated. Call SID: {call.sid}")
        return True
    except TwilioRestException as e:
        logging.error(f"Failed to make emergency call: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error making emergency call: {e}")
        return False


def wake_up_user(analysis: str, run_dir_name: str, context_paths: List[str]) -> None:
    logging.warning("INITIATING EMERGENCY USER NOTIFICATION - CALLING USER!")
    if not make_emergency_call(analysis, run_dir_name):
        logging.critical("Failed to call user! Manual intervention required.")
    else:
        logging.info("User notified via phone call.")


def get_run_files_dir(run_dir_name: str, base_dir: str = BASE_WANDB_DIR) -> Optional[str]:
    files_dir = os.path.join(base_dir, run_dir_name, "files")
    if not os.path.isdir(files_dir):
        logging.warning("Run files directory not found: %s", files_dir)
        return None
    return files_dir


def get_latest_completions_table(files_dir: str) -> Optional[str]:
    pattern = os.path.join(files_dir, "media", "table", "completions_*.table.json")
    paths = glob.glob(pattern)
    if not paths:
        logging.info("No completions tables found under %s", os.path.dirname(pattern))
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def extract_first_five_prompt_completion(table_json_path: str, max_rows: int = MAX_COMPLETION_ROWS) -> List[Tuple[str, str]]:
    try:
        with open(table_json_path, "r") as f:
            obj = json.load(f)
        columns = obj.get("columns") or obj.get("cols") or []
        data = obj.get("data") or []
        lowered = [str(c).lower() for c in columns]
        try:
            prompt_idx = lowered.index("prompt")
        except ValueError:
            prompt_idx = None
        completion_idx = None
        for name in ["completion", "generated", "output", "answer", "text"]:
            if name in lowered:
                completion_idx = lowered.index(name)
                break
        pairs: List[Tuple[str, str]] = []
        for row in data[:max_rows]:
            prompt_txt = str(row[prompt_idx])[:1000] if (prompt_idx is not None and prompt_idx < len(row)) else (
                str(row[0])[:500] if row else ""
            )
            compl_txt = str(row[completion_idx])[:1000] if (
                completion_idx is not None and completion_idx < len(row)
            ) else " | ".join([str(c)[:400] for c in row[:2]])
            pairs.append((prompt_txt, compl_txt))
        return pairs
    except Exception:
        logging.exception("Failed to parse table %s", table_json_path)
        return []


def get_latest_graph_image(files_dir: str) -> Optional[str]:
    graphs_dir = "/home/ubuntu/llm-rl/wandb/latest-run/plots"
    if not os.path.isdir(graphs_dir):
        logging.error("Graphs directory not found: %s", graphs_dir)
        return None
    imgs = glob.glob(os.path.join(graphs_dir, "*"))
    if not imgs:
        logging.error("No images found in %s", graphs_dir)
        return None
    imgs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return imgs[0]


def build_prompt(graph_image_path: Optional[str], prompt_completion_pairs: List[Tuple[str, str]]) -> str:
    header = (
        "You are supervising a training run. Review the most recent graph image (path) and the first five prompt+completion rows from the latest completions table.\n"
        "Decide whether something is wrong with the training (e.g., incoherent completions, obvious corruption, degenerate outputs).\n\n"
        "Output requirements (follow exactly):\n"
        "- Include an Explanation: <1-3 concise sentences explaining your decision, referencing the inputs>.\n"
        "- If you detect a real issue, include the exact phrase: ERROR FOUND (anywhere in your reply).\n"
        "- If everything looks fine, include the exact phrase: ALL IS GOOD (anywhere in your reply).\n"
        "- Include exactly one of these phrases, not both.\n"
        " - it is okay if the completions cut off early, as long as the content is coherent."
        "Please output a confidence score in your decision between 0 and 100."
    )
    parts: List[str] = [header]
    if graph_image_path:
        parts.append("Most recent graph image path:")
        parts.append(graph_image_path)
    if prompt_completion_pairs:
        parts.append("First five prompt+completion pairs (truncated):")
        for i, (p, c) in enumerate(prompt_completion_pairs, 1):
            parts.append(f"[{i}] Prompt: {p}")
            parts.append(f"    Completion: {c}")
    return "\n\n".join(parts)


def small_model_infer(prompt: str) -> str:
    """Call local Unsloth small text model for inference. Returns decoded text."""
    try:
        from unsloth import FastLanguageModel, FastModel
        import torch
    except Exception as e:
        logging.error("Unsloth is not installed or failed to import: %s", e)
        return "Model not available. OK"

    if _model_cache["model"] is None:
        logging.info("Loading Unsloth small model (4bit) once...")
        model, tokenizer = FastModel.from_pretrained(
            model_name=SMALL_MODEL_NAME,
            max_seq_length=2048,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            pass
        _model_cache["model"] = model
        _model_cache["tokenizer"] = tokenizer

    model = _model_cache["model"]
    tokenizer = _model_cache["tokenizer"]

    inputs = tokenizer([prompt], return_tensors="pt")
    try:
        import torch
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model.cuda()
    except Exception:
        pass

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
        )
    gen_ids = output_ids[0][len(inputs["input_ids"][0]):]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def small_model_infer_local_vision(prompt: str, image_path: str, vision_model_name: str) -> str:
    """Call a local Unsloth VLM (FastVisionModel) with an image + text prompt."""
    try:
        from unsloth import FastVisionModel
        import torch
        from PIL import Image
    except Exception as e:
        logging.error("Vision dependencies not available (unsloth/transformers/PIL): %s", e)
        return "Model not available. OK"

    if _vision_cache["model"] is None or _vision_cache["model_name"] != vision_model_name:
        logging.info("Loading Unsloth vision model %s (4bit) once...", vision_model_name)
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=vision_model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )
        try:
            FastVisionModel.for_inference(model)
        except Exception:
            pass
        _vision_cache["model"] = model
        _vision_cache["tokenizer"] = tokenizer
        _vision_cache["model_name"] = vision_model_name

    model = _vision_cache["model"]
    tokenizer = _vision_cache["tokenizer"]

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        logging.exception("Failed to open image %s", image_path)
        return ""

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}
    ]
    try:
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        try:
            import torch
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                model.cuda()
        except Exception:
            pass
        with torch.no_grad():
            # Disable cache here to avoid static cache errors on some VLMs
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                use_cache=False,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                min_p=0.1,
            )
        gen_ids = output_ids[0]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if input_text in decoded:
            idx = decoded.find(input_text) + len(input_text)
            return decoded[idx:].strip()
        return decoded.strip()
    except Exception:
        logging.exception("Vision model inference failed")
        return ""


def small_model_infer_remote(prompt: str, image_paths: Optional[List[str]] = None) -> str:
    """Call Anthropic Claude (multimodal) remotely, passing Base64-encoded images plus the same prompt."""
    try:
        import anthropic
    except Exception as e:
        logging.error("Anthropic SDK not available: %s", e)
        return "Model not available. OK"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logging.error("ANTHROPIC_API_KEY is not set; cannot call Claude")
        return "Model not available. OK"

    client = anthropic.Anthropic(api_key=api_key)

    content_blocks: List[dict] = []
    for img_path in (image_paths or []):
        if not img_path or not os.path.isfile(img_path):
            continue
        mime, _ = mimetypes.guess_type(img_path)
        if not mime:
            mime = "image/png"
        try:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64,
                    },
                }
            )
        except Exception:
            logging.exception("Failed to encode image %s", img_path)

    content_blocks.append({"type": "text", "text": prompt})

    try:
        resp = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=400,
            messages=[{"role": "user", "content": content_blocks}],
        )
        if hasattr(resp, "content") and resp.content:
            for blk in resp.content:
                if getattr(blk, "type", None) == "text" and hasattr(blk, "text"):
                    return (blk.text or "").strip()
            return str(resp.content)
        return ""
    except Exception:
        logging.exception("Claude API call failed")
        return ""


def escalate_to_claude(summary: str, context_paths: List[str]) -> None:
    """Execute external Claude code script; pass context via environment variables."""
    try:
        script_path = "/home/ubuntu/llm-rl/proctor/big_baddie/claude_code.sh"
        if not os.path.isfile(script_path):
            logging.error(f"Claude script not found: {script_path}")
            return
        # Ensure it's executable (not strictly required when calling via bash)
        try:
            st = os.stat(script_path)
            if not (st.st_mode & 0o111):
                os.chmod(script_path, st.st_mode | 0o111)
        except Exception:
            logging.exception("Failed to set execute permission on Claude script")
        env = os.environ.copy()
        env["CLAUDE_SUMMARY"] = (summary or "")[:4000]
        env["CLAUDE_CONTEXT"] = "\n".join(context_paths[:50])
        # Call explicitly via bash and capture output
        proc = subprocess.run(
            ["bash", script_path],
            cwd=os.path.dirname(script_path),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.stdout:
            logging.info(f"Claude script stdout:\n{proc.stdout}")
        if proc.stderr:
            logging.warning(f"Claude script stderr:\n{proc.stderr}")
        if proc.returncode != 0:
            logging.error(f"Claude script exited with code {proc.returncode}")
        else:
            logging.info("Claude script executed successfully.")
    except Exception:
        logging.exception("Failed to execute Claude script")


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python proctor/main.py <wandb_run_dir_name> [--use_remote] [--use_local_vision]"
        )
        print(
            "Example: python proctor/main.py /home/ubuntu/llm-rl/wandb/run-20250810_225813-cqhfwfvj --use_local_vision"
        )
        return 1

    run_dir_name: Optional[str] = None
    use_remote = False
    use_local_vision = False
    vision_model_name: str = DEFAULT_VISION_MODEL

    args_iter = iter(sys.argv[1:])
    for arg in args_iter:
        if arg == "--use_remote":
            use_remote = True
        elif arg == "--use_local_vision":
            use_local_vision = True
        elif not arg.startswith("-") and run_dir_name is None:
            run_dir_name = arg

    if not run_dir_name:
        print(
            "Error: missing <wandb_run_dir_name>\nUsage: python proctor/main.py <wandb_run_dir_name> [--use_remote] [--use_local_vision]"
        )
        return 1

    files_dir = get_run_files_dir(run_dir_name)
    if not files_dir:
        return 1

    logging.info(
        "Starting minimal supervisor for %s (remote=%s, local_vision=%s, vision_model=%s)",
        run_dir_name,
        use_remote,
        use_local_vision,
        vision_model_name,
    )

    while True:
        table_path = get_latest_completions_table(files_dir)
        pairs = extract_first_five_prompt_completion(table_path) if table_path else []
        latest_graph = get_latest_graph_image(files_dir)
        prompt = build_prompt(latest_graph, pairs)

        if use_local_vision:
            if not latest_graph:
                logging.info("No graph image found; skipping vision inference this cycle.")
                analysis = "ALL IS GOOD"
            else:
                logging.info("Calling local vision model for decision...")
                analysis = small_model_infer_local_vision(prompt, latest_graph, vision_model_name)
        else:
            logging.info("Calling %s model for decision...", "remote" if use_remote else "local")
            if use_remote:
                analysis = small_model_infer_remote(prompt, [latest_graph] if latest_graph else None)
            else:
                analysis = small_model_infer(prompt)

        print("\n=== Analysis ===\n" + (analysis or "") + "\n================\n")

        if analysis and ("ERROR FOUND" in analysis.upper()):
            context_paths = []
            if table_path:
                context_paths.append(table_path)
            if latest_graph:
                context_paths.append(latest_graph)
            # escalate_to_claude(analysis, context_paths)
            wake_up_user(analysis, run_dir_name, context_paths)
        else:
            logging.info("Decision: No escalation.")

        time.sleep(POLL_INTERVAL_SECS)


if __name__ == "__main__":
    sys.exit(main()) 