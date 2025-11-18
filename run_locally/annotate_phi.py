#!/usr/bin/env python3
# annotate_phi.py
# pip install requests

import argparse
import os
import sys
import time
from pathlib import Path
import requests

from tqdm import tqdm
import threading, time

def wait_with_spinner(request_func):
    done = False

    def spinner():
        with tqdm(bar_format="{desc} {elapsed}", desc="Waiting for model") as pbar:
            while not done:
                time.sleep(0.1)
                pbar.update()

    t = threading.Thread(target=spinner)
    t.start()

    try:
        return request_func()
    finally:
        done = True
        t.join()

DEFAULT_ENDPOINT = "http://localhost:8002/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"

PROMPT_HEADER = """Please indicate all pieces of personal identifying information of the below listed categories:

<name> (referring only to the patient)
<admission_date> (**only once in the beginning**)
<discharge_date> (**only once in the beginning**)
<dob> (**only once in the beginning**)
<age> (referring only to the patient's current age, explicit and including some form of "year old" or as a general description e.g. "elderly" , "middle-aged", "young") 
<sex> (**only in the beginning**) 
<pronoun_subject> (only "she" or "he") 
<pronoun_object> (only "her" or "him") 
<pronoun_possessive> (only "his" or "her") 
<pronoun_reflexive> (only "himself" or "herself") 
<form_of_address> (e.g. "Mr.", "Ms.") 
<gendered_noun> (describes the patients gender, e.g. "M", "F", "male", "female", "man", "woman", "gentleman", "lady") 
<race> (describes the patients race, e.g. "black", "white", "caucasian", "asian", "hispanic") 
<spouse> (e.g. "wife", "husband")
<child> (e.g. "son", "daughter", "grandchild", possibly plural)
<discharge_instructions> (speaks directly to the patient, at the end of the document)
<facility> (referring only to the current, anonymized hospital, named directly or inside a greeting like "your ___ care team") 
<attending> (**only once in the beginning**)

If not mentioned otherwise, this information can appear multiple times anywhere in the text,
pay close attention throughout the whole document and inspect the context of each annotation.
Placeholders ___ at the beginning of the text should **always** be annotated with the appropriate label.

Use xml tags and refer to the following examples:

{example_name}

Repeat the entire document {file_name} **verbatim** with all the xml annotations included, don't alter
the text in any other way.
"""

FILES_WRAPPER = """<FILES>
<FILE_NAME>
{file_name}
</FILE_NAME>
<FILE_CONTENT>

{file_content}

</FILE_CONTENT>
</FILES>
"""

def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def build_user_prompt(file_name: str, file_content: str, example_name: str = "single_notes_hand_labelled/24420677_l.txt", example_annotated: str | None = None) -> str:
    header = PROMPT_HEADER.format(example_name=example_name, file_name=file_name)
    files_blob = FILES_WRAPPER.format(file_name=file_name, file_content=file_content)
    if example_annotated:
        return f"{header}\n\n{example_annotated}\n\n{files_blob}"
    return f"{header}\n\n{files_blob}"

def call_chat(endpoint: str, api_key: str | None, model: str, prompt: str, temperature: float = 0.0, max_tokens: int | None = None, timeout: int = 1200) -> str:
    headers = {"Content-Type": "application/json"}
    # Many local servers ignore the Authorization header, but include it if given.
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful clinical document annotator. Return only the fully annotated document text."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise SystemExit(f"HTTP error {resp.status_code}: {resp.text[:1000]}") from e

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise SystemExit(f"Unexpected response: {data}") from e

def main():
    parser = argparse.ArgumentParser(description="Annotate clinical text files with XML tags using an OpenAI-compatible chat endpoint.")
    g_in = parser.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--inputs", nargs="+", help="One or more file paths to annotate.")
    g_in.add_argument("--input-dir", help="Directory with .txt files to annotate.")
    parser.add_argument("--glob", default="*.txt", help="Glob used with --input-dir (default: *.txt).")
    parser.add_argument("--output-dir", default="annotated", help="Directory to write annotated files.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Chat endpoint (default: %(default)s).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID (default: %(default)s).")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key (env OPENAI_API_KEY if omitted).")
    parser.add_argument("--example-name", default="24420677_l.txt", help="Name referenced in the instructions as the example.")
    parser.add_argument("--fewshot", help="Path to an annotated example block to include verbatim above the <FILES> payload (optional).")
    parser.add_argument("--max-tokens", type=int, default=None, help="max_tokens for the response (optional).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--suffix", default=".annotated.txt", help="Output file suffix (default: .annotated.txt).")
    args = parser.parse_args()

    print("[DEBUG] Parsed endpoint:", args.endpoint, file=sys.stderr)

    # Collect inputs
    files: list[Path] = []
    if args.inputs:
        files = [Path(p) for p in args.inputs]
    else:
        root = Path(args.input_dir)
        files = sorted(root.rglob(args.glob))

    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    example_annotated = None
    if args.fewshot:
        example_annotated = load_text(Path(args.fewshot)).strip()

    for fp in files:
        if not fp.is_file():
            continue
        try:
            raw = load_text(fp)
        except Exception as e:
            print(f"[skip] {fp}: {e}", file=sys.stderr)
            continue

        prompt = build_user_prompt(file_name=fp.name, file_content=raw, example_name=args.example_name, example_annotated=example_annotated)

        print(f"[DEBUG] Using endpoint={args.endpoint}, model={args.model}, file={fp.name}", file=sys.stderr)

        # Simple retry loop for flaky local servers
        for attempt in range(3):
            try:
                annotated = wait_with_spinner(lambda: call_chat(
                    endpoint=args.endpoint,
                    api_key=args.api_key,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                ))
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))

        out_path = outdir / (fp.stem + args.suffix)
        out_path.write_text(annotated, encoding="utf-8")
        print(f"[ok] {fp.name} -> {out_path}")

if __name__ == "__main__":
    main()