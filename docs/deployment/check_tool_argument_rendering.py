"""Check MiniCPM5 historical tool-argument rendering via llama.cpp /apply-template.

Regression coverage accompanying OpenBMB/MiniCPM#386; no inference is performed.
"""

import argparse
import json
import re
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=8080, help="Local llama-server port")
args = parser.parse_args()
if not 1 <= args.port <= 65535:
    parser.error("port must be between 1 and 65535")

def post(path, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    # Synthetic fixtures only; bypass environment proxies for the local server.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=15) as response:
        return json.load(response)

cases = {
    "array": ["latency", "error"],
    "object": {"unit": "minute", "n": 2},
    "empty_array": [],
    "empty_object": {},
    "false": False,
    "zero": 0,
    "null": None,
    "nested": {"x": [1, False, None]},
    "unicode": ["延迟", "错误"],
    "xml_string": "a<&b",
    "cdata_end": "x]]>y",
    "plain": "demo",
}
rows = []
for name, value in cases.items():
    messages = [
        {"role": "user", "content": "Inspect the demo service."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "probe",
                        "arguments": json.dumps({"value": value}, ensure_ascii=False),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        {"role": "user", "content": "Continue."},
    ]
    response = post("/apply-template", {"messages": messages})
    prompt = response["prompt"]
    match = re.search(r'<param name="value">(.*?)</param>', prompt, re.S)
    rendered = match.group(1) if match else None
    try:
        actual = json.loads(rendered)
    except (ValueError, TypeError):
        actual = rendered
    if isinstance(value, str):
        passed = rendered in (value, "<![CDATA[" + value + "]]>")
    else:
        passed = rendered is not None and (
            json.dumps(actual, sort_keys=True, ensure_ascii=False)
            == json.dumps(value, sort_keys=True, ensure_ascii=False)
        )
    rows.append(
        {
            "case": name,
            "expected": value,
            "rendered": rendered,
            "preserved": passed,
        }
    )
for row in rows:
    print(f"{row['case']}: {'PASS' if row['preserved'] else 'FAIL'}; rendered={row['rendered']!r}")
passed = sum(row["preserved"] for row in rows)
print(f"{passed}/{len(rows)} cases preserved")
raise SystemExit(0 if passed == len(rows) else 1)
