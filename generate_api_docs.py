import json
from datetime import datetime
from pathlib import Path

# Paths
input_file = Path("docs/openapi.json")
output_file = Path("docs/API_Documentation.md")

# Load OpenAPI JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Metadata
title = data.get("info", {}).get("title", "FastAPI Project")
version = data.get("info", {}).get("version", "1.0.0")
description = data.get("info", {}).get("description", "API Documentation")
paths = data.get("paths", {})

# Start Markdown
md = []
md.append(f"# {title} API Documentation\n")
md.append(f"**Version:** {version}\n")
md.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
md.append(f"\n---\n{description}\n---\n")

# Loop through endpoints
for path, methods in paths.items():
    md.append(f"\n## `{path}`\n")
    for method, details in methods.items():
        summary = details.get("summary", "")
        md.append(f"### {method.upper()}: {summary}\n")
        md.append(f"**Description:** {details.get('description', '')}\n")
        md.append(f"**Tags:** {', '.join(details.get('tags', []))}\n")

        # Parameters
        params = details.get("parameters", [])
        if params:
            md.append("\n**Parameters:**\n")
            for p in params:
                md.append(f"- `{p['name']}` ({p['in']}) — {p.get('description', '')}\n")

        # Request Body
        if "requestBody" in details:
            md.append("\n**Request Body Example:**\n")
            content = details["requestBody"]["content"]
            if "application/json" in content:
                example = content["application/json"].get("example")
                if example:
                    md.append(f"```json\n{json.dumps(example, indent=2)}\n```\n")

        # Responses
        responses = details.get("responses", {})
        md.append("\n**Responses:**\n")
        for code, resp in responses.items():
            md.append(f"- `{code}` — {resp.get('description', '')}\n")

        md.append("\n---\n")

# Write output file
output_file.write_text("\n".join(md), encoding="utf-8")
print(f"✅ Documentation generated at {output_file}")

