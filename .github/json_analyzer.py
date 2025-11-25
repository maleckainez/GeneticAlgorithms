"""Small python script to automage coverage badges."""

import json
from pathlib import Path

file_location = Path(__file__).resolve().parents[0]
with open(f"{file_location}/coverage.json") as f:
    coverage = json.load(f)
coverage_value = coverage["totals"]["percent_covered"]
badge_message = f"{coverage_value:.0f}%"
if coverage_value > 0 and coverage_value <= 75:
    color = "#e05d44"
elif coverage_value > 75 and coverage_value <= 90:
    color = "#dfb317"
elif coverage_value > 90 and coverage_value <= 95:
    color = "#a3c51c"
elif coverage_value > 95 and coverage_value <= 100:
    color = "#4c1"
else:
    color = "#9f9f9f"
    badge_message = "Unknown"

coverage_badge_content = {
    "schemaVersion": 1,
    "label": "Coverage",
    "message": badge_message,
    "color": color,
}
print(coverage_badge_content)
with open(Path(f"{file_location}/coverage.json"), "w") as file:
    json.dump(coverage_badge_content, file, indent=4)
