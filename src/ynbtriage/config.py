"""Environment-driven configuration.

Defaults assume the in-container layout (/data volume, /app working dir). For
local (non-Docker) runs, override with a .env file or shell exports.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional
    pass

# SQLite file. In Docker this lives on the shared `dbdata` volume.
DB_PATH = Path(os.environ.get("YNB_DB_PATH", "/data/ynobuild.db"))

# Taxonomy source of truth (mounted read-only in containers).
TAXONOMY_PATH = Path(
    os.environ.get("YNB_TAXONOMY_PATH", str(Path(__file__).resolve().parents[2] / "taxonomy" / "taxonomy_map.yaml"))
)

# Where the Streamlit UI finds the API.
API_URL = os.environ.get("YNB_API_URL", "http://api:8000")

# Optional GitHub token to raise rate limits when fetching Dockerfiles.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or None

# Default annotator name shown in the UI.
DEFAULT_ANNOTATOR = os.environ.get("YNB_ANNOTATOR", "unknown")

# Safety cap on stored log size (bytes). Logs are usually already tail-truncated
# upstream; this only bites pathological giants, keeping the tail (where the error
# is) and marking the build truncated. Set 0 to disable.
LOG_MAX_BYTES = int(os.environ.get("YNB_LOG_MAX_BYTES", "262144"))
