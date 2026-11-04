"""Delete every Fireworks training session named in the given logs (after a killed, stalled or failed trainer)."""
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from fireworks.training.sdk import FireworksClient  # noqa: E402

ids = sorted({m for f in sys.argv[1:] if Path(f).exists() for m in re.findall(r"ts-[0-9a-f]{32}", Path(f).read_text())})
c = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"])
for t in ids:
    try:
        c.delete_training_session(f"accounts/vvennela/trainingSessions/{t}")
        print("deleted", t)
    except Exception as e:  # noqa: BLE001 - already gone
        print("not deleted", t, type(e).__name__, str(e)[:80])
