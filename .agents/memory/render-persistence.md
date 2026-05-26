---
name: Render.com persistence
description: How to persist paper trades across deploys on Render.com.
---
Render.com has an ephemeral filesystem — files written to the project root are lost on deploy.

**Fix:** paper_trade.py uses PAPER_FILE = os.path.join(os.environ.get("DATA_DIR", "."), "paper_trades.json")

**How to apply on Render:** Set env var DATA_DIR=/var/data and mount a persistent disk at /var/data. Without this, trades are lost on every redeploy.
