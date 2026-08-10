from __future__ import annotations

import multiprocessing as mp
from app import app


if __name__ == "__main__":
    mp.freeze_support()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
