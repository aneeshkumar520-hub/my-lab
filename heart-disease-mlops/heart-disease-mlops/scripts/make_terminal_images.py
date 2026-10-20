"""Render real captured terminal output (logs/*.txt) as PNGs in screenshots/.
These are images of text output, NOT GUI screenshots."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JOBS = {
    "01_terminal_clean_setup_run": "clean_setup_run.txt",
    "02_terminal_training_results": "training_table.txt",
    "03_terminal_mlflow_runs": "mlflow_runs.txt",
    "04_terminal_pytest_coverage": "pytest_output.txt",
    "05_terminal_flake8": "flake8_output.txt",
    "06_terminal_ci_failure_demo": "ci_failure_demo.txt",
    "07_terminal_api_curl_demo": "api_demo.txt",
    "08_terminal_api_metrics": "api_metrics.txt",
    "09_terminal_api_request_logs": "api_logs.txt",
}


def render(src: Path, dst: Path, max_lines=40, max_cols=225):
    lines = [ln[:max_cols] for ln in src.read_text().splitlines()
             if "WARNING" not in ln and "Warning" not in ln][-max_lines:]
    w = max(8, min(21, max(len(ln) for ln in lines) * 0.095 + 0.6))
    h = 0.32 + 0.205 * len(lines)
    fig = plt.figure(figsize=(w, h), facecolor="#1e1e2e")
    for k, ln in enumerate(lines):
        color = "#a6e3a1" if ln.startswith("$") else "#cdd6f4"
        fig.text(0.015, 1 - (0.16 + 0.205 * k) / h, ln, family="DejaVu Sans Mono", size=8.2,
                 color=color, va="center", parse_math=False)
    fig.savefig(dst, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    out = ROOT / "screenshots"
    out.mkdir(exist_ok=True)
    for name, f in JOBS.items():
        p = ROOT / "logs" / f
        if p.exists():
            render(p, out / f"{name}.png")
            print("wrote", name)
