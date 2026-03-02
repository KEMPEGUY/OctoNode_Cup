# competition/render_leaderboard.py

import csv
from pathlib import Path
from datetime import datetime
from html import escape

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "leaderboard" / "leaderboard.csv"
MD_PATH = ROOT / "leaderboard" / "leaderboard.md"
HTML_PATH = ROOT / "docs" / "leaderboard.html"

REQUIRED_COLUMNS = {"team", "score", "timestamp_utc"}


def read_rows():
    if not CSV_PATH.exists():
        return []

    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(
                "leaderboard.csv must contain columns: team, score, timestamp_utc"
            )

        rows = []
        for r in reader:
            team = (r.get("team") or "").strip()
            score = r.get("score")
            timestamp = r.get("timestamp_utc")

            if not team:
                continue

            try:
                score = float(score)
            except (TypeError, ValueError):
                continue

            rows.append({
                "team": team,
                "score": score,
                "timestamp_utc": timestamp
            })

    return rows


def sort_rows(rows):
    # Primary: score DESC
    # Secondary: timestamp ASC (earlier submission wins tie)
    return sorted(
        rows,
        key=lambda r: (-r["score"], r["timestamp_utc"])
    )


def generate_markdown(rows):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Leaderboard\n\n",
        f"Last updated: {now}\n\n",
        "| Rank | Team | Score | Date |\n",
        "|---:|---|---:|---|\n"
    ]

    for i, r in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {escape(r['team'])} | {r['score']:.8f} | {r['timestamp_utc']} |\n"
        )

    MD_PATH.write_text("".join(lines), encoding="utf-8")


def generate_html(rows):
    table_rows = []

    for i, r in enumerate(rows, start=1):
        table_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{escape(r['team'])}</td>"
            f"<td>{r['score']:.8f}</td>"
            f"<td>{r['timestamp_utc']}</td>"
            "</tr>"
        )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OctoNode Cup Leaderboard</title>
    <link rel="stylesheet" href="leaderboard.css">
</head>
<body>
    <h1>OctoNode Cup Leaderboard</h1>
    <p>Last updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Team</th>
                <th>Score</th>
                <th>Date</th>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>
</body>
</html>
"""

    HTML_PATH.write_text(html_content, encoding="utf-8")


def main():
    rows = read_rows()
    rows = sort_rows(rows)
    generate_markdown(rows)
    generate_html(rows)


if __name__ == "__main__":
    main()
