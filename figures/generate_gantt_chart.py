"""
Generate Gantt Chart for Semantic EQ Project
Timeline: Late December 2025 - February 2026
Grouped by weeks
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import numpy as np

# Tasks organized by week with group headers
weeks = [
    {
        "name": "Week 1 (Dec 20-26): Research & Planning",
        "tasks": [
            ("Literature Review", "2025-12-20", "2025-12-26", "research"),
            ("Interim Report Writing", "2025-12-20", "2025-12-26", "research"),
        ]
    },
    {
        "name": "Week 2 (Dec 27 - Jan 2): Holiday / Prep",
        "tasks": [
            ("Literature Review (cont.)", "2025-12-27", "2026-01-02", "research"),
            ("Dataset Exploration", "2025-12-30", "2026-01-02", "dev_success"),
        ]
    },
    {
        "name": "Week 3 (Jan 3-9): Initial Model Development",
        "tasks": [
            ("SAFE-DB Dataset Prep", "2026-01-02", "2026-01-04", "dev_success"),
            ("V1 Training (Failed)", "2026-01-04", "2026-01-05", "dev_failed"),
            ("Problem Diagnosis", "2026-01-05", "2026-01-06", "analysis"),
            ("V2 Model (Log-scale Fix)", "2026-01-05", "2026-01-08", "dev_success"),
            ("Initial Figures", "2026-01-06", "2026-01-09", "figures"),
        ]
    },
    {
        "name": "Week 4 (Jan 10-16): E2E Pipeline Setup",
        "tasks": [
            ("W&B Integration", "2026-01-12", "2026-01-15", "dev_success"),
            ("Cluster Environment Setup", "2026-01-13", "2026-01-16", "dev_success"),
        ]
    },
    {
        "name": "Week 5 (Jan 17-23): E2E Architecture & Training",
        "tasks": [
            ("E2E DDSP Architecture", "2026-01-17", "2026-01-21", "dev_success"),
            ("Model Comparison Framework", "2026-01-18", "2026-01-21", "dev_success"),
            ("Audio Encoder Training", "2026-01-20", "2026-01-24", "training"),
            ("FMA Evaluation Suite", "2026-01-22", "2026-01-24", "evaluation"),
        ]
    },
    {
        "name": "Week 6 (Jan 24-30): Evaluation & Analysis",
        "tasks": [
            ("Echonest Correlation", "2026-01-24", "2026-01-26", "evaluation"),
            ("UCL Cluster Eval Runs", "2026-01-25", "2026-01-28", "evaluation"),
            ("Bug Fixes & Debugging", "2026-01-25", "2026-01-28", "bugfix"),
            ("Results Analysis", "2026-01-27", "2026-01-30", "analysis"),
        ]
    },
    {
        "name": "Week 7 (Jan 31 - Feb 2): Paper Preparation",
        "tasks": [
            ("Final Figure Generation", "2026-01-29", "2026-02-02", "figures"),
            ("AES Express Paper", "2026-01-30", "2026-02-02", "paper"),
        ]
    },
]

# Color scheme
colors = {
    "research": "#3498db",      # Blue
    "dev_failed": "#e74c3c",    # Red
    "dev_success": "#2ecc71",   # Green
    "analysis": "#f39c12",      # Orange
    "figures": "#9b59b6",       # Purple
    "evaluation": "#1abc9c",    # Teal
    "bugfix": "#e67e22",        # Dark orange
    "paper": "#34495e",         # Dark gray
    "training": "#e91e63",      # Pink
}

def create_gantt_chart():
    # Count total rows (tasks + week headers)
    total_rows = sum(len(w["tasks"]) + 1 for w in weeks)  # +1 for each week header

    fig, ax = plt.subplots(figsize=(14, total_rows * 0.45 + 2))

    # Parse dates
    start_date = datetime(2025, 12, 20)
    end_date = datetime(2026, 2, 5)
    total_days = (end_date - start_date).days

    # Plot tasks grouped by week
    y_pos = total_rows
    yticks = []
    yticklabels = []
    week_header_positions = []

    for week in weeks:
        # Week header row
        week_header_positions.append(y_pos)
        yticks.append(y_pos)
        yticklabels.append(week["name"])

        # Draw week background
        ax.axhspan(y_pos - len(week["tasks"]) - 0.5, y_pos + 0.5,
                   alpha=0.08, color='gray')

        y_pos -= 1

        # Tasks in this week
        for task_name, start_str, end_str, category in week["tasks"]:
            start = datetime.strptime(start_str, "%Y-%m-%d")
            end = datetime.strptime(end_str, "%Y-%m-%d")

            start_offset = (start - start_date).days
            duration = max((end - start).days, 1)

            color = colors.get(category, "#95a5a6")
            ax.barh(y_pos, duration, left=start_offset, height=0.6,
                    color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

            yticks.append(y_pos)
            yticklabels.append(f"    {task_name}")  # Indent task names
            y_pos -= 1

    # Configure axes
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)

    # Bold the week headers
    for i, label in enumerate(ax.get_yticklabels()):
        if yticks[i] in week_header_positions:
            label.set_fontweight('bold')
            label.set_fontsize(10)

    # X-axis: dates with weekly ticks
    xticks = list(range(0, total_days + 1, 7))
    xticklabels = [(start_date + timedelta(days=d)).strftime("%b %d") for d in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=9)

    # Add week vertical lines
    for d in range(0, total_days + 1, 7):
        ax.axvline(x=d, color='lightgray', linestyle='-', alpha=0.5, linewidth=0.5)

    # Month separators
    jan_start = (datetime(2026, 1, 1) - start_date).days
    feb_start = (datetime(2026, 2, 1) - start_date).days
    ax.axvline(x=jan_start, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axvline(x=feb_start, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

    # Today marker
    today = datetime(2026, 2, 2)
    today_offset = (today - start_date).days
    ax.axvline(x=today_offset, color='red', linestyle='-', alpha=0.8, linewidth=2)
    ax.text(today_offset + 0.3, 1, "TODAY", fontsize=8, color='red', rotation=90, va='bottom')

    # Deadline marker
    deadline = datetime(2026, 2, 1)
    deadline_offset = (deadline - start_date).days
    ax.axvline(x=deadline_offset, color='darkred', linestyle=':', alpha=0.8, linewidth=2)
    ax.text(deadline_offset - 0.3, total_rows, "AES Deadline", fontsize=8, color='darkred', ha='right')

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors["research"], label="Research"),
        mpatches.Patch(color=colors["dev_success"], label="Development"),
        mpatches.Patch(color=colors["dev_failed"], label="Failed"),
        mpatches.Patch(color=colors["training"], label="GPU Training"),
        mpatches.Patch(color=colors["analysis"], label="Analysis"),
        mpatches.Patch(color=colors["figures"], label="Figures"),
        mpatches.Patch(color=colors["evaluation"], label="Evaluation"),
        mpatches.Patch(color=colors["bugfix"], label="Bug Fix"),
        mpatches.Patch(color=colors["paper"], label="Paper"),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=3,
              bbox_to_anchor=(1, 1.02))

    # Styling
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title("Semantic EQ Project Timeline - Weekly Breakdown\n(December 2025 - February 2026)",
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xlim(-1, total_days + 2)
    ax.set_ylim(0, total_rows + 1)

    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    plt.savefig('figures/project_gantt_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig('figures/project_gantt_chart.pdf', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Saved Gantt chart to figures/project_gantt_chart.png")
    print("Saved PDF version to figures/project_gantt_chart.pdf")


if __name__ == '__main__':
    create_gantt_chart()
