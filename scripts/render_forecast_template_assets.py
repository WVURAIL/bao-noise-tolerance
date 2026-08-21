#!/usr/bin/env python3
"""Render dissertation assets from the forecast-template comparison CSVs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from baonoise.plots import GRID, MUTED, SERIES, SURFACE, setup_style
from baonoise.residual_templates import FAMILIES


COMPARISON_SCHEMA = "baonoise-forecast-template-comparison-csv-v1"
CHANNEL_SCHEMA = "baonoise-forecast-channel-mapping-csv-v1"
STATUS_SCHEMA = "baonoise-forecast-template-status-csv-v1"
MANIFEST_SCHEMA = "baonoise-forecast-completion-release-manifest-v1"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA_PATH = (
    ROOT / "docs" / "forecast-completion-release-manifest.schema.json")
DISPLAY = {
    "noise_shaped": "Noise-shaped unit",
    "low_kparallel": r"Low-$k_\parallel$",
    "wedge_like": "Wedge-like",
    "k_shell_localized": r"Localized $k$ shell",
}
TEXT_DISPLAY = {
    "noise_shaped": "Noise-shaped unit",
    "low_kparallel": "Low-$k_\\parallel$",
    "wedge_like": "Wedge-like",
    "k_shell_localized": "Localized $k$ shell",
}
PDF_METADATA_DATE = datetime(2026, 8, 20, tzinfo=timezone.utc)

CAPTION = (
    "Model-only residual-amplitude tolerance on the common f-sigma-8 target "
    "at one on-sky year, mapped to ATSC channels 14--36. Panel (a) uses the "
    "independent per-bin Appendix-A estimator and panel (b) uses the joint "
    "multi-redshift-bin Fisher estimator. For a channel crossing a redshift-"
    "bin edge, the plotted value is the smaller accepted tolerance over every "
    "bin with non-zero frequency overlap; it is not an overlap-weighted "
    "average. All plotted f-sigma-8 points pass the plus/minus-10-percent "
    "stability gate. The four curves are analytic unit-response sensitivity "
    "hypotheses normalized to contemporaneous thermal-noise power, not "
    "empirically inferred template probabilities or measured channel "
    "residuals. Frequency-, baseline-, and sidereal-dependent visibility "
    "templates remain explicitly data-dependent and incomplete."
)


def _read_rows(path: Path, schema: str) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row.get("schema") != schema for row in rows):
        raise ValueError(f"{path} is not a complete {schema} table")
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _scientific_identity_summary(evidence_paths: list[Path]) -> dict:
    per_evidence = []
    bao_digests = set()
    bao_manifests = set()
    bao_commits = set()
    radiofisher_digests = set()
    radiofisher_manifests = set()
    radiofisher_commits = set()
    for path in evidence_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload["bank"]["scientific_identity"]
        if identity.get("schema") \
                != "baonoise-scientific-evaluation-identity-v1" \
                or identity.get("verified_equal") is not True:
            raise ValueError(
                f"{path.name} has no verified build/evaluation identity")
        build = identity["bank_build"]
        evaluation = identity["evaluation"]
        for package in ("baonoise", "radiofisher"):
            if build[package].get("git_dirty") is not False \
                    or evaluation[package].get("git_dirty") is not False:
                raise ValueError(
                    f"{path.name} {package} source identity is dirty")
            for field in ("working_tree_sha256", "source_manifest"):
                if build[package][field] != evaluation[package][field]:
                    raise ValueError(
                        f"{path.name} {package} build/evaluation {field} "
                        "mismatch")
        bao = build["baonoise"]
        radiofisher = build["radiofisher"]
        bao_digests.add(bao["working_tree_sha256"])
        bao_manifests.add(json.dumps(
            bao["source_manifest"], sort_keys=True, separators=(",", ":")))
        bao_commits.update((
            bao["git_commit"], evaluation["baonoise"]["git_commit"]))
        radiofisher_digests.add(radiofisher["working_tree_sha256"])
        radiofisher_manifests.add(json.dumps(
            radiofisher["source_manifest"],
            sort_keys=True, separators=(",", ":")))
        radiofisher_commits.update((
            radiofisher["git_commit"],
            evaluation["radiofisher"]["git_commit"]))
        per_evidence.append({
            "path": _portable_path(path),
            "baonoise_build_git_commit": bao["git_commit"],
            "baonoise_evaluation_git_commit":
                evaluation["baonoise"]["git_commit"],
            "radiofisher_build_git_commit": radiofisher["git_commit"],
            "radiofisher_evaluation_git_commit":
                evaluation["radiofisher"]["git_commit"],
            "verified_equal": True,
        })
    if len(bao_digests) != 1 or len(bao_manifests) != 1 \
            or len(radiofisher_digests) != 1 \
            or len(radiofisher_manifests) != 1 \
            or len(radiofisher_commits) != 1:
        raise ValueError(
            "evidence files do not share one scientific Bao/RadioFisher "
            "content identity")
    return {
        "authoritative_identity_json_pointer":
            "/bank/scientific_identity",
        "identity_schema": "baonoise-scientific-evaluation-identity-v1",
        "all_build_evaluation_pairs_verified_equal": True,
        "baonoise": {
            "clean_git_commits": sorted(bao_commits),
            "working_tree_sha256": next(iter(bao_digests)),
            "source_manifest": json.loads(next(iter(bao_manifests))),
        },
        "radiofisher": {
            "clean_git_commit": next(iter(radiofisher_commits)),
            "working_tree_sha256": next(iter(radiofisher_digests)),
            "source_manifest": json.loads(next(iter(radiofisher_manifests))),
        },
        "per_evidence": per_evidence,
        "commit_note": (
            "The scalar baseline and named-template banks were evaluated at "
            "different clean Bao commits, but the canonical scientific "
            "source manifest and content digest are identical. Release-only "
            "scripts, tests, documentation, and outputs are outside that "
            "scientific source manifest. The Git commit containing this "
            "manifest is reported externally to avoid a self-reference."),
    }


def _validate(comparison_rows: list[dict], channel_rows: list[dict]) -> None:
    expected_bin_keys = {
        (family, ibin) for family in FAMILIES for ibin in range(5, 12)}
    bin_keys = {
        (row["family"], int(row["bin_index"]))
        for row in comparison_rows}
    expected_channel_keys = {
        (family, channel)
        for family in FAMILIES for channel in range(14, 37)}
    channel_keys = {
        (row["family"], int(row["channel"])) for row in channel_rows}
    if bin_keys != expected_bin_keys or len(comparison_rows) != 28:
        raise ValueError("comparison table must contain four families x seven bins")
    if channel_keys != expected_channel_keys or len(channel_rows) != 92:
        raise ValueError("channel table must contain four families x 23 channels")
    if any(row["shared_target"] != "fs8"
           or row["perbin_status"] != "accepted"
           or row["combined_status"] != "accepted"
           for row in channel_rows):
        raise ValueError("figure requires accepted common-fs8 channel mappings")


def _aggregate(comparison_rows: list[dict], channel_rows: list[dict]) \
        -> dict[str, dict]:
    summary = {}
    for family in FAMILIES:
        bins = [row for row in comparison_rows if row["family"] == family]
        channels = [row for row in channel_rows if row["family"] == family]
        perbin_values = np.array([
            float(row["perbin_fs8_tolerance"]) for row in bins])
        combined_values = np.array([
            float(row["combined_fs8_tolerance"]) for row in bins])
        perbin_ranks = {int(row["perbin_strictness_rank"]) for row in channels}
        combined_ranks = {
            int(row["combined_strictness_rank"]) for row in channels}
        if len(perbin_ranks) != 1 or len(combined_ranks) != 1:
            raise ValueError(
                f"{family} strictness rank changes across physical channels")
        summary[family] = {
            "perbin_accepted": sum(int(row["perbin_accepted"]) for row in bins),
            "perbin_rejected": sum(int(row["perbin_rejected"]) for row in bins),
            "combined_noise_accepted": sum(
                int(row["combined_noise_grid_accepted"]) for row in bins),
            "combined_noise_rejected": sum(
                int(row["combined_noise_grid_rejected"]) for row in bins),
            "combined_fixed_accepted": sum(
                int(row["combined_fixed_grid_accepted"]) for row in bins),
            "combined_fixed_rejected": sum(
                int(row["combined_fixed_grid_rejected"]) for row in bins),
            "perbin_min": float(perbin_values.min()),
            "perbin_max": float(perbin_values.max()),
            "combined_min": float(combined_values.min()),
            "combined_max": float(combined_values.max()),
            "perbin_rank": next(iter(perbin_ranks)),
            "combined_rank": next(iter(combined_ranks)),
        }
    return summary


def _write_tex(path: Path, summary: dict[str, dict]) -> None:
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{All-seven-DTV-bin model-only residual-template forecast "
        r"summary. A/R is the accepted/rejected Fisher-point count after the "
        r"$\pm10\%$ integration-time stability gate. The per-bin column "
        r"contains $7\times3=21$ one-year target points; each combined column "
        r"contains $7\times3\times3=63$ bin/time/target points. Ranges are "
        r"accepted one-year $f\sigma_8$ residual-amplitude tolerances in "
        r"contemporaneous thermal-noise units. Analytic-family rank is a "
        r"sensitivity-envelope ordering only and does not revise a measured "
        r"channel-policy ranking.}",
        r"\label{tab:forecast-template-comparison}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Template & Per-bin A/R & Combined noise A/R & Combined fixed A/R "
        r"& Per-bin $f\sigma_8$ range & Combined $f\sigma_8$ range \\",
        r"\hline",
    ]
    for family in FAMILIES:
        row = summary[family]
        lines.append((
            f"{TEXT_DISPLAY[family]} & "
            f"{row['perbin_accepted']}/{row['perbin_rejected']} & "
            f"{row['combined_noise_accepted']}/"
            f"{row['combined_noise_rejected']} & "
            f"{row['combined_fixed_accepted']}/"
            f"{row['combined_fixed_rejected']} & "
            f"{row['perbin_min']:.4g}--{row['perbin_max']:.4g} & "
            f"{row['combined_min']:.4g}--{row['combined_max']:.4g} "
            + r"\\"))
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _render_figure(path: Path, channel_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True)
    fields = (
        ("perbin_conservative_tolerance",
         "(a) Independent per-bin Appendix-A estimator"),
        ("combined_conservative_tolerance",
         "(b) Joint multi-redshift-bin Fisher estimator"),
    )
    colors = dict(zip(FAMILIES, SERIES[:len(FAMILIES)]))
    straddling_channels = (17, 20, 23, 26, 30, 34)
    for ax, (field, title) in zip(axes, fields):
        for channel in straddling_channels:
            ax.axvspan(
                channel - 0.48, channel + 0.48,
                facecolor=GRID, alpha=0.30, edgecolor="none", zorder=0)
        for family in FAMILIES:
            rows = sorted(
                (row for row in channel_rows if row["family"] == family),
                key=lambda row: int(row["channel"]))
            x = np.array([int(row["channel"]) for row in rows])
            y = np.array([float(row[field]) for row in rows])
            ax.step(
                x, y, where="mid", color=colors[family],
                label=DISPLAY[family], lw=1.75)
            ax.plot(
                x, y, linestyle="none", marker="o", ms=3.4,
                color=colors[family], markeredgecolor=SURFACE,
                markeredgewidth=0.65)
        ax.set_yscale("log")
        ax.set_ylim(1.2e-3, 4.6e-3)
        ax.set_yticks([1.5e-3, 2e-3, 3e-3, 4e-3])
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, _position: f"{1e3 * value:g}"))
        ax.grid(True, axis="y", which="major")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_ylabel(r"Accepted $f\sigma_8$ tolerance [$\times10^{-3}$]")
        ax.spines["bottom"].set_color(GRID)
        ax.spines["left"].set_visible(False)
    axes[1].set_xticks(range(14, 37))
    axes[1].set_xticklabels([str(value) for value in range(14, 37)], fontsize=8)
    axes[1].set_xlim(13.5, 36.5)
    axes[1].set_xlabel("ATSC physical channel (470--608 MHz; 6 MHz each)")
    axes[1].text(
        0.01, -0.30,
        "Gray bands mark channels crossing a forecast-bin edge; the smaller "
        "overlap-bin tolerance is plotted.",
        transform=axes[1].transAxes, color=MUTED, fontsize=8.4,
        va="top", ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 0.985), columnspacing=1.4, handlelength=2.4)
    fig.suptitle(
        r"Model-only residual-shape sensitivity on the common $f\sigma_8$ target",
        y=1.035, fontsize=12)
    fig.subplots_adjust(top=0.88, bottom=0.14, hspace=0.31)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, dpi=220, bbox_inches="tight",
        metadata={
            "Title": "Model-only residual-shape sensitivity by ATSC channel",
            "Author": "Dylan Gormley",
            "Description": CAPTION,
        })
    pdf = path.with_suffix(".pdf")
    fig.savefig(
        pdf, bbox_inches="tight",
        metadata={
            "Title": "Model-only residual-shape sensitivity by ATSC channel",
            "Author": "Dylan Gormley",
            "Subject": CAPTION,
            "Creator": "BaoNoiseTolerance reproducible forecast renderer",
            "CreationDate": PDF_METADATA_DATE,
            "ModDate": PDF_METADATA_DATE,
        })
    plt.close(fig)


def _write_manifest(path: Path, *, comparison: Path, channels_path: Path,
                    status: Path, figure: Path, table: Path, caption: Path,
                    comparison_rows: list[dict], status_rows: list[dict]) \
        -> None:
    evidence_paths = sorted({
        comparison.parent / row["evidence_file"]
        for row in comparison_rows})
    artifact_paths = [
        *evidence_paths, comparison, channels_path, status, figure,
        figure.with_suffix(".pdf"), table, caption, MANIFEST_SCHEMA_PATH,
    ]
    missing = [item for item in artifact_paths if not item.is_file()]
    if missing:
        raise ValueError(
            "release manifest cannot find: "
            + ", ".join(str(item) for item in missing))
    empirical_refusals = [
        {
            "family": row["family"],
            "status": row["execution_status"],
            "required": row["external_data_or_interface_required"],
        }
        for row in status_rows
        if row["category"] == "empirical_visibility"]
    if len(empirical_refusals) != 3 or any(
            row["status"] != "data_dependent_incomplete"
            for row in empirical_refusals):
        raise ValueError(
            "status table must retain all three empirical refusals")
    scientific_identities = _scientific_identity_summary(evidence_paths)
    payload = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "manifest_schema_sha256": _file_sha256(MANIFEST_SCHEMA_PATH),
        "wall_clock_fields_included": False,
        "absolute_paths_included": False,
        "artifact_count": len(artifact_paths),
        "artifacts": [
            {
                "path": _portable_path(item),
                "sha256": _file_sha256(item),
                "size_bytes": item.stat().st_size,
            }
            for item in artifact_paths
        ],
        "generators": [
            {
                "path": _portable_path(
                    ROOT / "scripts" / "forecast_template_comparison.py"),
                "sha256": _file_sha256(
                    ROOT / "scripts" / "forecast_template_comparison.py"),
            },
            {
                "path": _portable_path(Path(__file__)),
                "sha256": _file_sha256(Path(__file__)),
            },
        ],
        "scientific_scope": {
            "all_dtv_bin_indices": list(range(5, 12)),
            "physical_channels": list(range(14, 37)),
            "analytic_family_count": 4,
            "common_channel_target": "fs8",
            "new_telescope_data_used": False,
            "existing_policy_status_or_ranking_changed": False,
        },
        "scientific_identities": scientific_identities,
        "empirical_template_refusals": empirical_refusals,
        "limitations": [
            "analytic unit-response shapes are sensitivity hypotheses, not "
            "measured template probabilities",
            "boundary-channel propagation takes the minimum accepted "
            "tolerance across every non-zero-overlap bin",
            "rejected dilation values are retained for diagnosis and are not "
            "eligible for channel-policy propagation",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", type=Path,
        default=Path("out/forecast_completion_template_comparison.csv"))
    parser.add_argument(
        "--channels", type=Path,
        default=Path("out/forecast_completion_channel_mapping.csv"))
    parser.add_argument(
        "--status", type=Path,
        default=Path("out/forecast_completion_template_status.csv"))
    parser.add_argument(
        "--figure", type=Path,
        default=Path("out/forecast_completion_channel_tolerances.png"))
    parser.add_argument(
        "--table", type=Path,
        default=Path("out/forecast_completion_template_summary.tex"))
    parser.add_argument(
        "--caption", type=Path,
        default=Path("out/forecast_completion_channel_tolerances_caption.txt"))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("out/forecast_completion_release_manifest.json"))
    args = parser.parse_args(argv)

    try:
        comparison_rows = _read_rows(args.comparison, COMPARISON_SCHEMA)
        channel_rows = _read_rows(args.channels, CHANNEL_SCHEMA)
        status_rows = _read_rows(args.status, STATUS_SCHEMA)
        _validate(comparison_rows, channel_rows)
        summary = _aggregate(comparison_rows, channel_rows)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))

    _render_figure(args.figure, channel_rows)
    _write_tex(args.table, summary)
    args.caption.parent.mkdir(parents=True, exist_ok=True)
    args.caption.write_text(CAPTION + "\n", encoding="utf-8", newline="\n")
    try:
        _write_manifest(
            args.manifest, comparison=args.comparison,
            channels_path=args.channels, status=args.status,
            figure=args.figure, table=args.table, caption=args.caption,
            comparison_rows=comparison_rows, status_rows=status_rows)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"wrote {args.figure} and {args.figure.with_suffix('.pdf')}")
    print(f"wrote {args.table}")
    print(f"wrote {args.caption}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
