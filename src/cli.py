"""Command-line interface for PA3 matching phase."""

import argparse
from pathlib import Path

from .io import load_mesh, load_rigid_body, load_samples
from .matching import compute_matches
from .output import format_output_line


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute PA3 closest-point matches for pointer samples."
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="Path to mesh file (Problem3Mesh.sur).",
    )
    parser.add_argument(
        "--body-a",
        type=Path,
        required=True,
        help="Path to rigid body definition for body A (Problem3-BodyA.txt).",
    )
    parser.add_argument(
        "--body-b",
        type=Path,
        required=True,
        help="Path to rigid body definition for body B (Problem3-BodyB.txt).",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="Path to sample readings file (pa3-*-SampleReadings*.txt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for PA3 output file (pa3-X-Output.txt).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for PA3 matching computation."""
    args = parse_args()

    vertices, triangles, mesh_accel = load_mesh(args.mesh)
    body_a = load_rigid_body(args.body_a)
    body_b = load_rigid_body(args.body_b)
    samples, _sample_label = load_samples(
        args.samples, num_a=body_a.markers.shape[0], num_b=body_b.markers.shape[0]
    )

    results = compute_matches(vertices, triangles, body_a, body_b, samples, mesh_accel)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        header_label = args.output.name
        fh.write(f"{len(results)} {header_label} 0\n")
        for result in results:
            fh.write(
                format_output_line(result.sample_point, result.closest_point, result.distance)
            )
            fh.write("\n")

