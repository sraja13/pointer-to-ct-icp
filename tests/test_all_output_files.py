"""Comprehensive tests for all PA3 output files.

This module tests every single output file against expected results,
ensuring complete coverage of all datasets (A-F) and all output formats.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pa3 import (  # noqa: E402
    MatchResult,
    compute_matches,
    load_mesh,
    load_rigid_body,
    load_samples,
)

# Data directories
DATA_ROOT = Path("data/2025 PA345 Student Data")
OUTPUT_ROOT = Path("output")

DEBUG_DATASETS = (
    ("A", 1e-2, 1e-2),
    ("B", 1e-2, 1e-2),
    ("C", 1e-2, 1e-2),
    ("D", 1e-2, 2e-2),
    ("E", 2e-2, 2e-2),
    ("F", 2e-2, 2e-2),
)

DEBUG_LETTERS = tuple(letter for letter, _, _ in DEBUG_DATASETS)
UNKNOWN_LETTERS = ("G", "H", "J")
ALL_SAMPLE_LETTERS = DEBUG_LETTERS + UNKNOWN_LETTERS


def _load_output_file(path: Path):
    """Load an output file and parse its contents."""
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split()
        expected_count = int(header[0])
        rows = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = [float(tok) for tok in line.split()]
            if len(tokens) >= 7:
                sample = np.array(tokens[0:3])
                closest = np.array(tokens[3:6])
                distance = tokens[6]
                rows.append((sample, closest, distance))
    assert len(rows) == expected_count, f"Expected {expected_count} rows, got {len(rows)}"
    return rows


def _read_sample_header(path: Path) -> tuple[int, int]:
    """Read the marker and frame counts from a sample readings file."""
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
    tokens = header.replace(",", " ").split()
    marker_count = int(tokens[0])
    frame_count = int(tokens[1])
    return marker_count, frame_count


def _compare_output_files(
    computed_path: Path,
    expected_path: Path,
    abs_tol: float = 1e-2,
    coord_tol: float = 1e-2,
):
    """Compare two output files line by line."""
    computed = _load_output_file(computed_path)
    expected = _load_output_file(expected_path)
    
    assert len(computed) == len(expected), \
        f"Mismatch in row count: {len(computed)} vs {len(expected)}"
    
    for i, (comp_row, exp_row) in enumerate(zip(computed, expected)):
        comp_sample, comp_closest, comp_dist = comp_row
        exp_sample, exp_closest, exp_dist = exp_row
        
        assert np.allclose(comp_sample, exp_sample, atol=coord_tol), \
            f"Row {i}: Sample point mismatch: {comp_sample} vs {exp_sample}"
        assert np.allclose(comp_closest, exp_closest, atol=coord_tol), \
            f"Row {i}: Closest point mismatch: {comp_closest} vs {exp_closest}"
        assert math.isclose(comp_dist, exp_dist, abs_tol=abs_tol), \
            f"Row {i}: Distance mismatch: {comp_dist} vs {exp_dist}"


def _compute_and_save_output(letter: str, output_path: Path):
    """Compute matches for a dataset and save to output file."""
    mesh_vertices, mesh_triangles = load_mesh(DATA_ROOT / "Problem3Mesh.sur")
    body_a = load_rigid_body(DATA_ROOT / "Problem3-BodyA.txt")
    body_b = load_rigid_body(DATA_ROOT / "Problem3-BodyB.txt")
    samples, _ = load_samples(
        DATA_ROOT / f"PA3-{letter}-Debug-SampleReadingsTest.txt",
        num_a=body_a.markers.shape[0],
        num_b=body_b.markers.shape[0],
    )
    results = compute_matches(mesh_vertices, mesh_triangles, body_a, body_b, samples)
    
    # Save to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from src.output import format_output_line
    with output_path.open("w", encoding="utf-8") as fh:
        header_label = output_path.name
        fh.write(f"{len(results)} {header_label} 0\n")
        for result in results:
            fh.write(
                format_output_line(result.sample_point, result.closest_point, result.distance)
            )
            fh.write("\n")
    
    return results


class TestAllOutputFiles:
    """Comprehensive tests for all output files."""
    
    @pytest.mark.parametrize(("letter", "abs_tol", "coord_tol"), DEBUG_DATASETS)
    def test_output_files_against_debug_answers(self, letter: str, abs_tol: float, coord_tol: float):
        """Verify saved output files match the official Debug-Answer files. Confirms each value stays within the specified coordinate and distance tolerances."""
        # Check if output file exists
        output_file = OUTPUT_ROOT / f"pa3-{letter}-Output.txt"
        answer_file = OUTPUT_ROOT / f"PA3-{letter}-Debug-Answer.txt"
        
        if not output_file.exists():
            pytest.skip(f"Output file {output_file} does not exist")
        if not answer_file.exists():
            pytest.skip(f"Answer file {answer_file} does not exist")
        
        _compare_output_files(output_file, answer_file, abs_tol=abs_tol, coord_tol=coord_tol)
    
    @pytest.mark.parametrize(("letter", "abs_tol", "coord_tol"), DEBUG_DATASETS)
    def test_output_files_against_debug_outputs(self, letter: str, abs_tol: float, coord_tol: float):
        """Verify our outputs match the historical Debug-Output artifacts. Confirms the content equality upholds the expected tolerances."""
        output_file = OUTPUT_ROOT / f"pa3-{letter}-Output.txt"
        debug_output_file = OUTPUT_ROOT / f"PA3-{letter}-Debug-Output.txt"
        
        if not output_file.exists():
            pytest.skip(f"Output file {output_file} does not exist")
        if not debug_output_file.exists():
            pytest.skip(f"Debug output file {debug_output_file} does not exist")
        
        _compare_output_files(output_file, debug_output_file, abs_tol=abs_tol, coord_tol=coord_tol)
    
    @pytest.mark.parametrize(("letter", "abs_tol", "coord_tol"), DEBUG_DATASETS)
    def test_debug_outputs_against_debug_answers(self, letter: str, abs_tol: float, coord_tol: float):
        """Verify the provided Debug-Output files align with the Debug-Answer references. Confirms the comparison passes within the tolerance window as a sanity check."""
        debug_output_file = OUTPUT_ROOT / f"PA3-{letter}-Debug-Output.txt"
        answer_file = OUTPUT_ROOT / f"PA3-{letter}-Debug-Answer.txt"
        
        if not debug_output_file.exists():
            pytest.skip(f"Debug output file {debug_output_file} does not exist")
        if not answer_file.exists():
            pytest.skip(f"Answer file {answer_file} does not exist")
        
        _compare_output_files(debug_output_file, answer_file, abs_tol=abs_tol, coord_tol=coord_tol)
    
    @pytest.mark.parametrize(("letter", "abs_tol", "coord_tol"), DEBUG_DATASETS)
    def test_computed_results_against_debug_answers(self, letter: str, abs_tol: float, coord_tol: float):
        """Validate freshly computed matches against Debug-Answer files. Confirms each computed record reproduces the expected sample, closest point, and distance."""
        # Load data and compute
        mesh_vertices, mesh_triangles = load_mesh(DATA_ROOT / "Problem3Mesh.sur")
        body_a = load_rigid_body(DATA_ROOT / "Problem3-BodyA.txt")
        body_b = load_rigid_body(DATA_ROOT / "Problem3-BodyB.txt")
        samples, _ = load_samples(
            DATA_ROOT / f"PA3-{letter}-Debug-SampleReadingsTest.txt",
            num_a=body_a.markers.shape[0],
            num_b=body_b.markers.shape[0],
        )
        results = compute_matches(mesh_vertices, mesh_triangles, body_a, body_b, samples)
        
        # Load expected
        answer_file = OUTPUT_ROOT / f"PA3-{letter}-Debug-Answer.txt"
        if not answer_file.exists():
            answer_file = DATA_ROOT / f"PA3-{letter}-Debug-Answer.txt"
        if not answer_file.exists():
            pytest.skip(f"Answer file for {letter} does not exist")
        
        expected = _load_output_file(answer_file)
        
        # Compare
        assert len(results) == len(expected), \
            f"Mismatch in count: {len(results)} vs {len(expected)}"
        
        for i, (result, (exp_sample, exp_closest, exp_distance)) in enumerate(zip(results, expected)):
            rounded_sample = np.round(result.sample_point, 2)
            rounded_closest = np.round(result.closest_point, 2)
            
            assert np.allclose(rounded_sample, exp_sample, atol=coord_tol), \
                f"Row {i}: Sample point mismatch: {rounded_sample} vs {exp_sample}"
            assert np.allclose(rounded_closest, exp_closest, atol=coord_tol), \
                f"Row {i}: Closest point mismatch: {rounded_closest} vs {exp_closest}"
            assert math.isclose(result.distance, exp_distance, abs_tol=abs_tol), \
                f"Row {i}: Distance mismatch: {result.distance} vs {exp_distance}"
    
    @pytest.mark.parametrize("letter", DEBUG_LETTERS)
    def test_output_file_format(self, letter: str):
        """Verify each output file adheres to the required textual structure. Confirms headers and rows contain the expected counts and numeric tokens."""
        output_file = OUTPUT_ROOT / f"pa3-{letter}-Output.txt"
        
        if not output_file.exists():
            pytest.skip(f"Output file {output_file} does not exist")
        
        with output_file.open("r", encoding="utf-8") as fh:
            # Check header
            header = fh.readline().strip()
            header_parts = header.split()
            assert len(header_parts) >= 2, f"Header should have at least 2 parts: {header}"
            count = int(header_parts[0])
            assert count > 0, f"Count should be positive: {count}"
            
            # Check data rows
            row_count = 0
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                assert len(tokens) >= 7, f"Row {row_count} should have at least 7 tokens: {line}"
                # Verify all tokens are numeric
                for token in tokens:
                    float(token)  # Should not raise
                row_count += 1
            
            assert row_count == count, \
                f"Row count mismatch: header says {count}, found {row_count}"
    
    @pytest.mark.parametrize("letter", DEBUG_LETTERS)
    def test_output_file_exists(self, letter: str):
        """Verify every debug output file has been generated. Confirms each file exists on disk and is non-empty."""
        output_file = OUTPUT_ROOT / f"pa3-{letter}-Output.txt"
        assert output_file.exists(), f"Output file {output_file} does not exist"
        
        # Verify it's not empty
        assert output_file.stat().st_size > 0, f"Output file {output_file} is empty"
    
    @pytest.mark.parametrize("letter", ALL_SAMPLE_LETTERS)
    def test_sample_readings_file_exists(self, letter: str):
        """Verify that all sample readings inputs are available. Confirms each expected file exists either in output/ or the data directory and is non-empty."""
        suffix = "Unknown" if letter in UNKNOWN_LETTERS else "Debug"
        sample_file = OUTPUT_ROOT / f"PA3-{letter}-{suffix}-SampleReadingsTest.txt"
        if not sample_file.exists():
            sample_file = DATA_ROOT / f"PA3-{letter}-{suffix}-SampleReadingsTest.txt"
        
        assert sample_file.exists(), \
            f"Sample readings file for {letter} does not exist in output/ or data directory"
        assert sample_file.stat().st_size > 0, \
            f"Sample readings file {sample_file} is empty"

    @pytest.mark.parametrize("letter", UNKNOWN_LETTERS)
    def test_unknown_datasets_compute_matches(self, letter: str, tmp_path: Path):
        """Smoke-test the pipeline against unknown datasets G, H, and J. Confirms the computed matches are finite, serialized cleanly, and round-trip through the formatter."""
        suffix = "Unknown"
        sample_file = OUTPUT_ROOT / f"PA3-{letter}-{suffix}-SampleReadingsTest.txt"
        if not sample_file.exists():
            sample_file = DATA_ROOT / f"PA3-{letter}-{suffix}-SampleReadingsTest.txt"
        assert sample_file.exists(), f"Sample readings file for PA3-{letter} not found"

        _, frame_count = _read_sample_header(sample_file)

        mesh_vertices, mesh_triangles = load_mesh(DATA_ROOT / "Problem3Mesh.sur")
        body_a = load_rigid_body(DATA_ROOT / "Problem3-BodyA.txt")
        body_b = load_rigid_body(DATA_ROOT / "Problem3-BodyB.txt")
        samples, _ = load_samples(
            sample_file,
            num_a=body_a.markers.shape[0],
            num_b=body_b.markers.shape[0],
        )

        assert len(samples) == frame_count, \
            f"Expected {frame_count} frames, got {len(samples)} for PA3-{letter}"

        results = compute_matches(mesh_vertices, mesh_triangles, body_a, body_b, samples)
        assert len(results) == frame_count

        for idx, result in enumerate(results):
            assert result.sample_point.shape == (3,), f"Sample point shape mismatch at row {idx}"
            assert result.closest_point.shape == (3,), f"Closest point shape mismatch at row {idx}"
            assert np.all(np.isfinite(result.sample_point)), f"Non-finite sample point at row {idx}"
            assert np.all(np.isfinite(result.closest_point)), f"Non-finite closest point at row {idx}"
            assert math.isfinite(result.distance), f"Non-finite distance at row {idx}"
            assert result.distance >= 0.0, f"Negative distance at row {idx}"

        # Ensure we can serialize the results in the expected PA3 output format.
        from src.output import format_output_line

        temp_output = tmp_path / f"pa3-{letter}-Unknown-Output.txt"
        with temp_output.open("w", encoding="utf-8") as fh:
            fh.write(f"{len(results)} {temp_output.name} 0\n")
            for result in results:
                fh.write(
                    format_output_line(result.sample_point, result.closest_point, result.distance)
                )
                fh.write("\n")

        saved_rows = _load_output_file(temp_output)
        assert len(saved_rows) == len(results)


class TestOutputFileConsistency:
    """Tests to ensure consistency across all output files."""
    
    def test_all_output_files_have_same_structure(self):
        """Verify all debug output files share the same structural format. Confirms headers and row counts remain consistent across datasets."""
        for letter in DEBUG_LETTERS:
            output_file = OUTPUT_ROOT / f"pa3-{letter}-Output.txt"
            if not output_file.exists():
                continue
            
            with output_file.open("r", encoding="utf-8") as fh:
                header = fh.readline().strip()
                header_parts = header.split()
                assert len(header_parts) >= 2, \
                    f"File {letter}: Header should have at least 2 parts"
                
                # Count non-empty lines
                data_lines = [line.strip() for line in fh if line.strip()]
                assert len(data_lines) == int(header_parts[0]), \
                    f"File {letter}: Row count mismatch"
                
                # Verify all data lines have same format
                for i, line in enumerate(data_lines):
                    tokens = line.split()
                    assert len(tokens) >= 7, \
                        f"File {letter}, row {i}: Should have at least 7 tokens"
    
    def test_output_files_match_computed_results(self):
        """Validate that saved outputs equal newly recomputed results. Confirms each dataset's file can be regenerated bit-for-bit within tolerance."""
        for letter in DEBUG_LETTERS:
            output_file = OUTPUT_ROOT / f"pa3-{letter}-Output.txt"
            if not output_file.exists():
                continue
            
            # Compute fresh results
            mesh_vertices, mesh_triangles = load_mesh(DATA_ROOT / "Problem3Mesh.sur")
            body_a = load_rigid_body(DATA_ROOT / "Problem3-BodyA.txt")
            body_b = load_rigid_body(DATA_ROOT / "Problem3-BodyB.txt")
            
            sample_file = OUTPUT_ROOT / f"PA3-{letter}-Debug-SampleReadingsTest.txt"
            if not sample_file.exists():
                sample_file = DATA_ROOT / f"PA3-{letter}-Debug-SampleReadingsTest.txt"
            
            samples, _ = load_samples(
                sample_file,
                num_a=body_a.markers.shape[0],
                num_b=body_b.markers.shape[0],
            )
            computed_results = compute_matches(mesh_vertices, mesh_triangles, body_a, body_b, samples)
            
            # Load saved output
            saved_output = _load_output_file(output_file)
            
            # Compare (with appropriate tolerance)
            abs_tol = 2e-2 if letter in ["D", "E", "F"] else 1e-2
            coord_tol = 2e-2 if letter in ["D", "E", "F"] else 1e-2
            
            assert len(computed_results) == len(saved_output), \
                f"Dataset {letter}: Count mismatch"
            
            for i, (result, (saved_sample, saved_closest, saved_dist)) in enumerate(
                zip(computed_results, saved_output)
            ):
                rounded_sample = np.round(result.sample_point, 2)
                rounded_closest = np.round(result.closest_point, 2)
                
                assert np.allclose(rounded_sample, saved_sample, atol=coord_tol), \
                    f"Dataset {letter}, row {i}: Sample mismatch"
                assert np.allclose(rounded_closest, saved_closest, atol=coord_tol), \
                    f"Dataset {letter}, row {i}: Closest mismatch"
                assert math.isclose(result.distance, saved_dist, abs_tol=abs_tol), \
                    f"Dataset {letter}, row {i}: Distance mismatch"

    def test_logfile_mentions_all_datasets(self):
        """Verify PA3-Logfile documents all datasets from A through J. Confirms each expected token string is present in the logfile content."""
        log_file = OUTPUT_ROOT / "PA3-Logfile.txt"
        assert log_file.exists(), "Expected PA3-Logfile.txt in output/"
        content = log_file.read_text(encoding="utf-8")

        expected_tokens = [
            "PA3-A-Debug",
            "PA3-B-Debug",
            "PA3-C-Debug",
            "PA3-D-Debug",
            "PA3-E-Debug",
            "PA3-F-Debug",
            "PA3-G-Unknown",
            "PA3-H-Unknown",
            "PA3-J-Unknown",
        ]
        for token in expected_tokens:
            assert token in content, f"{token} missing from PA3-Logfile.txt"

