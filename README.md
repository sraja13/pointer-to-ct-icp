# pointer-to-ct-icp

PA3 matching phase for a simplified ICP (Iterative Closest Point) pipeline. This assignment computes pointer tip positions for each sample frame and locates the closest point on the mesh surface using rigid body transformations and geometric calculations.

[Here](https://ciis.lcsr.jhu.edu/doku.php?id=courses:455-655:455-655) is the course page.

## Overview

In computer integrated surgery, accurate alignment between the patient's anatomy and preoperative images is essential for precise navigation and guidance. The Iterative Closest Point (ICP) algorithm is a core method used for this purpose, establishing spatial correspondence between physical measurements and a reference model. This assignment specifically focuses on implementing the **matching phase** of the ICP algorithm, where each measured point is paired with its associated nearest point on the reference surface.

### Problem Setup

The system uses two rigid bodies tracked by an optical tracker:
- **Body A**: Acts as a pointer with a known tip position in Body A's coordinate frame
- **Body B**: Rigidly attached to the bone, defining the bone's coordinate frame

The optical tracker measures LED marker positions on both bodies in the tracker coordinate frame, while each body's local marker geometry is known. By combining these measurements, we can:
1. Determine each body's position relative to the tracker for any frame
2. Express the pointer tip location in the bone coordinate system
3. Compare measured tip points to the CT surface mesh (since bone frame = CT coordinate system)
4. Identify the closest surface point for every measured tip position

This creates point correspondences (between measured points and closest surface mesh points) that form the foundation for full ICP registration.

If you use this template or any of the code within for your project, please cite

```bibtex
@misc{benjamindkilleen2022Sep,
 author = {Killeen, Benjamin D.},
 title = {{cispa: Template for CIS I programming assignments at Johns Hopkins}},
 journal = {GitHub},
 year = {2022},
 month = {Sep},
 url = {https://github.com/benjamindkilleen/cispa}
}
```

## Aside on SSH Keys

SSH keys are a more secure, less annoying alternative to typing your password every time you commit
your code. This allows you to commit more often, leading to more granular code updates and a better
sense of progress. When working on group projects, this is desirable to make sure everyone is
working on the same code.

[Here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
are the instructions for adding an ssh key to GitHub. If you don't know what an ssh key, first
[check to see if your computer already has
one](https://docs.github.com/en/articles/checking-for-existing-ssh-keys). If not, [generate
one](https://docs.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent),
then add it to GitHub.

## Dependencies

We recommend using [Anaconda3](https://www.anaconda.com/products/individual) to manage your environments.

- MacOS: use the command line installer. From the terminal, run

  ```sh
  wget https://repo.anaconda.com/archive/Anaconda3-2021.05-MacOSX-x86_64.sh
  sh Anaconda3-2021.05-MacOSX-x86_64.sh
  ```

  and follow the install instructions with `~/anaconda3` as the install location (unless you really know what you're doing).

- Linux: similarly:

  ```sh
  wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
  sh Anaconda3-2021.05-Linux-x86_64.sh
  ```

- Windows: Install Anaconda through the GUI, and then I recommend using a linux-like terminal.

## Install

In a terminal, clone this repo:

```sh
git clone git@github.com:YOUR_USERNAME/pointer-to-ct-icp.git
# Or download and extract the zip file
```

Then change into the directory, create the Anaconda environment, and activate it.

```bash
cd pointer-to-ct-icp
conda env create -f environment.yml
conda activate pointer-to-ct-icp
```

Alternatively, if you prefer using pip:

```bash
pip install -r requirements.txt
```

The pip requirements install `numpy`, `scipy`, and `pytest` for local development.

## Mathematical Approach

The goal is to compute, for every sample frame k, the pointer tip position in the bone coordinate system and determine the closest corresponding point on the CT mesh surface.

### Part 1: Rigid Body Registration

For each frame k, the optical tracker measures marker positions `a_i,k` and `b_i,k` in the tracker coordinate system. The body-frame marker coordinates `A_i` and `B_i` are known. We compute transformations:

```
a_i,k = F_A,k * A_i = [R_A,k | t_A,k] * A_i
b_i,k = F_B,k * B_i = [R_B,k | t_B,k] * B_i
```

where `R_A,k` and `R_B,k` are 3×3 rotation matrices, and `t_A,k` and `t_B,k` are translation vectors.

**SVD-based Absolute Orientation:**
The optimal rotation and translation minimize:
```
min_{R,t} Σ_i ||R*A_i + t - a_i,k||²
```

**Algorithm:**
1. Compute centroids: `Ā` and `ā_k` for body A markers
2. Form centered point sets: `A_i' = A_i - Ā` and `a_i,k' = a_i,k - ā_k`
3. Compute cross-covariance: `H = Σ_i A_i' * (a_i,k')ᵀ`
4. SVD decomposition: `H = U*S*Vᵀ`
5. Optimal rotation: `R = V*Uᵀ` (if det(R) < 0, flip last column of V)
6. Optimal translation: `t = ā_k - R*Ā`

### Part 2: Pointer Tip Transformation

The pointer tip position `A_tip` in Body A's frame is known. Its position in Body B's (bone) coordinate frame is:

```
d_k = F_B,k⁻¹ * F_A,k * A_tip
```

### Part 3: CT Coordinate Registration

For this assignment, the bone coordinate system is assumed perfectly aligned with the CT coordinate system (`F_reg = I`), so `d_k` is already in CT coordinates.

### Part 4: Surface Mesh Closest Point

For each pointer tip `d_k`, find the closest point on the mesh `c_k` by:

```
c_k = arg min_{x ∈ mesh} ||d_k - x||
```

The correspondence distance is: `Error_k = ||d_k - c_k||`

## Algorithmic Approach

The implementation follows a structured workflow translating the mathematical procedures into computational steps:

### Step 1: Loading Input Data

Input files are parsed into appropriate data structures:
- **Rigid body definition files** (A and B): Parsed into `RigidBody` objects storing marker coordinates and tip location in body coordinates
- **Sample readings file**: Parsed into `SampleFrame` objects, each containing optical coordinates of Body A and Body B markers for frame k
- **CT surface mesh file**: Parsed into two NumPy arrays:
  - Vertex coordinates: shape (M, 3) where M is the number of vertices
  - Triangle indices: shape (T, 3) where T is the number of triangles

### Step 2: Computing Rigid Body Transformations

For each frame k, two transformations are computed:
- `F_A,k = [R_A,k | t_A,k]` for Body A
- `F_B,k = [R_B,k | t_B,k]` for Body B

**Implementation in `point_cloud_registration()`:**
1. Compute centroids of both point sets
2. Form centered coordinates and compute cross-covariance matrix H
3. Perform SVD: `H = U*S*Vᵀ`
4. Compute rotation: `R = V*Uᵀ`
5. If `det(R) < 0`, flip the sign of the last column of V to enforce `det(R) = 1`
6. Compute translation: `t = ā_k - R*Ā`
7. Return the homogeneous 4×4 transform matrix

### Step 3: Compute Pointer Tip in Bone Coordinates

Using the transformations from Step 2 and the known tip position `A_tip`, compute:
```
d_k = F_B,k⁻¹ * F_A,k * A_tip
```

This is computed for each frame k using matrix multiplication and inversion operations.

### Step 4: CT Frame Alignment

Since the bone coordinate frame (Body B) and CT coordinates are assumed aligned (`F_reg = I`), each pointer tip position `d_k` is already expressed in CT coordinates.

### Step 5: Finding Closest Point on Surface Mesh

For each pointer tip position `d_k`:
e1. Query the triangle KD-tree (built once during mesh load) to shortlist nearby triangles
2. For each candidate triangle, compute the closest point on its surface using geometric projection
3. Select the point `c_k` with minimum Euclidean distance: `min ||d_k - c_k||`

The KD-tree leverages triangle centroids via `scipy.spatial.cKDTree` to avoid scanning the entire mesh, while the barycentric projection still handles vertex-, edge-, and interior-contact scenarios exactly.

### Step 6: Outputting Results

For each frame, write to the output file:
- Pointer tip coordinates in bone coordinate frame: `d_k`
- Corresponding closest point on mesh surface: `c_k`
- Euclidean distance: `Error_k = ||d_k - c_k||`

## Usage

This program implements the matching phase algorithm through the following steps:

1. **Loads input data**: Mesh vertices/triangles, rigid body definitions (markers and tip positions), and sample readings
2. **Computes rigid body transformations**: Uses SVD-based point cloud registration for each frame
3. **Transforms pointer tip**: Applies transformations to compute tip positions in bone/CT coordinates
4. **Finds closest points**: Uses a KD-tree accelerator to narrow the triangle set, then runs exact projections to find minimum Euclidean distance
5. **Outputs correspondences**: Writes tip coordinates, closest mesh points, and distances

### Running the Program

Run the program with the following command-line arguments:

```bash
python3 -m src.pa3 --mesh <mesh_file> --body-a <body_a_file> --body-b <body_b_file> --samples <samples_file> --output <output_file>
```

**Required arguments:**
- `--mesh`: Path to mesh file (`.sur` format with vertices and triangles)
- `--body-a`: Path to rigid body A definition file (markers and tip position)
- `--body-b`: Path to rigid body B definition file (markers and tip position)
- `--samples`: Path to sample readings file (marker positions for each frame)
- `--output`: Path to output file where results will be written

**Example usage:**

```bash
python3 -m src.pa3 \
  --mesh "Problem3Mesh.sur" \
  --body-a "Problem3-BodyA.txt" \
  --body-b "Problem3-BodyB.txt" \
  --samples "PA3-A-Debug-SampleReadingsTest.txt" \
  --output "output/pa3-A-Output.txt"
```

**Output format:**
The output file contains one line per sample frame with:
- Sample point coordinates (3 values)
- Closest point on mesh coordinates (3 values)
- Distance between sample and closest point (1 value)

### Testing

All automated tests are written with [pytest](https://docs.pytest.org/en/stable/). The suite mixes focused unit tests, dataset driven integration checks, and file iff regressions 
**Quick commands**

| Goal | Command |
| --- | --- |
| Run everything | `python3 -m pytest tests` |
| Stream stdout/stderr for debugging | `python3 -m pytest tests -s` |
| Verbose per-test reporting | `python3 -m pytest tests -v` |
| Stop after first failure (faster triage) | `python3 -m pytest tests --maxfail=1 -x` |

**Targeted runs**

```bash
# Integration test for the full PA3 pipeline (datasets A–F)
python3 -m pytest tests/test_pa3.py -v

# Similar regression checks plus output-file validation and logfile coverage
python3 -m pytest tests/test_all_output_files.py -v

# Focused unit suites
python3 -m pytest tests/test_io.py -v            # File I/O loaders and error handling
python3 -m pytest tests/test_geometry.py -v      # Triangle/mesh geometry utilities
python3 -m pytest tests/test_transforms.py -v    # Rigid-body registration and transforms
python3 -m pytest tests/test_matching.py -v      # Pointer-to-mesh matching logic
python3 -m pytest tests/test_output.py -v        # Output line formatting

# Drill into a single test 
python3 -m pytest tests/test_pa3.py::test_pa3_debug_datasets -v
```

**What each test module covers**

- `tests/test_io.py` &mdash; Success/error cases for loading meshes, rigid bodies, and sample readings, including malformed file scenarios.
- `tests/test_geometry.py` &mdash; Analytical triangle/mesh proximity checks plus validation against real PA3-A datasets.
- `tests/test_transforms.py` &mdash; Point-cloud registration, transform inversion, and transform application on synthetic setups and the PA3-A data.
- `tests/test_matching.py` &mdash; `compute_matches` behaviour for mock inputs and official PA3 debug answers.
- `tests/test_output.py` &mdash; Formatting guarantees for the PA3 output line writer (precision, ordering, sign handling).
- `tests/test_pa3.py` &mdash; End-to-end regression for datasets A through F using the refactored modular entry point.
- `tests/test_all_output_files.py` &mdash; Re-runs the full pipeline, compares every generated file against the published answers, smoke-tests unknown datasets (G/H/J), and checks that `PA3-Logfile.txt` documents every dataset.

**Coverage summary (current suite of tests)**

- 130 total tests (unit, integration, and regression) &mdash; all passing as of the latest run.
- Synthetic unit tests ensure edge cases are enforced before large dataset runs.
- Real-data tests reuse the course-provided PA3 debug files to guarantee output parity with the original script.

- Tip: add the `-s` flag when you want to surface `print` statements or logging output during a failing test run.

### Validation Results

The implementation has been validated against reference outputs for all PA3 debug datasets (A through F). Validation compares computed outputs to ground truth files, focusing on per-sample Euclidean distance values:

```
Error_k = ||d_k - c_k|| = √((d_x - c_x)² + (d_y - c_y)² + (d_z - c_z)²)
```

**Validation Metrics:**
- **Mean Error**: Average difference between computed and reference distances
- **RMS Error**: Root mean square of differences
- **Max Error**: Maximum difference across all samples

**Results Summary:**

| Dataset | # Samples | Mean Error (mm) | RMS Error (mm) | Max Error (mm) | Status |
|---------|-----------|-----------------|----------------|----------------|--------|
| A       | 15        | 0.0000          | 0.0000         | 0.0000         | Pass   |
| B       | 15        | 0.0030          | 0.0038         | 0.0080         | Pass   |
| C       | 15        | 0.0023          | 0.0032         | 0.0070         | Pass   |
| D       | 15        | 0.0028          | 0.0033         | 0.0070         | Pass   |
| E       | 15        | 0.0027          | 0.0041         | 0.0100         | Pass   |
| F       | 15        | 0.0036          | 0.0052         | 0.0130         | Pass   |

**Validation Criteria:** Datasets pass if RMS error < 0.01 mm, consistent with expected floating-point precision.

**Interpretation:**
- All datasets passed validation, confirming correctness of rigid-body registration, transformations, and surface matching
- Negligible differences indicate accurate SVD-based registration and proper coordinate frame transformations
- Small non-zero errors are attributable to floating-point arithmetic propagation and mesh resolution variations
- Errors are within tolerance and primarily due to input data characteristics rather than algorithmic issues

## Project Structure

The codebase is organized into modular components:

- `src/models.py` - Data classes (RigidBody, SampleFrame, MatchResult)
- `src/io.py` - File I/O functions (mesh loader now builds a reusable KD-tree accelerator alongside the raw arrays)
- `src/transforms.py` - Transform and registration functions
- `src/geometry.py` - Geometry utilities (triangle projections plus optional KD-tree accelerated mesh queries)
- `src/matching.py` - Main matching computation logic (reuses the mesh accelerator for fast nearest-point lookups)
- `src/output.py` - Output formatting functions
- `src/cli.py` - Command-line interface
- `src/pa3.py` - Main entry point script

