"""Setup script for pointer-to-ct-icp package."""

from setuptools import find_packages, setup

setup(
    name="pointer-to-ct-icp",
    version="1.0.0",
    description="PA3 matching phase for a simplified ICP pipeline",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pa3=pa3:main",
        ],
    },
)

