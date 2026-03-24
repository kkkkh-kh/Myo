from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent
README_PATH = PROJECT_ROOT / "README.md"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

install_requires = [
    line.strip()
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="sign-language-translator",
    version="0.1.0",
    description="Chinese sign language gloss to natural Chinese sentence translator",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="OpenAI Codex",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    package_data={"data": ["reorder_rules.json"]},
    install_requires=install_requires,
    python_requires=">=3.10,<3.12",
    entry_points={"console_scripts": ["sign-translate=inference.translate:main"]},
)