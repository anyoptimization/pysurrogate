"""pysurrogate's pyclawd config — drives `pyclawd test/lint/typecheck/...` for this repo."""

from pyclawd import DoctorConfig, GoldenConfig, Project, QualityConfig, TestConfig

project = Project(
    name="pysurrogate",
    conda_env=None,
    root_markers=["src/pysurrogate/__init__.py"],
    # The pyclawd this config was built on. `pyclawd doctor` WARNs if the
    # running pyclawd has drifted to a different minor (migration may be needed).
    pyclawd_version="0.1.1",
    # Default directory `pyclawd ls` lists (the code/source root).
    src_dir="src",
    quality=QualityConfig(
        lint_cmd=["ruff", "check"],
        lint_fix_cmd=["ruff", "check", "--fix"],
        format_cmd=["ruff", "format"],
        format_check_cmd=["ruff", "format", "--check", "--quiet"],
        typecheck_cmd=["mypy"],
        check_sequence=["format-check", "lint", "typecheck", "descriptions", "test"],
    ),
    test=TestConfig(
        tests_dir="tests/",
        classname_prefix="tests.",
        integration_files=[],
        # golden is its own tier (agents compare, humans bless) — keep it out of
        # the unit tiers so `pyclawd test` never tries to re-bless baselines.
        markers={
            "fast": "not slow and not integration and not golden",
            "default": "not slow and not golden",
            "all": "not golden",
        },
    ),
    golden=GoldenConfig(baseline_dir="tests/dace/golden", marker="golden"),
    doctor=DoctorConfig(
        core_deps=["numpy", "scipy", "scikit-learn", "pandas"],
        dev_deps=["pytest", "pytest-xdist", "pytest-cov"],
        tool_files=[],
        binaries=[
            ("ruff", "pip install ruff"),
            ("mypy", "pip install mypy"),
        ],
    ),
)
