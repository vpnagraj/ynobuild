"""Synthetic labelled builds, so the modeling pipeline can run before annotation is done.

Writes to a SEPARATE database (refuses to touch one holding non-synthetic builds).
Logs are assembled from per-leaf error templates plus generic build noise, with a
few deliberately confusable lines and a small rate of label noise so the task is
not trivially separable. Scores on this data say nothing about real performance;
it only proves the plumbing works.

    ynbtriage synth-db --db /tmp/ynb-synth.db --n 600
"""
from __future__ import annotations

import random

from ..db import ensure_schema, get_conn
from ..repository import add_annotation
from ..splits import assign_splits
from ..taxonomy import seed_taxonomy

_PKGS = ["libhts-dev", "zlib1g-dev", "libcurl4-openssl-dev", "python3-dev", "libbz2-dev",
         "liblzma-dev", "r-base", "openjdk-8-jdk", "cmake", "libssl-dev", "samtools", "numpy"]
_DISTROS = [("debian", "buster"), ("debian", "stretch"), ("ubuntu", "xenial"), ("ubuntu", "bionic"),
            ("debian", "jessie"), ("ubuntu", "eoan")]
_TOOLS = ["bwa", "samtools", "bcftools", "gatk", "star", "kallisto", "salmon", "hisat2", "minimap2",
          "freebayes", "bowtie2", "spades", "megahit", "prokka", "fastp", "trimmomatic", "multiqc",
          "deeptools", "macs2", "homer", "bedtools", "vcftools", "plink", "beagle", "canu", "flye",
          "quast", "busco", "diamond", "blast", "hmmer", "mafft", "iqtree", "raxml", "beast",
          "cellranger", "scanpy", "seurat", "velocyto", "bismark", "methyldackel", "picard"]


def _v(r):
    return f"{r.randint(0, 3)}.{r.randint(0, 20)}.{r.randint(0, 9)}"


TEMPLATES = {
    "dependency_rot": [
        lambda r: f"E: Version '{_v(r)}-{r.randint(1, 5)}' for '{r.choice(_PKGS)}' was not found",
        lambda r: "Could not fetch URL https://pypi.org/simple/pip/: There was a problem confirming the ssl certificate\n"
                  "ERROR: No matching distribution found for setuptools>=40",
        lambda r: f"pip {r.randint(8, 9)}.0.1 is no longer supported\nERROR: Could not find a version that satisfies the requirement {r.choice(_PKGS)}",
        lambda r: "Solving environment: failed\nPackagesNotFoundError: The following packages are not available from current channels",
    ],
    "vanished_dependency": [
        lambda r: f"--<TS>-- https://example.org/releases/{r.choice(_TOOLS)}-{_v(r)}.tar.gz\nERROR 404: Not Found.",
        lambda r: f"curl: (22) The requested URL returned error: 404\ntar: {r.choice(_TOOLS)}.tar.gz: Cannot open: No such file or directory",
        lambda r: f"fatal: remote error: upload-pack: not our ref <HEX>\nfatal: reference is not a tree: <HEX>",
    ],
    "package_conflict": [
        lambda r: f"ERROR: Cannot install {r.choice(_PKGS)}=={_v(r)} and {r.choice(_PKGS)}=={_v(r)} because these package versions have conflicting dependencies.",
        lambda r: "UnsatisfiableError: The following specifications were found to be incompatible with each other:",
        lambda r: f"The following packages have unmet dependencies:\n {r.choice(_PKGS)} : Depends: {r.choice(_PKGS)} (>= {_v(r)}) but {_v(r)} is to be installed\nE: Unable to correct problems, you have held broken packages.",
    ],
    "parent_image_unavailable": [
        lambda r: "E: The repository 'http://deb.{0}.org/{0} {1} Release' does not have a Release file.\nE: Failed to fetch http://deb.{0}.org/{0}/dists/{1}/Release  404  Not Found".format(*r.choice(_DISTROS)),
        lambda r: "Err:1 http://archive.ubuntu.com/ubuntu {1} InRelease\n  404  Not Found\nE: The repository 'http://archive.ubuntu.com/ubuntu {1} Release' no longer has a Release file.".format(*r.choice(_DISTROS)),
        lambda r: "W: GPG error: http://security.{0}.org {1}/updates InRelease: The following signatures were invalid: EXPKEYSIG\nE: The repository is not signed.".format(*r.choice(_DISTROS)),
    ],
    "dockerfile_parse_error": [
        lambda r: f"failed to solve: dockerfile parse error on line {r.randint(1, 30)}: unknown instruction: {r.choice(['FRON', 'RUNN', 'COPPY', 'ENV='])}",
        lambda r: f"Error response from daemon: Dockerfile parse error line {r.randint(1, 30)}: unknown flag: {r.choice(['chown', 'from', 'mount'])}",
    ],
    "context_mismatch": [
        lambda r: f'failed to compute cache key: "/../{r.choice(_TOOLS)}/requirements.txt": not found',
        lambda r: "COPY failed: forbidden path outside the build context: ../src ()",
    ],
    "repo_structure_changed": [
        lambda r: f'Step {r.randint(3, 9)}/12 : COPY {r.choice(["build.sh", "environment.yml", "setup.py", "install.R"])} /opt/\nfailed to compute cache key: "/{r.choice(["build.sh", "environment.yml", "setup.py"])}": not found',
        lambda r: f"COPY failed: file not found in build context or excluded by .dockerignore: stat {r.choice(['scripts/', 'conda/', 'docker/'])}: file does not exist",
    ],
    "build_script_error": [
        lambda r: f"{r.choice(['main', 'engine', 'align', 'index'])}.c:{r.randint(10, 400)}:10: fatal error: {r.choice(['htslib/sam.h', 'zlib.h', 'bzlib.h'])}: No such file or directory\ncompilation terminated.\nmake: *** [Makefile:{r.randint(10, 90)}: all] Error 1",
        lambda r: "/usr/bin/ld: cannot find -lz\ncollect2: error: ld returned 1 exit status\nmake: *** [all] Error 2",
        lambda r: f"error: implicit declaration of function '{r.choice(['strlcpy', 'memset_s', 'getline'])}' [-Werror=implicit-function-declaration]",
        lambda r: f"Traceback (most recent call last):\n  File \"setup.py\", line {r.randint(5, 80)}\nSyntaxError: invalid syntax",
    ],
    "resource_exhaustion": [
        lambda r: "cc1plus: out of memory allocating 2147483648 bytes\nKilled",
        lambda r: "write /var/lib/docker/tmp/<HEX>: no space left on device",
        lambda r: "c++: fatal error: Killed signal terminated program cc1plus\ncompilation terminated.",
    ],
    "platform_incompatibility": [
        lambda r: "standard_init_linux.go:228: exec user process caused: exec format error",
        lambda r: "nvcc fatal : Unsupported gpu architecture 'compute_35'",
        lambda r: f"/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.{r.randint(28, 35)}' not found",
    ],
    "network_download_failure": [
        lambda r: "curl: (28) Connection timed out after 30001 milliseconds",
        lambda r: "curl: (56) Recv failure: Connection reset by peer",
        lambda r: "fatal: unable to access 'https://github.com/x/y/': The requested URL returned error: 503",
        lambda r: "ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.",
    ],
    "auth_required": [
        lambda r: "CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels.",
        lambda r: "Error response from daemon: pull access denied for private/image, repository does not exist or may require 'docker login'",
        lambda r: "HTTP request sent, awaiting response... 401 Unauthorized\nUsername/Password Authentication Failed.",
    ],
}

# Relative frequencies, roughly the corpus shape (environment_decay dominant).
WEIGHTS = {"dependency_rot": 18, "vanished_dependency": 14, "package_conflict": 8,
           "parent_image_unavailable": 20, "dockerfile_parse_error": 3, "context_mismatch": 2,
           "repo_structure_changed": 4, "build_script_error": 12, "resource_exhaustion": 3,
           "platform_incompatibility": 4, "network_download_failure": 7, "auth_required": 5}

_NOISE = [
    lambda r: f"Step {r.randint(1, 12)}/12 : RUN apt-get update && apt-get install -y {r.choice(_PKGS)}",
    lambda r: f"Get:{r.randint(1, 40)} http://deb.debian.org/debian {r.choice(_DISTROS)[1]}/main amd64 Packages [<SIZE>]",
    lambda r: "Reading package lists...",
    lambda r: "debconf: delaying package configuration, since apt-utils is not installed",
    lambda r: f"Collecting {r.choice(_PKGS)}=={_v(r)}",
    lambda r: f" ---> Running in <HEX>",
    lambda r: f"warning: {r.choice(['unused variable', 'deprecated declaration', 'comparison of integers'])} [-W{r.choice(['unused', 'deprecated', 'sign-compare'])}]",
    lambda r: "  Downloading https://files.pythonhosted.org/packages/xx/yy/pkg.whl (<SIZE>)",
    lambda r: f"#{r.randint(4, 15)} [{r.randint(2, 9)}/12] RUN make -j4",
]


def make_log(leaf: str, r: random.Random) -> str:
    lines = [r.choice(_NOISE)(r) for _ in range(r.randint(5, 40))]
    lines.append(r.choice(TEMPLATES[leaf])(r))
    if r.random() < 0.15:  # a confusable line from a different leaf, earlier in the log
        other = r.choice([k for k in TEMPLATES if k != leaf])
        lines.insert(r.randint(0, len(lines) - 1), r.choice(TEMPLATES[other])(r).splitlines()[0])
    lines += [r.choice(_NOISE)(r) for _ in range(r.randint(0, 4))]
    lines.append(f"The command '/bin/sh -c ...' returned a non-zero code: {r.choice([1, 2, 100, 127, 137])}")
    return "\n".join(lines)


def synth_db(db_path, n: int = 600, seed: int = 7400, label_noise: float = 0.05,
             prefilled_frac: float = 0.0) -> dict:
    r = random.Random(seed)
    leaves, weights = zip(*WEIGHTS.items())
    with get_conn(db_path) as conn:
        ensure_schema(conn)
        seed_taxonomy(conn)
        real = conn.execute(
            "SELECT COUNT(*) n FROM builds WHERE COALESCE(source_batch,'') != 'synthetic'"
        ).fetchone()["n"]
        if real:
            raise SystemExit(f"refusing: {db_path} holds {real} non-synthetic builds. Use a separate DB.")
        start = conn.execute("SELECT COUNT(*) n FROM builds").fetchone()["n"]
        for i in range(start, start + n):
            leaf = r.choices(leaves, weights)[0]
            log = make_log(leaf, r)
            label = r.choice(leaves) if r.random() < label_noise else leaf
            cur = conn.execute(
                """INSERT INTO builds(external_id, tool_name, tool_version, log_tail, log_line_count,
                                      build_status, source_batch)
                   VALUES(?,?,?,?,?,?,?)""",
                (f"synth-{i:05d}", r.choice(_TOOLS), _v(r), log, log.count("\n") + 1, "failed", "synthetic"),
            )
            status = "prefilled" if r.random() < prefilled_frac else "confirmed"
            add_annotation(conn, cur.lastrowid, leaf_id=label, status=status,
                           annotator="synth", source="synthetic")
    return assign_splits(db_path=db_path, seed=seed)
