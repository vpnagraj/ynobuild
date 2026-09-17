"""Seed the DB with synthetic demo data when you don't have your real CSV yet:

  * demo_seed()  -> insert a small set of synthetic, representative builds
                    (some pre-labelled) so the UI has something to show.
"""
from __future__ import annotations

from .db import ensure_schema, get_conn
from .repository import add_annotation
from .taxonomy import seed_taxonomy

# (tool, repo_url, leaf_id or None, log_tail) — logs mirror real taxonomy signals.
_DEMO = [
    ("bwa", "https://github.com/lh3/bwa", "parent_image_unavailable",
     "Get:1 http://deb.debian.org/debian buster/main amd64 Packages\n"
     "E: The repository 'http://deb.debian.org/debian buster Release' does not have a Release file.\n"
     "E: Failed to fetch ... 404  Not Found\n"),
    ("samtools", "https://github.com/samtools/samtools", "dependency_rot",
     "Reading package lists...\n"
     "E: Version '1.9-4' for 'libhts-dev' was not found\n"),
    ("bcftools", "https://github.com/samtools/bcftools", "vanished_dependency",
     "--2024-11-02-- https://example.org/releases/bcftools-1.2.tar.bz2\n"
     "2024-11-02 ERROR 404: Not Found.\n"),
    ("gatk", "https://github.com/broadinstitute/gatk", "build_script_error",
     "gcc -O2 -c engine.c\n"
     "engine.c:88:10: fatal error: htslib/sam.h: No such file or directory\n"
     "compilation terminated.\n"
     "make: *** [engine.o] Error 1\n"),
    ("star", "https://github.com/alexdobin/STAR", "resource_exhaustion",
     "g++ -std=c++11 -O3 ...\n"
     "cc1plus: out of memory allocating 2147483648 bytes\n"
     "Killed\n"),
    ("deepvariant", "https://github.com/google/deepvariant", "platform_incompatibility",
     "Step 7/12 : RUN ./configure\n"
     "standard_init_linux.go:228: exec user process caused: exec format error\n"),
    ("kallisto", "https://github.com/pachterlab/kallisto", "network_download_failure",
     "curl -L -o hdf5.tar.gz https://support.hdfgroup.org/.../hdf5.tar.gz\n"
     "curl: (56) Recv failure: Connection reset by peer\n"
     "curl: (28) Connection timed out after 30001 ms\n"),
    ("cellranger", "https://github.com/10XGenomics/cellranger", "auth_required",
     "Collecting package metadata (repodata.json):\n"
     "CondaToSNonInteractiveError: Terms of Service have not been accepted for the "
     "following channels. Please accept ...\n"),
    ("salmon", "https://github.com/COMBINE-lab/salmon", "dockerfile_parse_error",
     "failed to solve: dockerfile parse error on line 3: unknown instruction: FRON\n"),
    ("hisat2", "https://github.com/DaehwanKimLab/hisat2", "repo_structure_changed",
     "Step 4/9 : COPY build_helper.sh /opt/\n"
     'failed to compute cache key: "/build_helper.sh": not found\n'),
    ("minimap2", "https://github.com/lh3/minimap2", None,
     "cc -O2 -Wall -c main.c\n"
     "main.c:120: warning: implicit declaration\n"
     "/usr/bin/ld: cannot find -lz\n"
     "collect2: error: ld returned 1 exit status\n"),
    ("freebayes", "https://github.com/freebayes/freebayes", None,
     "Cloning into 'vcflib'...\n"
     "fatal: unable to access 'https://github.com/vcflib/vcflib/': The requested URL returned error: 503\n"),
]


def demo_seed(n: int = 12) -> int:
    inserted = 0
    with get_conn() as conn:
        ensure_schema(conn)
        seed_taxonomy(conn)
        for i, (tool, repo, leaf, log) in enumerate(_DEMO[:n], start=1):
            ext = f"demo-{i:03d}"
            if conn.execute("SELECT 1 FROM builds WHERE external_id=?", (ext,)).fetchone():
                continue
            forge = "github" if "github.com" in repo else "other"
            cur = conn.execute(
                """INSERT INTO builds(external_id, tool_name, source_repo_url, source_forge,
                                      dockerfile_fetch_status, log_tail, log_line_count,
                                      build_status, source_batch)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (ext, tool, repo, forge, "pending", log, log.count("\n") + 1, "failed", "demo"),
            )
            bid = cur.lastrowid
            inserted += 1
            if leaf:
                add_annotation(conn, bid, leaf_id=leaf, annotator="seed", source="seed")
    return inserted