"""The lifecycle-file tamper table (shared by the device definition and its host logical controls). Each case: `(name, edit kwargs, expected-refusal regex, seal)` where `edit` mutates the OUTER manifest (`outer`) or the CHILD manifest
(`child`) dict, and `seal` is `"own"` (the forged file's own, validly recomputed seal: a SEMANTIC refusal is expected) or `"writer"` (the seal the original writer returned: the FILE-IDENTITY refusal is expected). A recomputed seal is
not authenticity; the table never promises adversarial authentication."""
from __future__ import annotations

from maple_syrup import floating_lifecycle_checkpoint as cpul


def lifecycle_cases(D: dict, windowed: bool) -> list:
    """`D` is the ORIGINAL `lifecycle` record of the file (so every edit is relative to real values and provably changes the data)."""
    return [
        ("profile_cpu", {"outer": lambda m: m.update(profile=cpul.PROFILE)}, "unknown lifecycle profile", "own"),
        ("lifecycle_kind", {"outer": lambda m: m["lifecycle"].update(kind="boundary")}, "boundary record|keys|kind", "own"),
        ("routing_generation", {"outer": lambda m: m["lifecycle"]["routing"].update(generation=D["routing"]["generation"] + 1)}, "routing", "own"),
        ("bridge_stable", {"outer": lambda m: m["lifecycle"]["bridge"].update(stable_sha256="0" * 64)}, "differs|bridge", "own"),
        ("daily_cursor_ordinal", {"outer": lambda m: m["lifecycle"]["daily"]["cursor"].update(ordinal=D["daily"]["cursor"]["ordinal"] + 1)}, "digest|cursor", "own"),  # the CPU S1 contract's cursor control
        ("seq_negative_own_seal", {"outer": lambda m: m["lifecycle"]["head"].update(seq=-1)}, "seq|head record|cannot be restored", "own"),
        ("seq_plus5_writer_seal", {"outer": lambda m: m["lifecycle"]["head"].update(seq=D["head"]["seq"] + 5)}, "EXPECTED", "writer"),
        ("kernel", {"child": lambda m: m["kernel"].update(fmc_source_sha256="0" * 64)}, "kernel|not bound", "own"),
        ("source", {"outer": lambda m: m["source"].update(maple_package_sha256="0" * 64)}, "source pair", "own"),
        ("child_schema", {"outer": lambda m: m["child"].update(schema=("maple-syrup-floating-pond-checkpoint/8" if windowed else "maple-syrup-floating-pond-checkpoint/7"))}, "child schema", "own"),
        ("backend", {"outer": lambda m: m.update(array_backend="numpy")}, "unsupported backend", "own"),
    ]
