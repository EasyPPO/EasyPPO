# Experiment status

## AM critic training — RERUN REQUIRED (2026-11-03)

Bugs were found during AM critic training and have now been fixed. **The AM
critic training experiment must be rerun with the fixed native Miles code.**
Start a fresh run from the intended pretrained initialization using the existing
frozen trajectories and replay; do not resume a pre-fix AM checkpoint. Keep old
AM runs as superseded records, and regenerate results derived from them.
The local 0.8B checks do not replace this production training rerun.

