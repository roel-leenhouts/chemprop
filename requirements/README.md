# requirements

This directory contains static lists of _known working_ external dependencies for historical upstream releases.

## Usage

You can recreate the dependency set for a tagged upstream release by running `pip install -r X.Y.Z_requirements.txt`.

Installing the published upstream package directly may resolve to newer external dependencies, which can introduce backward-compatibility differences relative to these pinned snapshots.

## Updating

Run `. get_requirements.sh` with Docker installed and running and all of the requirements files will be re-generated.
This only needs to be done whenever a new tagged upstream release is created and published to Docker.
