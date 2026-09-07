#!/usr/bin/env bash
if [ ! -z "${DEBUG}" ]; then
    set -x
fi

USER_ID=$(stat -c %u /work)
GROUP_ID=$(stat -c %g /work)

groupadd -g $GROUP_ID usergroup 2>/dev/null
useradd -m -l -u $USER_ID -g usergroup user 2>/dev/null

if [ ! -z "${DEBUG}" ]; then
    env
    whoami
    python -c "import pymesh; print('PyMesh OK')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    gosu user whoami
fi

exec gosu user "$@"
