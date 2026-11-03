#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if [[ "$(uname -s)" != "Darwin" ]]; then
	printf 'run_mac.sh is intended for macOS.\n' >&2
	exit 1
fi

for dependency in curl ffmpeg ffprobe; do
	if ! command -v "$dependency" >/dev/null 2>&1; then
		printf 'Required command not found: %s. Install it before launching FaceFusion.\n' "$dependency" >&2
		exit 1
	fi
done

python_bin="${PYTHON:-python3}"
if [[ -z "${PYTHON:-}" && ( -n "${CONDA_PREFIX:-}" || -n "${VIRTUAL_ENV:-}" ) ]]; then
	python_bin='python'
fi

if command -v "$python_bin" >/dev/null 2>&1 \
	&& "$python_bin" -c 'import onnxruntime as ort; assert "CoreMLExecutionProvider" in ort.get_available_providers()' >/dev/null 2>&1 \
	&& "$python_bin" facefusion.py run --help >/dev/null 2>&1; then
	exec "$python_bin" facefusion.py run "$@" --execution-providers coreml cpu --open-browser
fi

if command -v uv >/dev/null 2>&1; then
	printf 'Python environment %s is not ready with CoreML and FaceFusion dependencies; starting with uv.\n' "$python_bin" >&2
	exec uv run \
		--isolated \
		--no-project \
		--python 3.12 \
		--with-requirements "$project_dir/requirements.txt" \
		python facefusion.py run "$@" --execution-providers coreml cpu --open-browser
fi

printf 'Python environment %s is not ready with CoreML and FaceFusion dependencies, and uv is unavailable.\n' "$python_bin" >&2
printf 'Install uv or activate a Python environment with FaceFusion dependencies and ONNX Runtime CoreML support.\n' >&2
exit 1
