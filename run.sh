#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
SSL_CERT="$SCRIPT_DIR/.venv/lib/python3.10/site-packages/certifi/cacert.pem"

usage() {
    cat <<'EOF'
Photo Curation Pipeline

Usage:
  ./run.sh stage1 <album-dir> [flags]                  Stage 1: quality filtering
  ./run.sh stage2 <stage1-dir> [flags]                 Stage 2: VLM scene ranking (Ollama)
  ./run.sh stage3 <portfolio-dir> [flags]              Stage 3: Vertex AI improvement
  ./run.sh fix <image> [images...] [flags]             Quick-fix individual images via Vertex
  ./run.sh check                                       Verify environment health

Quick-fix examples:
  ./run.sh fix photo.JPG                               Default auto-prompt
  ./run.sh fix photo.JPG --gemma                       Gemma analyzes image + generates prompt
  ./run.sh fix photo.JPG --gemma --prompt "text"       Gemma + your custom prompt merged
  ./run.sh fix photo.JPG --gemma --prompt "text" --selective-edit   Inpainting mode
  ./run.sh fix photo.JPG --gemma --prompt "text" --ultra-preserve   Museum-quality preservation

Stage flags are passed through to the underlying scripts. Use --help on any command.
EOF
}

run_check() {
    local pass=0
    local fail=0

    echo "=== Environment Check ==="
    echo ""

    # 1. System python3
    printf "  System python3 ... "
    if python3 -c "import cv2, open_clip; print('OK')" 2>/dev/null; then
        ((pass++))
    else
        echo "FAIL (missing cv2 or open_clip)"
        ((fail++))
    fi

    # 2. Venv python3
    printf "  Venv python3 ... "
    if [ -x "$VENV_PYTHON" ]; then
        echo "OK ($VENV_PYTHON)"
        ((pass++))
    else
        echo "FAIL (not found at $VENV_PYTHON)"
        ((fail++))
    fi

    # 3. SSL cert
    printf "  SSL certificate ... "
    if [ -f "$SSL_CERT" ]; then
        echo "OK"
        ((pass++))
    else
        echo "FAIL (not found at $SSL_CERT)"
        ((fail++))
    fi

    # 4. Venv imports
    printf "  Venv imports ... "
    if SSL_CERT_FILE="$SSL_CERT" "$VENV_PYTHON" -c "import google.auth, PIL, pillow_heif, certifi; print('OK')" 2>/dev/null; then
        ((pass++))
    else
        echo "FAIL"
        ((fail++))
    fi

    # 5. Stage 1 import
    printf "  Stage 1 (pipeline) ... "
    if python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from pipeline import CurationPipeline; print('OK')" 2>/dev/null; then
        ((pass++))
    else
        echo "FAIL"
        ((fail++))
    fi

    # 6. Stage 2 import
    printf "  Stage 2 (scene family) ... "
    if python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); import stage2_scene_family_pipeline; print('OK')" 2>/dev/null; then
        ((pass++))
    else
        echo "FAIL"
        ((fail++))
    fi

    # 7. Vertex quick-fix import
    printf "  Quick-fix (vertex) ... "
    if SSL_CERT_FILE="$SSL_CERT" "$VENV_PYTHON" -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from vertex_quick_fix import gemma_analyze_image; print('OK')" 2>/dev/null; then
        ((pass++))
    else
        echo "FAIL"
        ((fail++))
    fi

    # 8. Ollama
    printf "  Ollama (gemma4:31b) ... "
    if curl -s --max-time 5 http://localhost:11434/api/tags 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = [m['name'] for m in data.get('models', [])]
if any('gemma4' in m for m in models):
    print('OK (' + ', '.join(models) + ')')
else:
    print('WARN: gemma4:31b not found (have: ' + ', '.join(models) + ')')
    sys.exit(1)
" 2>/dev/null; then
        ((pass++))
    else
        echo "FAIL (Ollama not running or gemma4:31b not pulled)"
        ((fail++))
    fi

    # 9. Google Cloud credentials
    printf "  Google Cloud auth ... "
    if SSL_CERT_FILE="$SSL_CERT" "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from vertex_ranked_photo_improver2 import load_vertex_credentials
_, pid = load_vertex_credentials(None)
print(f'OK (project={pid})')
" 2>/dev/null; then
        ((pass++))
    else
        echo "FAIL (run: gcloud auth application-default login)"
        ((fail++))
    fi

    echo ""
    echo "=== $pass passed, $fail failed ==="
    return $fail
}

# --- Main dispatch ---

if [ $# -eq 0 ]; then
    usage
    exit 0
fi

CMD="$1"
shift

case "$CMD" in
    stage1)
        cd "$SCRIPT_DIR"
        if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
            exec python3 "$SCRIPT_DIR/main.py" --help
        fi
        exec python3 "$SCRIPT_DIR/main.py" --local-root "$@"
        ;;
    stage2)
        cd "$SCRIPT_DIR"
        if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
            exec python3 "$SCRIPT_DIR/stage2_scene_family_pipeline.py" --help
        fi
        exec python3 "$SCRIPT_DIR/stage2_scene_family_pipeline.py" --stage1-dir "$@"
        ;;
    stage3)
        cd "$SCRIPT_DIR"
        if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
            exec env SSL_CERT_FILE="$SSL_CERT" "$VENV_PYTHON" "$SCRIPT_DIR/vertex_ranked_photo_improver2.py" --help
        fi
        exec env SSL_CERT_FILE="$SSL_CERT" "$VENV_PYTHON" "$SCRIPT_DIR/vertex_ranked_photo_improver2.py" --portfolio-dir "$@"
        ;;
    fix)
        if [ $# -eq 0 ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
            exec env SSL_CERT_FILE="$SSL_CERT" "$VENV_PYTHON" "$SCRIPT_DIR/vertex_quick_fix.py" --help
        fi
        # Collect image paths (everything before the first --flag)
        IMAGES=()
        EXTRA_ARGS=()
        in_images=true
        while [ $# -gt 0 ]; do
            case "$1" in
                --gemma)
                    in_images=false
                    EXTRA_ARGS+=("--gemma-analyze")
                    shift
                    ;;
                --ultra)
                    in_images=false
                    EXTRA_ARGS+=("--ultra-preserve")
                    shift
                    ;;
                --*)
                    in_images=false
                    EXTRA_ARGS+=("$1")
                    shift
                    ;;
                *)
                    if $in_images; then
                        IMAGES+=("$1")
                    else
                        EXTRA_ARGS+=("$1")
                    fi
                    shift
                    ;;
            esac
        done
        cd "$SCRIPT_DIR"
        if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
            exec env SSL_CERT_FILE="$SSL_CERT" "$VENV_PYTHON" "$SCRIPT_DIR/vertex_quick_fix.py" \
                --images "${IMAGES[@]}"
        else
            exec env SSL_CERT_FILE="$SSL_CERT" "$VENV_PYTHON" "$SCRIPT_DIR/vertex_quick_fix.py" \
                --images "${IMAGES[@]}" "${EXTRA_ARGS[@]}"
        fi
        ;;
    check)
        run_check
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $CMD"
        echo ""
        usage
        exit 1
        ;;
esac
