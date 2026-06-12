#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRESET_FILE="${PRESET_FILE:-$ROOT/llamacpp-models.ini}"
BENCH_BIN="${BENCH_BIN:-$ROOT/llamacpp/llama-bench}"

detect_cpu_threads() {
    local count=""

    if command -v getconf >/dev/null 2>&1; then
        count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
        if [[ "$count" =~ ^[0-9]+$ ]] && [ "$count" -gt 0 ]; then
            printf '%s' "$count"
            return
        fi
    fi

    if command -v nproc >/dev/null 2>&1; then
        count="$(nproc 2>/dev/null || true)"
        if [[ "$count" =~ ^[0-9]+$ ]] && [ "$count" -gt 0 ]; then
            printf '%s' "$count"
            return
        fi
    fi

    if command -v sysctl >/dev/null 2>&1; then
        count="$(sysctl -n hw.ncpu 2>/dev/null || true)"
        if [[ "$count" =~ ^[0-9]+$ ]] && [ "$count" -gt 0 ]; then
            printf '%s' "$count"
            return
        fi
    fi

    printf '1'
}

default_threads() {
    local max_threads="$1"
    local values=(-1)
    local candidate
    local last

    for candidate in 2 4 8; do
        if [ "$candidate" -le "$max_threads" ]; then
            values+=("$candidate")
        fi
    done

    if [ "$max_threads" -gt 8 ]; then
        for ((candidate = 12; candidate <= max_threads; candidate += 4)); do
            values+=("$candidate")
        done
    fi

    last="${values[$(( ${#values[@]} - 1 ))]}"
    if [ "$last" != "$max_threads" ]; then
        values+=("$max_threads")
    fi

    local IFS=,
    printf '%s' "${values[*]}"
}

MAX_CPU_THREADS="$(detect_cpu_threads)"

# Override these at invocation time, for example:
#   THREADS=6,7,8,9,10 REPETITIONS=5 ./bench.sh
THREADS="${THREADS:-$(default_threads "$MAX_CPU_THREADS")}"
REPETITIONS="${REPETITIONS:-3}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"
DEPTHS="${DEPTHS:-0}"
DRY_RUN="${DRY_RUN:-0}"

ini_value() {
    local section="$1"
    local key="$2"

    awk -v want_section="$section" -v want_key="$key" '
        function trim(s) {
            sub(/^[[:space:]]+/, "", s)
            sub(/[[:space:]]+$/, "", s)
            return s
        }

        /^[[:space:]]*[;#]/ || /^[[:space:]]*$/ {
            next
        }

        /^[[:space:]]*\[/ {
            current = $0
            sub(/^[[:space:]]*\[/, "", current)
            sub(/\][[:space:]]*$/, "", current)
            current = trim(current)
            next
        }

        current == want_section {
            line = $0
            sub(/[;#].*$/, "", line)
            eq = index(line, "=")
            if (eq == 0) {
                next
            }
            k = trim(substr(line, 1, eq - 1))
            v = trim(substr(line, eq + 1))
            if (k == want_key) {
                print v
            }
        }
    ' "$PRESET_FILE" | tail -n 1
}

preset_value_for() {
    local section="$1"
    local key="$2"
    local value

    value="$(ini_value "$section" "$key")"
    if [ -z "$value" ]; then
        value="$(ini_value "*" "$key")"
    fi
    printf '%s' "$value"
}

preset_value() {
    preset_value_for "$MODEL_SECTION" "$1"
}

require_preset_value() {
    local key="$1"
    local value

    value="$(preset_value "$key")"
    if [ -z "$value" ]; then
        echo "Error: required preset key '$key' not found for [$MODEL_SECTION] in $PRESET_FILE" >&2
        exit 1
    fi
    printf '%s' "$value"
}

list_model_sections() {
    awk '
        function trim(s) {
            sub(/^[[:space:]]+/, "", s)
            sub(/[[:space:]]+$/, "", s)
            return s
        }

        /^[[:space:]]*\[/ {
            section = $0
            sub(/^[[:space:]]*\[/, "", section)
            sub(/\][[:space:]]*$/, "", section)
            section = trim(section)
            if (section != "*") {
                print section
            }
        }
    ' "$PRESET_FILE"
}

model_label() {
    local section="$1"
    local alias
    local hf

    alias="$(preset_value_for "$section" alias)"
    hf="$(preset_value_for "$section" hf)"

    if [ -n "$alias" ]; then
        printf '%s [%s]' "$alias" "$section"
    elif [ -n "$hf" ]; then
        printf '%s' "$hf"
    else
        printf '%s' "$section"
    fi
}

pick_model() {
    if [ "${#MODEL_SECTIONS[@]}" -eq 0 ]; then
        echo "Error: no model sections found in $PRESET_FILE" >&2
        exit 1
    fi

    if [ -n "${MODEL_SECTION:-}" ]; then
        return
    fi

    if [ ! -t 0 ]; then
        echo "Error: MODEL_SECTION is unset and stdin is not interactive." >&2
        exit 1
    fi

    local selected=0
    local key=""
    local rest=""
    local i
    local rows=$(( ${#MODEL_SECTIONS[@]} + 2 ))

    printf '\033[?25l'
    trap 'printf "\033[?25h\n"; exit 130' INT

    while true; do
        printf 'Select a llama.cpp preset to benchmark (up/down, enter; q to quit)\n\n'
        for i in "${!MODEL_SECTIONS[@]}"; do
            if [ "$i" -eq "$selected" ]; then
                printf '  > %s\n' "$(model_label "${MODEL_SECTIONS[$i]}")"
            else
                printf '    %s\n' "$(model_label "${MODEL_SECTIONS[$i]}")"
            fi
        done

        IFS= read -rsn1 key
        case "$key" in
            "")
                break
                ;;
            q|Q)
                printf '\033[?25h\n'
                exit 0
                ;;
            $'\x1b')
                IFS= read -rsn2 -t 1 rest || true
                case "$rest" in
                    "[A")
                        if [ "$selected" -le 0 ]; then
                            selected=$(( ${#MODEL_SECTIONS[@]} - 1 ))
                        else
                            selected=$(( selected - 1 ))
                        fi
                        ;;
                    "[B")
                        selected=$(( (selected + 1) % ${#MODEL_SECTIONS[@]} ))
                        ;;
                esac
                ;;
        esac

        printf '\033[%dA' "$rows"
        for ((i = 0; i < rows; i++)); do
            printf '\033[2K\033[1B'
        done
        printf '\033[%dA' "$rows"
    done

    printf '\033[?25h\n'
    trap - INT
    MODEL_SECTION="${MODEL_SECTIONS[$selected]}"
}

bool_to_01() {
    local value

    value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"

    case "$value" in
        true|on|yes|1)
            printf '1'
            ;;
        false|off|no|0)
            printf '0'
            ;;
        *)
            printf '%s' "$1"
            ;;
    esac
}

add_value_arg() {
    local key="$1"
    local flag="$2"
    local value

    value="$(preset_value "$key")"
    if [ -n "$value" ]; then
        args+=("$flag" "$value")
        MATCHED_VALUES+=("$key=$value")
    fi
}

add_bool_arg() {
    local key="$1"
    local flag="$2"
    local value

    value="$(preset_value "$key")"
    if [ -n "$value" ]; then
        value="$(bool_to_01 "$value")"
        args+=("$flag" "$value")
        MATCHED_VALUES+=("$key=$value")
    fi
}

add_inverse_bool_arg() {
    local key="$1"
    local flag="$2"
    local value

    value="$(preset_value "$key")"
    if [ -n "$value" ]; then
        case "$(bool_to_01 "$value")" in
            1) value=0 ;;
            0) value=1 ;;
        esac
        args+=("$flag" "$value")
        MATCHED_VALUES+=("$key=$value")
    fi
}

unsupported_line() {
    local key="$1"
    local value

    value="$(preset_value "$key")"
    if [ -n "$value" ]; then
        UNSUPPORTED_VALUES+=("$key=$value")
    fi
}

shell_join() {
    local quoted=()
    local arg

    for arg in "$@"; do
        printf -v arg '%q' "$arg"
        quoted+=("$arg")
    done
    printf '%s ' "${quoted[@]}"
}

if [ ! -x "$BENCH_BIN" ]; then
    echo "Error: llama-bench not found or not executable: $BENCH_BIN" >&2
    echo "Run ./setup-llamacpp.sh first." >&2
    exit 1
fi

if [ ! -f "$PRESET_FILE" ]; then
    echo "Error: preset file not found: $PRESET_FILE" >&2
    exit 1
fi

MODEL_SECTIONS=()
while IFS= read -r section; do
    MODEL_SECTIONS+=("$section")
done < <(list_model_sections)
pick_model

MODEL_SLUG="$(printf '%s' "$MODEL_SECTION" | tr '/: ' '---' | tr -cd '[:alnum:]_.-')"
OUTPUT="${OUTPUT:-$ROOT/logs/bench-${MODEL_SLUG}-threads-$(date +%Y%m%d-%H%M%S).csv}"

HF_REPO="$(preset_value hf)"
MODEL_PATH="$(preset_value model)"

if [ -z "$HF_REPO" ] && [ -z "$MODEL_PATH" ]; then
    echo "Error: selected preset needs either 'hf' or 'model' for llama-bench." >&2
    exit 1
fi

MATCHED_VALUES=()
UNSUPPORTED_VALUES=()

args=(
    "$BENCH_BIN"
    -pg "$PROMPT_TOKENS,$GEN_TOKENS"
    --threads "$THREADS"
    --repetitions "$REPETITIONS"
    --output csv
    --progress
)

if [ -n "$HF_REPO" ]; then
    args+=(--hf-repo "$HF_REPO")
    MATCHED_VALUES+=("hf=$HF_REPO")
else
    args+=(--model "$MODEL_PATH")
    MATCHED_VALUES+=("model=$MODEL_PATH")
fi

add_value_arg hf-file --hf-file
add_value_arg hf-token --hf-token
add_value_arg batch-size --batch-size
add_value_arg ubatch-size --ubatch-size
add_value_arg cache-type-k --cache-type-k
add_value_arg cache-type-v --cache-type-v
add_value_arg n-gpu-layers --n-gpu-layers
add_value_arg gpu-layers --n-gpu-layers
add_value_arg n-cpu-moe --n-cpu-moe
add_value_arg split-mode --split-mode
add_value_arg main-gpu --main-gpu
add_bool_arg no-kv-offload --no-kv-offload
add_value_arg flash-attn --flash-attn
add_value_arg device --device
add_bool_arg mmap --mmap
add_inverse_bool_arg no-mmap --mmap
add_bool_arg direct-io --direct-io
add_bool_arg embeddings --embeddings
add_value_arg tensor-split --tensor-split
add_value_arg override-tensor --override-tensor
add_bool_arg no-op-offload --no-op-offload
add_bool_arg no-host --no-host
add_value_arg fit-target --fit-target
add_value_arg fit-ctx --fit-ctx
add_value_arg cpu-mask --cpu-mask
add_bool_arg cpu-strict --cpu-strict
add_value_arg poll --poll

if [ -n "$DEPTHS" ]; then
    args+=(--n-depth "$DEPTHS")
fi

unsupported_line alias
unsupported_line load-on-startup
unsupported_line np
unsupported_line models-max
unsupported_line reasoning
unsupported_line ctx-size
unsupported_line jinja
unsupported_line temp
unsupported_line top-p
unsupported_line top-k
unsupported_line min-p
unsupported_line cpu-moe
unsupported_line spec-type
unsupported_line spec-draft-n-max
unsupported_line spec-ngram-mod-n-match
unsupported_line spec-ngram-mod-n-min
unsupported_line spec-ngram-mod-n-max
unsupported_line image-min-tokens
unsupported_line image-max-tokens
unsupported_line cache-ram
unsupported_line ctx-checkpoints
unsupported_line slot-prompt-similarity

cat >&2 <<EOF
Benchmarking [$MODEL_SECTION]
Preset: $PRESET_FILE
Output: $OUTPUT

llama-bench values matched from preset:
EOF

for value in "${MATCHED_VALUES[@]}"; do
    printf '  %s\n' "$value" >&2
done

cat >&2 <<EOF

Thread sweep:
  threads=$THREADS
  detected_cpu_threads=$MAX_CPU_THREADS
  repetitions=$REPETITIONS
  pg=$PROMPT_TOKENS,$GEN_TOKENS
  depth=${DEPTHS:-<llama-bench default>}

EOF

if [ "${#UNSUPPORTED_VALUES[@]}" -gt 0 ]; then
    cat >&2 <<EOF
Not passed because llama-bench does not support these llama-server preset keys:
EOF
    for value in "${UNSUPPORTED_VALUES[@]}"; do
        printf '  %s\n' "$value" >&2
    done
    printf '\n' >&2
fi

printf 'Command:\n  %s%s\n\n' "$(shell_join "${args[@]}")" "$(shell_join "$@")" >&2

if [ "$DRY_RUN" = "1" ]; then
    exit 0
fi

if [ "$OUTPUT" = "-" ]; then
    "${args[@]}" "$@"
else
    mkdir -p "$(dirname "$OUTPUT")"
    "${args[@]}" "$@" | tee "$OUTPUT"
fi
