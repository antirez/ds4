#!/usr/bin/env bash
# distributed_launch.sh — Launch ds4-server across multiple nodes via JACCL
#
# Discovers RDMA interfaces via asmi, generates JACCL env vars, and SSHs
# to each node to start ds4-server with --distributed.
#
# Usage:
#   ./distributed_launch.sh --nodes hub,m3u4 --model gguf/ds4flash.gguf [--ctx 32768] [--port 8080]

set -euo pipefail

# --- Defaults ---
NODES=""
MODEL=""
CTX=32768
PORT=8080
DS4_BIN="./ds4-server"
EXTRA_ARGS=""

usage() {
    cat <<USAGE
Usage: $(basename "$0") --nodes node1,node2 --model <path> [OPTIONS]

Required:
  --nodes node1,node2   Comma-separated list of nodes (hostnames or Tailscale names)
  --model <path>        Path to GGUF model file (must exist on all nodes)

Options:
  --ctx N               Context length (default: 32768)
  --port P              Server port (default: 8080)
  --bin <path>          Path to ds4-server binary (default: ./ds4-server)
  --extra "<args>"      Extra args to pass to ds4-server
  -h, --help            Show this help

Environment:
  The script uses asmi (port 9090) on each node to discover RDMA interfaces.
  The coordinator IP is rank 0's LAN IP (from Tailscale, NOT TB5 /30 IPs).

Examples:
  # 2-node launch
  $(basename "$0") --nodes hub,m3u4 --model gguf/ds4flash.gguf --ctx 32768

  # 4-node launch with custom port
  $(basename "$0") --nodes hub,m3u1,m3u3,m3u4 --model gguf/ds4flash.gguf --port 9090
USAGE
    exit 0
}

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)    NODES="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        --ctx)      CTX="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --bin)      DS4_BIN="$2"; shift 2 ;;
        --extra)    EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *)          echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$NODES" || -z "$MODEL" ]]; then
    echo "Error: --nodes and --model are required"
    echo ""
    usage
fi

# --- Split nodes into array ---
IFS=',' read -ra NODE_LIST <<< "$NODES"
WORLD_SIZE=${#NODE_LIST[@]}

if [[ $WORLD_SIZE -lt 2 ]]; then
    echo "Error: distributed mode requires at least 2 nodes"
    exit 1
fi

echo "=== ds4 distributed launch ==="
echo "Nodes:      ${NODE_LIST[*]}"
echo "World size: $WORLD_SIZE"
echo "Model:      $MODEL"
echo "Context:    $CTX"
echo "Port:       $PORT"
echo ""

# --- Resolve coordinator IP (rank 0's LAN IP via Tailscale) ---
# Use the LAN IP (10.x.x.x), NOT TB5 /30 IPs which cause JACCL error 60.
COORD_NODE="${NODE_LIST[0]}"
echo "Resolving coordinator LAN IP for $COORD_NODE..."

# Try tailscale status to get the 100.x IP, then fall back to 10.x from /etc/hosts or DNS
COORD_IP=$(ssh -o ConnectTimeout=5 "$COORD_NODE" \
    "tailscale ip -4 2>/dev/null || hostname -I 2>/dev/null | awk '{print \$1}'" 2>/dev/null)

if [[ -z "$COORD_IP" ]]; then
    echo "Error: could not resolve LAN IP for coordinator $COORD_NODE"
    exit 1
fi
echo "Coordinator: $COORD_IP ($COORD_NODE)"
echo ""

# --- Discover RDMA interfaces via asmi ---
# Build the JACCL_IBV_DEVICES JSON matrix.
# For N nodes, this is an NxN matrix where [i][j] is the RDMA interface on node i
# that connects to node j (null for self).
#
# Example for 2 nodes: [[null, "rdma_en7"], ["rdma_en5", null]]

echo "Discovering RDMA interfaces via asmi..."

declare -A RDMA_IFACE  # RDMA_IFACE[i,j] = interface name on node i for link to node j

for (( i=0; i<WORLD_SIZE; i++ )); do
    node="${NODE_LIST[$i]}"
    echo "  Querying $node:9090/links..."

    # asmi /links returns JSON array of RDMA link objects with peer_hostname and rdma_device
    links_json=$(curl -sf "http://${node}:9090/links" 2>/dev/null || echo "[]")

    if [[ "$links_json" == "[]" ]]; then
        echo "  Warning: no RDMA links found on $node (asmi may not be running)"
    fi

    for (( j=0; j<WORLD_SIZE; j++ )); do
        if [[ $i -eq $j ]]; then
            RDMA_IFACE[$i,$j]="null"
            continue
        fi

        peer="${NODE_LIST[$j]}"
        # Extract the RDMA device name for the link to this peer
        iface=$(echo "$links_json" | python3 -c "
import json, sys
links = json.load(sys.stdin)
for link in links:
    peer = link.get('peer_hostname', link.get('peer', ''))
    if '$peer' in peer:
        print(link.get('rdma_device', link.get('interface', '')))
        break
" 2>/dev/null || echo "")

        if [[ -n "$iface" ]]; then
            RDMA_IFACE[$i,$j]="\"$iface\""
        else
            echo "  Warning: no RDMA interface found on $node for peer $peer"
            RDMA_IFACE[$i,$j]="null"
        fi
    done
done

# --- Build JACCL_IBV_DEVICES JSON ---
IBV_JSON="["
for (( i=0; i<WORLD_SIZE; i++ )); do
    [[ $i -gt 0 ]] && IBV_JSON+=", "
    IBV_JSON+="["
    for (( j=0; j<WORLD_SIZE; j++ )); do
        [[ $j -gt 0 ]] && IBV_JSON+=", "
        IBV_JSON+="${RDMA_IFACE[$i,$j]}"
    done
    IBV_JSON+="]"
done
IBV_JSON+="]"

echo ""
echo "JACCL_IBV_DEVICES=$IBV_JSON"
echo ""

# --- Launch on each node ---
PIDS=()
for (( rank=0; rank<WORLD_SIZE; rank++ )); do
    node="${NODE_LIST[$rank]}"
    echo "Launching rank $rank on $node..."

    ssh -o ConnectTimeout=10 "$node" bash -c "'
        export JACCL_RANK=$rank
        export JACCL_WORLD_SIZE=$WORLD_SIZE
        export JACCL_COORDINATOR=$COORD_IP
        export JACCL_IBV_DEVICES='"'"'$IBV_JSON'"'"'
        echo \"ds4: starting rank $rank/$WORLD_SIZE on \$(hostname)\"
        echo \"  JACCL_COORDINATOR=$COORD_IP\"
        echo \"  JACCL_IBV_DEVICES=\$JACCL_IBV_DEVICES\"
        cd \$(dirname $DS4_BIN) 2>/dev/null || true
        exec $DS4_BIN --distributed --metal --host 0.0.0.0 --port $PORT --ctx $CTX --model $MODEL $EXTRA_ARGS
    '" &
    PIDS+=($!)
done

echo ""
echo "=== All $WORLD_SIZE ranks launched ==="
echo "PIDs: ${PIDS[*]}"
echo "Press Ctrl-C to stop all nodes"
echo ""

# --- Wait for any child to exit ---
cleanup() {
    echo ""
    echo "Shutting down all ranks..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    echo "All ranks stopped."
}
trap cleanup INT TERM

wait -n "${PIDS[@]}" 2>/dev/null || true
echo "A rank exited. Shutting down remaining..."
cleanup
