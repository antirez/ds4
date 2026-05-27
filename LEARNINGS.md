# DS4 Project Learnings

This file records friction points, root causes, fixes, and general patterns for working with the DS4 project.

---

## Sparse Worktrees for Public Branches

### Problem

When contributing to upstream DS4 (github.com/antirez/ds4), we need to push branches to a public fork (github.com/ljubomirj/l26f). However, the main ds4 working directory contains private files that shouldn't be public:
- `README.LJ` - personal admin notes
- `.viminfo` - vim state file

### Solution: Git Sparse Worktree

Create a separate working tree with selective file checkout for public branches.

#### Process

```bash
cd ~/ds4  # Always work from ~/ds4 top-level directory

# Create worktree for specific branch
git worktree add ~/ds4/worktrees/ds4-public reap-compact-support

# Configure sparse checkout (in worktree directory)
cd ~/ds4/worktrees/ds4-public
git sparse-checkout init
git sparse-checkout set '/*'  # Include all files by default

# Exclude private files (manually remove them)
rm README.LJ .viminfo

# Build and verify
cd ~/ds4/worktrees/ds4-public
make
./ds4 --version
```

#### Key Constraints

1. **All DS4 work happens in ~/ds4 directory only** - Never touch ~/l26f or other home dir projects
2. **Worktree location**: `~/ds4/worktrees/` subdirectory
3. **Push from main tree**: `git push fork reap-compact-support` (from ~/ds4, not worktree)

#### Sparse Checkout Gotchas

- `git worktree add --sparse` is NOT valid - sparse is configured after creation
- `git sparse-checkout set '/*'` - must include patterns, NOT directories
- `git sparse-checkout reapply` - NOT `apply`
- Simplest approach: manually delete private files instead of complex sparse patterns

#### Worktree Management

```bash
# List all worktrees
git worktree list

# Remove worktree when done
git worktree remove ~/ds4/worktrees/ds4-public
```

---

## REAP-Compact GGUF Implementation

### Key Discovery

GGUF `reap.layer.expert_count` metadata contains the **original** expert count (256), NOT the actual compacted count. Must infer per-layer expert counts from tensor dimensions directly.

### Pattern

Layer 0-2: `blk.N.ffn_gate_inp.weight` shape = [4096, 256] (hash-preserved)
Layer 3+: `blk.N.ffn_gate_inp.weight` shape = [4096, 192] (REAP-pruned)

### Implementation Pattern

```c
static uint32_t g_reap_layer_expert_count[DS4_MAX_LAYER];

static inline uint32_t reap_layer_expert_count(uint32_t il) {
    return g_reap_layer_expert_count[il];
}

// Usage: replace DS4_N_EXPERT with reap_layer_expert_count(il)
const uint32_t n_expert = reap_layer_expert_count(il);
tensor_expect_layout(l->ffn_gate_inp, DS4_TENSOR_F16, 2, DS4_N_EMBD, n_expert, 0);
```

---

## Build System

### Makefile Dependencies

The Makefile rebuilds correctly when headers change:
- `ds4.o` depends on `ds4.h ds4_gpu.h`
- Modify `ds4.h` → automatic rebuild of `ds4.o`

### Common Commands

```bash
make              # Build ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent
make clean        # Remove build outputs
make test         # Build and run tests (requires model + Metal)
```

---

## Testing Notes

### Smoke Test

```bash
./ds4 -m <model.gguf> --ctx 512 --nothink --temp 0 -n 1 -p "hello"
# Expected: Output "Hello" with performance metrics
```

### Memory Test

Watch `ds4: mapped` line for memory usage at different context sizes.
- Stock DS4: ~82GB at ctx=512
- REAP25: ~65GB at ctx=512 (~17GB savings)
