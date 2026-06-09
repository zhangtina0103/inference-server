import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class KVBlock:
    """one block of KV cache — fixed size chunk of tokens"""
    block_id: int
    num_tokens: int = 0
    max_tokens: int = 16
    key:   torch.Tensor = None
    value: torch.Tensor = None

    def __post_init__(self):
        # preallocate key/value tensors for this block
        # shape: (max_tokens, num_heads, head_dim)
        self.key   = torch.zeros(self.max_tokens, 8, 64)
        self.value = torch.zeros(self.max_tokens, 8, 64)

    @property
    def is_full(self):
        return self.num_tokens >= self.max_tokens

    @property
    def is_empty(self):
        return self.num_tokens == 0


class PagedKVCache:
    """
    simplified paged KV cache — mirrors vLLM's PagedAttention concept.

    instead of allocating max_seq_len for every request (wasteful),
    we allocate fixed-size blocks on demand. when a request needs more
    tokens, we allocate another block. when it finishes, blocks are freed.

    this is exactly what vLLM does under the hood.
    """

    def __init__(self, num_blocks=128, block_size=16, num_heads=8, head_dim=64):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads  = num_heads
        self.head_dim   = head_dim

        # pool of free blocks
        self.free_blocks = list(range(num_blocks))

        # map from request_id → list of block_ids
        self.request_to_blocks: dict[int, list[int]] = {}

        # actual block storage
        self.blocks = {
            i: KVBlock(block_id=i, max_tokens=block_size)
            for i in range(num_blocks)
        }

    def allocate(self, request_id: int) -> bool:
        """allocate first block for a new request"""
        if not self.free_blocks:
            return False  # OOM — no free blocks

        block_id = self.free_blocks.pop(0)
        self.request_to_blocks[request_id] = [block_id]
        return True

    def write(self, request_id: int, token_idx: int,
              key: torch.Tensor, value: torch.Tensor):
        """write KV for one token"""
        blocks = self.request_to_blocks[request_id]
        current_block = self.blocks[blocks[-1]]

        # if current block is full, allocate a new one
        if current_block.is_full:
            if not self.free_blocks:
                raise RuntimeError("KV cache OOM")
            new_block_id = self.free_blocks.pop(0)
            blocks.append(new_block_id)
            current_block = self.blocks[new_block_id]

        # write into current block
        pos = current_block.num_tokens
        current_block.key[pos]   = key
        current_block.value[pos] = value
        current_block.num_tokens += 1

    def read(self, request_id: int):
        """read all KV tensors for a request"""
        blocks   = self.request_to_blocks[request_id]
        keys     = []
        values   = []

        for block_id in blocks:
            block = self.blocks[block_id]
            if block.num_tokens > 0:
                keys.append(block.key[:block.num_tokens])
                values.append(block.value[:block.num_tokens])

        if not keys:
            return None, None

        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def free(self, request_id: int):
        """free all blocks for a finished request"""
        if request_id not in self.request_to_blocks:
            return
        for block_id in self.request_to_blocks[request_id]:
            self.blocks[block_id].num_tokens = 0
            self.free_blocks.append(block_id)
        del self.request_to_blocks[request_id]

    @property
    def num_free_blocks(self):
        return len(self.free_blocks)

    @property
    def utilization(self):
        return 1 - (len(self.free_blocks) / self.num_blocks)

    def __repr__(self):
        return (f"PagedKVCache(blocks={self.num_blocks}, "
                f"free={self.num_free_blocks}, "
                f"utilization={self.utilization:.1%})")
