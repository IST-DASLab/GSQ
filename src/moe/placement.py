import random

class ExpertSharder:
    def __init__(self, num_experts, world_size, mapping=None, random_assign=True, seed=42):
        self.num_experts = num_experts
        self.world_size = world_size

        if mapping is not None:
            self._owner = dict(mapping)
        else:
            if random_assign:
                rng = random.Random(seed)
                base = []
                full_rounds, remainder = divmod(num_experts, world_size)

                for _ in range(full_rounds):
                    base.extend(range(world_size))

                base.extend(range(remainder))

                rng.shuffle(base)

                self._owner = {e: base[e] for e in range(num_experts)}
            else:
                self._owner = {e: e % world_size for e in range(num_experts)}

        self._locals = {r: [] for r in range(world_size)}
        for e, r in self._owner.items():
            self._locals[r].append(e)

    def owner(self, expert_id):
        return self._owner[expert_id]

    def local_experts(self, rank):
        return self._locals[rank]
