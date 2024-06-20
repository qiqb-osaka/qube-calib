from __future__ import annotations

from collections import defaultdict, deque  # , namedtuple
from typing import Optional

# from copy import copy


class Tree(defaultdict):
    def __init__(self, root: Optional[int] = None):
        super().__init__(list)
        if root is not None:
            self[root]

    @property
    def root(self) -> int:
        for k in self.keys():
            c = [k for v in self.values() if k in v]
            if not c:
                return k
            else:
                continue
        raise ValueError("loop detected")

    def adopt(self, parent: int, child: int) -> None:
        if child not in self.children:
            self[parent].append(child)
        else:
            raise ValueError("The child object is already adopted to another parent.")

    def breadth_first_search(self, start: Optional[int] = None) -> list[int]:
        t = self
        lst = []
        if start is None:
            start = t.root
        q = deque([start])
        while q:
            p = q.popleft()
            lst.append(p)
            for c in t[p]:
                q.append(c)
        return lst

    def bfs(self, start: Optional[int] = None) -> list[int]:
        return self.breadth_first_search(start)

    def parentof(self, child: int) -> int:
        t = self
        for k in t.keys():
            if child in t[k]:
                return k
            else:
                continue
        raise ValueError("this node is root")

    def insert(self, parent: int, child: int) -> None:
        t = self
        c = child
        p = self.parentof(c)
        t[p].remove(c)
        self.adopt(parent, c)
        if p is None:
            raise ValueError("it seems root node")
        self.adopt(p, parent)

    @property
    def parents(self) -> list[int]:
        return [k for k in self.keys() if self[k]]

    @property
    def children(self) -> list[int]:
        return list(set([i for v in self.values() for i in v]))

    @property
    def all(self) -> list[int]:
        return list(set(self.parents + self.children))

    def duplicate(self) -> Tree:
        new = Tree()
        for p in self.parents:
            new[p] = self[p]

        return new


class CostedTree:
    def __init__(self) -> None:
        self._tree = Tree()
        self._cost: dict[int, Optional[float]] = dict()
        # self._total_cost: dict[int, Optional[float]] = dict()

    def adopt(self, parent: int, child: int, cost: float) -> None:
        self._tree.adopt(parent, child)
        self._cost[child] = cost

    def evaluate(self, root: Optional[int] = None) -> dict[int, float]:
        total_costs = dict()
        if root is None:
            if self._tree.root is None:
                raise ValueError("root is not found")
            root = self._tree.root
        total_costs[root] = 0.0
        bfs = self._tree.breadth_first_search(root)
        for node in bfs[1:]:
            cost = self._cost[node]
            total_cost = total_costs[self._tree.parentof(node)]
            if cost is None:
                raise ValueError(f"cost is None, {node}: {self._cost}")
            # if total_costs is None:
            #     raise ValueError("total cost is None")
            total_costs[node] = cost + total_cost
        return total_costs

    def parentof(self, child: int) -> int:
        return self._tree.parentof(child)

    def breadth_first_search(self, start: Optional[int] = None) -> list[int]:
        return self._tree.breadth_first_search(start)


if __name__ == "__main__":
    t = Tree()
    t.adopt(0, 1)
    t.adopt(0, 2)
    t.adopt(1, 3)
    t.adopt(3, 4)
    t.adopt(2, 5)
    print(t)
    print(t.root)
    t.insert(6, 3)
    print(t.all)
    print(t)
    print(t.parentof(4))
    print(t.breadth_first_search())
    print(t.duplicate())
