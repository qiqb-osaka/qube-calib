from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, MutableSequence, Optional

import numpy as np
from numpy.typing import NDArray

from .tree import CostedTree, Tree


class SequenceTree:
    def __init__(self) -> None:
        self._tree = CostedTree()
        self._active_node = 0
        self._latest_node = 0
        self._nodes_items: Dict[int, Item] = {}

    def append(self, item: Item) -> int:
        self._latest_node += 1
        self._tree.adopt(
            self._active_node,
            self._latest_node,
            cost=-1,
        )
        self._nodes_items[self._latest_node] = item
        self._active_node = self._latest_node
        return self._latest_node

    def branch(self, branch: Branch) -> None:
        # 枝分かれの開始点を退避する
        branched_node = self._active_node
        # blanch を木構造に追加する
        self.append(branch)
        # 枝分かれ終了後の復帰点を退避する
        branch._next_node = self._active_node
        # 枝分かれの開始点を木構造に教える
        self._active_node = branched_node
        # 枝分かれの開始点に Dummy を追加する
        branch_root = self.append(Dummy())
        # branch 以下の最長経路を計算するための開始点を branch に教える
        branch._root_node = branch_root

    def finalize(self) -> None:
        # deeper branch
        # 深い branch から順に slot を配置する
        # print(self._tree._tree)
        # print(self._nodes_items)
        for _ in [
            self._nodes_items[_]
            for _ in self.breadth_first_search()[1:]
            if isinstance(self._nodes_items[_], Branch)
        ][::-1]:
            # print(_)
            # print(self._tree._tree)
            # print(self._tree._cost)
            _.place(self)
            # print(_)

        # sub tree list
        # for node, item in self._nodes_items.items():
        #     if isinstance(item, Branch):
        #         continue
        #     if item.duration is None:
        #         raise ValueError(f"{item} duration is not defined")
        #     self._tree._cost[node] = item.duration

        # t = {
        #     node: item
        #     for node, item in self._nodes_items.items()
        #     if isinstance(item, Branch)
        # }
        # for node, item in t.items():
        #     i = [
        #         self._nodes_items[_]
        #         for _ in self.breadth_first_search(item._root_node)
        #         if isinstance(self._nodes_items[_], Branch)
        #     ]

        #     print(i)

        # 最終的な slot 配置を確定する（SubSequenceを必ず Toplevel に置くなら必要ないかも？）
        # 各アイテムのコストを更新する
        for node, item in self._nodes_items.items():
            self._tree._cost[node] = item.duration
        total_costs = self._tree.evaluate()
        # print(self._tree.evaluate())
        for node, slot in self._nodes_items.items():
            end = total_costs[node]
            if end is None:
                raise ValueError("invalid value of total cost")
            duration = slot.duration
            if duration is None:
                raise ValueError(f"invalid duration of slot {slot}")
            slot.begin = end - duration

    def parentof(self, child: int) -> int:
        return self._tree.parentof(child)

    def breadth_first_search(self, start: Optional[int] = None) -> List[int]:
        return self._tree.breadth_first_search(start)


@dataclass
class RunningConfig:
    # tree: Optional[SequenceTree] = None
    contexts: Final[MutableSequence] = deque()


__rc__: Final[RunningConfig] = RunningConfig()


class ContextNode:
    def __init__(self) -> None:
        if len(__rc__.contexts):
            __rc__.contexts[-1].append(self)


class Item:
    def __init__(self, duration: Optional[float] = None) -> None:
        super().__init__()
        self._duration = duration
        self.begin: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        self._duration = duration

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(duration={self.duration}, begin={self.begin})"
        )


class Slot(Item, ContextNode):
    pass


class Blank(Item):
    pass


class Branch(Item):
    def __init__(self) -> None:
        super().__init__()
        self.duration = None
        self._next_node: Optional[int] = None
        self._root_node: Optional[int] = None

    def place(self, tree: SequenceTree) -> None:
        print(f"{self.__class__.__name__}, root_node={self._root_node}")
        # 最大長を計算する
        for _ in tree.breadth_first_search(self._root_node)[1:]:
            tree._tree._cost[_] = tree._nodes_items[_].duration
        max_duration = max([_ for _ in tree._tree.evaluate(self._root_node).values()])
        # branch の duration は最大長に揃えると同時に cost も確定する
        self.duration = max_duration
        if self._next_node is None:
            raise ValueError("_next_node is None")
        tree._tree._cost[self._next_node] = self.duration

    # @property
    # def duration(self) -> Optional[float]:
    #     if self._tree is None:
    #         raise ValueError("_tree is None")
    #     nodes_items = self._tree._nodes_items
    #     bfs = self._tree.breadth_first_search(self._root_node)
    #     for node in bfs[1:]:
    #         # if not isinstance(nodes_items[node], Branch):
    #         self._tree._tree._cost[node] = self._tree._nodes_items[node].duration
    #     return max([_ for _ in self._tree._tree.evaluate(self._root_node).values()])

    # @duration.setter
    # def duration(self, duration: float) -> None:
    #     """Branch object cannot set duration value"""
    #     raise ValueError("Branch object cannot set duration value")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(duration={self.duration}, begin={self.begin}, next_node={self._next_node}, root_node={self._root_node})"


class Dummy(Item):
    def __init__(self) -> None:
        super().__init__()
        self._duration = 0

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        """Branch object cannot set duration value"""
        raise ValueError("Branch object cannot set duration value")


class Modifier(Item, ContextNode):
    def __init__(self) -> None:
        super().__init__()
        self.duration = 0
        self.begin: Optional[float] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(begin={self.begin})"


class DequeWithContext(deque):
    def __enter__(self) -> DequeWithContext:
        __rc__.contexts.append(self)
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        __rc__.contexts.pop()


class Sequence(DequeWithContext):
    # def __enter__(self) -> Sequence:
    #     super().__enter__()
    #     self._tree = __rc__.tree = SequenceTree()
    #     return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        self._tree = SequenceTree()
        # print(f"{self.__class__.__name__}")
        # print(self._tree._tree._tree)
        # if self._tree is None:
        #     raise ValueError("SequenceTree is not prepared.")
        for item in self:
            if isinstance(item, Item):
                slot = item
                self._tree.append(slot)
            elif isinstance(item, SequenceTree):
                _tree = item
                # update index
                # offset = self._tree._active_node
                all_nodes = self._tree._tree._tree.all
                if all_nodes:
                    offset = max(all_nodes)
                else:
                    offset = 0
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                # offset = max(self._tree._tree._tree.all)
                root = _tree._tree._tree.root
                for node, items in _tree._tree._tree.items():
                    if node == root:
                        self._tree._tree._tree[self._tree._active_node] += [
                            _ + offset for _ in items
                        ]
                        continue
                    self._tree._tree._tree[node + offset] = [_ + offset for _ in items]
                for node in _tree.breadth_first_search()[1:]:
                    self._tree._nodes_items[node + offset] = _tree._nodes_items[node]
                    self._tree._tree._cost[node + offset] = -1
                # print(tree._tree._tree.all)
                # for child in tree.breadth_first_search()[1:]:
                #     parent = tree._tree._tree.parentof(child)
                #     self._tree._tree.adopt(parent + offset, child + offset, cost=-1)
                #     self._tree._nodes_items[child + offset] = tree._nodes_items[child]
                # self._active_node = self._latest_node
                # return self._latest_node
                # self._tree.

                # print(tree._tree._tree)
                # print(tree._nodes_items)
                # print(tree._tree._cost)
                # print(self._tree._latest_node)

    def _get_tree(self) -> Tree:
        if self._tree is None:
            raise ValueError("SequenceTree is not prepared.")
        return self._tree._tree._tree


class SubSequenceBranch(Branch):
    pass


class SubSequence(DequeWithContext):
    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        # このブランチ用のローカルツリーを作る
        tree = SequenceTree()
        __rc__.contexts[-1].append(tree)  # with 内の定義の所定の位置にツリーを追加
        # ツリーの根本にブランチアイテムを作る．このブランチの外のアイテムはこのブランチアイテムの次につながる
        branch = SubSequenceBranch()
        tree.branch(branch)
        # with 内で定義された item を舐める
        for item in self:
            if isinstance(item, Item):
                # Item ならばそのまま登録
                slot = item
                tree.append(slot)
            elif isinstance(item, SequenceTree):
                # サブツリーを見つけたらブランチツリーとマージする
                _tree = item
                # サブツリーのルート直下第 1 アイテムは branch のはず
                # サブツリーを抜けたら _branch._next_node に次のアイテムをぶら下げる
                _branch = _tree._nodes_items[1]
                if isinstance(_branch, Branch):
                    if _branch._next_node is None:
                        raise ValueError("_next_node is None")
                    next_item = _branch._next_node
                else:
                    raise ValueError("1st node is not Branch")
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                offset = max(tree._tree._tree.all)
                root = _tree._tree._tree.root
                for parent, children in _tree._tree._tree.items():
                    if parent == root:
                        tree._tree._tree[tree._active_node] += [
                            _ + offset for _ in children
                        ]
                        continue
                    tree._tree._tree[parent + offset] = [_ + offset for _ in children]
                for node in _tree.breadth_first_search()[1:]:
                    _item = _tree._nodes_items[node]
                    tree._nodes_items[node + offset] = _item
                    tree._tree._cost[node + offset] = -1
                    if isinstance(_item, Branch):
                        # _item._tree = tree
                        if _item._root_node is None:
                            raise ValueError("_root_node is None")
                        _item._root_node += offset
                        if _item._next_node is None:
                            raise ValueError("_next_node is None")
                        _item._next_node += offset  #     # branch だったら _next_node や _root_node も新しい node 名に更新
                # offset = tree._active_node
                # offset = max(tree._tree._tree.all)
                # # # 幅優先探索でサブツリーの全アイテムを舐める
                # bfsitems = _tree.breadth_first_search()

                # for child in bfsitems[1:]:
                #     parent = _tree._tree._tree.parentof(child)
                #     # ローカルツリーに追加する
                #     tree._tree.adopt(parent + offset, child + offset, cost=-1)
                #     # ローカルツリーにアイテムを登録する
                #     tree._nodes_items[child + offset] = _tree._nodes_items[child]
                #     # branch だったら _next_node や _root_node も新しい node 名に更新
                #     _item = _tree._nodes_items[child]
                #     if isinstance(_item, Branch):
                #         # _item._tree = tree
                #         if _item._root_node is None:
                #             raise ValueError("_root_node is None")
                #         _item._root_node += offset
                #         if _item._next_node is None:
                #             raise ValueError("_next_node is None")
                #         _item._next_node += offset
                # ローカルツリーのカウンターをアップデート
                # tree._latest_node = max(_tree.breadth_first_search()) + offset
                tree._latest_node = max(tree._tree._tree.all)  # + offset
                # ローカルツリーの次の追加先を先頭 branch の次に指定
                # print(f"{_tree.__class__.__name__}, next={next_item + offset}")
                tree._active_node = next_item + offset
        # if branch._next_node is None:
        #     raise ValueError("_next_node is None")
        # # 追加が終わったら親ツリーの追加先を SubSequence の後に指定
        # tree._active_node = branch._next_node


class SeriesBranch(Branch):
    pass


class Series(DequeWithContext):
    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        # この context 用のローカルツリーを作る
        tree = SequenceTree()
        # print(f"exit {self.__class__.__name__}")
        # 外側の context にローカルツリーを渡す
        __rc__.contexts[-1].append(tree)
        # ローカルツリーのルート直下を branch して branch item を登録する
        branch = SeriesBranch()
        tree.branch(branch)
        # with 内で定義された item を舐める
        for item in self:
            if isinstance(item, Item):
                # Item ならばそのまま登録
                slot = item
                tree.append(slot)
            elif isinstance(item, SequenceTree):
                # サブツリーを見つけたらローカルツリーとマージする
                _tree = item
                # サブツリーのルート直下第 1 アイテムは branch のはず
                # サブツリーを抜けたら _branch._next_node に次のアイテムをぶら下げる
                _branch = _tree._nodes_items[1]
                if isinstance(_branch, Branch):
                    if _branch._next_node is None:
                        raise ValueError("_next_node is None")
                    next_item = _branch._next_node
                else:
                    raise ValueError("1st node is not Branch")
                # ツリーをマージするためにノード番号を更新する
                offset = max(tree._tree._tree.all)
                # print("local", offset, tree._active_node, tree._tree._tree)
                # print("sub", _tree._tree._tree)
                # offset = tree._active_node
                # offset = max(tree._tree._tree.all)
                # 幅優先探索で全アイテムを舐める
                for child in _tree.breadth_first_search()[1:]:
                    parent = _tree._tree._tree.parentof(child)
                    # ブランチ用ツリーに追加する
                    tree._tree.adopt(parent + offset, child + offset, cost=-1)
                    # ブランチ用ツリーのあいてむに登録する
                    tree._nodes_items[child + offset] = _tree._nodes_items[child]
                    # branch だったら _next_node や _root_node も新しい node 名に更新
                    _item = _tree._nodes_items[child]
                    if isinstance(_item, Branch):
                        # _item._tree = tree
                        if _item._root_node is None:
                            raise ValueError("_root_node is None")
                        _item._root_node += offset
                        if _item._next_node is None:
                            raise ValueError("_next_node is None")
                        _item._next_node += offset
                # 親ツリーのカウンターをアップデート
                tree._latest_node = max(_tree.breadth_first_search()) + offset
                # 親ツリーの追加先を先頭 branch の次に指定
                tree._active_node = next_item + offset
        # print("local", tree._tree._tree)
        # if branch._next_node is None:
        #     raise ValueError("_next_node is None")
        # 追加が終わったら親ツリーの追加先を SubSequence の後に指定
        # tree._active_node = branch._next_node
        # print(self.__class__.__name__, tree._tree._tree)
        # print(self.__class__.__name__, tree._tree._tree, tree.breadth_first_search())


class FlushleftBranch(Branch):
    pass
    # def place(self, tree: SequenceTree) -> None:
    #     # 最大長を計算する
    #     for _ in tree.breadth_first_search(self._root_node)[1:]:
    #         tree._tree._cost[_] = tree._nodes_items[_].duration
    #     max_duration = max([_ for _ in tree._tree.evaluate(self._root_node).values()])
    #     # branch の duration は最大長に揃えると同時に cost も確定する
    #     self.duration = max_duration
    #     if self._next_node is None:
    #         raise ValueError("_next_node is None")
    #     tree._tree._cost[self._next_node] = self.duration
    #     # # _root_node にぶら下がっている blank node を取得する
    #     # for _ in tree._tree._tree[self._root_node]:
    #     #     # blank にぶら下がっているノードの最大長を取得する
    #     #     branch_duration = max([_ for _ in tree._tree.evaluate(_).values()])
    #     #     # 右揃えになるよう blank を調整するかつ cost も確定する
    #     #     tree._nodes_items[_].duration = max_duration - branch_duration
    #     #     tree._tree._cost[_] = tree._nodes_items[_].duration

    # @property
    # def duration(self) -> Optional[float]:
    #     if self._tree is None:
    #         raise ValueError("_tree is None")

    #     bfs = self._tree.breadth_first_search(self._root_node)
    #     for node in bfs[1:]:
    #         # if not isinstance(nodes_items[node], Branch):
    #         self._tree._tree._cost[node] = self._tree._nodes_items[node].duration
    #     return max([_ for _ in self._tree._tree.evaluate(self._root_node).values()])

    # @duration.setter
    # def duration(self, duration: float) -> None:
    #     """Branch object cannot set duration value"""
    #     raise ValueError("Branch object cannot set duration value")


class Flushleft(DequeWithContext):
    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        # このブランチ用のサブツリーを作る
        tree = SequenceTree()
        __rc__.contexts[-1].append(tree)  # with 内の定義の所定の位置にツリーを追加
        # print(f"exit {self.__class__.__name__}")
        # ツリーの根本にブランチアイテムを作る．このブランチの外のアイテムはこのブランチアイテムの次につながる
        branch = FlushleftBranch()
        tree.branch(branch)
        # with 内で定義された item を舐める
        if branch._root_node is None:
            raise ValueError("_root_node is None")
        tree._active_node = branch._root_node
        for item in self:
            # print("left", tree._active_node)
            if isinstance(item, Item):
                # Item ならばそのまま登録
                # ただしパラレルなので注意
                tree.append(item)
                if branch._root_node is None:
                    raise ValueError("_root_node is None")
                tree._active_node = branch._root_node
            elif isinstance(item, SequenceTree):
                # サブツリーを見つけたら親ツリーとマージする
                # サブツリーのルート直下第 1 アイテムは branch のはず
                # サブツリーを抜けたら _branch._root_node に次のアイテムをぶら下げる
                _tree = item
                # _branch = _tree._nodes_items[1]
                # if isinstance(_branch, Branch):
                #     if _branch._next_node is None:
                #         raise ValueError("_next_node is None")
                #     next_item = _branch._next_node
                #     if _branch._root_node is None:
                #         raise ValueError("_root_node is None")
                #     root_node = _branch._root_node
                # else:
                #     raise ValueError("1st node is not Branch")
                # ツリーをマージするためにノード番号を更新する
                offset = max(tree._tree._tree.all)
                # print("local", offset, tree._active_node, tree._tree._tree)
                # print("sub", _tree._tree._tree)
                # offset = tree._active_node
                # 幅優先探索で全アイテムを舐める
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                offset = max(tree._tree._tree.all)
                root = _tree._tree._tree.root
                for parent, children in _tree._tree._tree.items():
                    if parent == root:
                        tree._tree._tree[tree._active_node] += [
                            _ + offset for _ in children
                        ]
                        continue
                    tree._tree._tree[parent + offset] = [_ + offset for _ in children]
                for node in _tree.breadth_first_search()[1:]:
                    _item = _tree._nodes_items[node]
                    tree._nodes_items[node + offset] = _item
                    tree._tree._cost[node + offset] = -1
                    if isinstance(_item, Branch):
                        # _item._tree = tree
                        if _item._root_node is None:
                            raise ValueError("_root_node is None")
                        _item._root_node += offset
                        if _item._next_node is None:
                            raise ValueError("_next_node is None")
                        _item._next_node += offset
                # for child in _tree.breadth_first_search()[1:]:
                #     parent = _tree._tree._tree.parentof(child)
                #     # グローバルツリーに追加する
                #     tree._tree.adopt(parent + offset, child + offset, cost=-1)
                #     # グローバルツリーにアイテムを登録する
                #     tree._nodes_items[child + offset] = _tree._nodes_items[child]
                #     # branch だったら _next_node や _root_node も新しい node 名に更新
                #     _item = _tree._nodes_items[child]
                # 親ツリーのカウンターをアップデート
                tree._latest_node = max(_tree.breadth_first_search()) + offset
                # 親ツリーの追加先を先頭 branch の次に指定
                if branch._root_node is None:
                    raise ValueError("_root_node is None")
                tree._active_node = branch._root_node
        # print("local", tree._tree._tree)
        # if branch._next_node is None:
        #     raise ValueError("_next_node is None")
        # # 追加が終わったら親ツリーの追加先を SubSequence の後に指定
        # tree._active_node = branch._next_node


class FlushrightBranch(Branch):
    def place(self, tree: SequenceTree) -> None:
        super().place(tree)
        # # blank が 0 の状態で最大長を計算する
        # for _ in tree.breadth_first_search(self._root_node)[1:]:
        #     tree._tree._cost[_] = tree._nodes_items[_].duration
        # max_duration = max([_ for _ in tree._tree.evaluate(self._root_node).values()])
        # # branch の duration は最大長に揃えると同時に cost も確定する
        # self.duration = max_duration
        # if self._next_node is None:
        #     raise ValueError("_next_node is None")
        # tree._tree._cost[self._next_node] = self.duration
        max_duration = self._duration
        # _root_node にぶら下がっている blank node を取得する
        for _ in tree._tree._tree[self._root_node]:
            # blank にぶら下がっているノードの最大長を取得する
            branch_duration = max([_ for _ in tree._tree.evaluate(_).values()])
            # 右揃えになるよう blank を調整するかつ cost も確定する
            tree._nodes_items[_].duration = max_duration - branch_duration
            tree._tree._cost[_] = tree._nodes_items[_].duration

        # # 最長の経路を探すためにツリー全体の leaves をみつける
        # leaves = [
        #     _ for _ in tree._tree._tree.children if _ not in tree._tree._tree.parents
        # ]
        # # 経路にこのブランチのルートノードを含む leaves をみつける
        # retval = []
        # for leaf in leaves:
        #     _leaf = leaf
        #     while True:
        #         _leaf = tree.parentof(_leaf)
        #         if _leaf == tree._tree._tree.root:
        #             break
        #         if _leaf == self._root_node:
        #             retval.append(leaf)
        # print(retval)

    # @property
    # def duration(self) -> Optional[float]:
    #     if self._tree is None:
    #         raise ValueError("_tree is None")
    #     # nodes_items = self._tree._nodes_items

    #     bfs = self._tree.breadth_first_search(self._root_node)
    #     for node in bfs[1:]:
    #         # if not isinstance(nodes_items[node], Branch):
    #         self._tree._tree._cost[node] = self._tree._nodes_items[node].duration
    #     nodes_totalcosts = self._tree._tree.evaluate(self._root_node)
    #     print("flushright", nodes_totalcosts)
    #     _duration = max([_ for _ in nodes_totalcosts.values()])

    #     for leaf in self.find_leaves():
    #         blank = self.find_blank(leaf)
    #         blank.duration = _duration - nodes_totalcosts[leaf]
    #     for node in bfs[1:]:
    #         # if not isinstance(nodes_items[node], Branch):
    #         self._tree._tree._cost[node] = self._tree._nodes_items[node].duration

    #     self._tree._tree.evaluate(self._root_node)
    #     return _duration

    # @duration.setter
    # def duration(self, duration: float) -> None:
    #     """Branch object cannot set duration value"""
    #     raise ValueError("Branch object cannot set duration value")

    # def find_leaves(self) -> MutableSequence[int]:
    #     leaves = [
    #         _
    #         for _ in self._tree._tree._tree.children
    #         if _ not in self._tree._tree._tree.parents
    #     ]
    #     retval = []
    #     for leaf in leaves:
    #         _leaf = leaf
    #         while True:
    #             _leaf = self._tree.parentof(_leaf)
    #             if _leaf == self._tree._tree._tree.root:
    #                 break
    #             if _leaf == self._root_node:
    #                 retval.append(leaf)
    #     return retval

    # def find_blank(self, leaf: int) -> Item:
    #     _leaf = leaf
    #     while True:
    #         _leaf = self._tree.parentof(_leaf)
    #         if _leaf == self._tree._tree._tree.root:
    #             raise ValueError("Blank not found")
    #         if isinstance(self._tree._nodes_items[_leaf], Blank):
    #             return self._tree._nodes_items[_leaf]


class Flushright(DequeWithContext):
    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        # このブランチ用のサブツリーを作る
        tree = SequenceTree()
        __rc__.contexts[-1].append(tree)  # with 内の定義の所定の位置にツリーを追加
        # ツリーの根本にブランチアイテムを作る．このブランチの外のアイテムはこのブランチアイテムの次につながる
        branch = FlushrightBranch()
        tree.branch(branch)
        # with 内で定義された item を舐める
        for item in self:
            if isinstance(item, Item):
                # Item ならばそのまま登録
                # ただしパラレルなので注意
                tree.append(Blank(0))
                tree.append(item)
                if branch._root_node is None:
                    raise ValueError("_root_node is None")
                tree._active_node = branch._root_node
            elif isinstance(item, SequenceTree):
                # サブツリーを見つけたら親ツリーとマージする
                # サブツリーのルート直下第 1 アイテムは branch のはず
                # サブツリーを抜けたら _branch._root_node に次のアイテムをぶら下げる
                tree.append(
                    Blank(0)
                )  # flushright は特別に branch の前に Padding をつける
                _tree = item  # ローカルの SequenceTree
                # _branch = _tree._nodes_items[1]
                # if isinstance(_branch, Branch):
                #     if _branch._next_node is None:
                #         raise ValueError("_next_node is None")
                #     next_item = _branch._next_node
                #     if _branch._root_node is None:
                #         raise ValueError("_root_node is None")
                #     root_node = _branch._root_node
                # else:
                #     raise ValueError("1st node is not Branch")
                # ツリーをマージするためにノード番号を更新する
                # offset = tree._active_node
                offset = max(tree._tree._tree.all)
                # 幅優先探索で全アイテムを舐める
                # for child in _tree.breadth_first_search()[1:]:
                #     parent = _tree._tree._tree.parentof(child)
                #     # グローバルツリーに追加する
                #     tree._tree.adopt(parent + offset, child + offset, cost=-1)
                #     # グローバルツリーにアイテムを登録する
                #     tree._nodes_items[child + offset] = _tree._nodes_items[child]
                #     # branch だったら _next_node や _root_node も新しい node 名に更新
                #     _item = _tree._nodes_items[child]
                #     if isinstance(_item, Branch):
                #         # _item._tree = tree
                #         if _item._root_node is None:
                #             raise ValueError("_root_node is None")
                #         _item._root_node += offset
                #         if _item._next_node is None:
                #             raise ValueError("_next_node is None")
                #         _item._next_node += offset
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                offset = max(tree._tree._tree.all)
                root = _tree._tree._tree.root
                for parent, children in _tree._tree._tree.items():
                    if parent == root:
                        tree._tree._tree[tree._active_node] += [
                            _ + offset for _ in children
                        ]
                        continue
                    tree._tree._tree[parent + offset] = [_ + offset for _ in children]
                for node in _tree.breadth_first_search()[1:]:
                    _item = _tree._nodes_items[node]
                    tree._nodes_items[node + offset] = _item
                    tree._tree._cost[node + offset] = -1
                    if isinstance(_item, Branch):
                        # _item._tree = tree
                        if _item._root_node is None:
                            raise ValueError("_root_node is None")
                        _item._root_node += offset
                        if _item._next_node is None:
                            raise ValueError("_next_node is None")
                        _item._next_node += offset  # 親ツリーのカウンターをアップデート
                tree._latest_node = max(_tree.breadth_first_search()) + offset
                # # 親ツリーの追加先を先頭 branch の次に指定
                if branch._root_node is None:
                    raise ValueError("_root_node is None")
                tree._active_node = branch._root_node
        # if branch._next_node is None:
        #     raise ValueError("_next_node is None")
        # # 追加が終わったら親ツリーの追加先を SubSequence の後に指定
        # tree._active_node = branch._next_node


@dataclass
class SampledSequenceBase:
    target_name: str
    first_blank: float = 0  # second
    sampling_period: float = 2e-9  # second

    def asdict(self) -> Dict:
        return {}


@dataclass
class GenSampledSequence(SampledSequenceBase):
    sub_sequences: List[GenSampledSubSequence] = field(default_factory=list)


@dataclass
class GenSampledSubSequence:
    real: NDArray[np.float64]
    imag: NDArray[np.float64]
    post_blank: float  # second
    repeat: int

    def asdict(self) -> Dict:
        return {}


@dataclass
class CapSampledSequence(SampledSequenceBase):
    sub_sequences: List[CapSampledSubSequence] = field(default_factory=list)


@dataclass
class CapSampledSubSequence:
    duration: float  # second
    bost_blank: float  # second

    def asdict(self) -> Dict:
        return {}


# class SubSequence(DequeWithContext):
#     def __init__(self) -> None:
#         self,
if __name__ == "__main__":
    with Sequence() as sequence:
        Slot()
    print(sequence)
