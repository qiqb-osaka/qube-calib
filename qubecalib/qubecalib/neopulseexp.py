from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Final, List, MutableSequence, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .tree import CostedTree, Tree

DEFAULT_SAMPLING_PERIOD: Final[float] = 2e-9


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

    def branch(self, branch: Branch) -> Branch:
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
        return branch

    def place_slots(self) -> None:
        # 深い branch から順に slot を配置する
        for _ in [
            self._nodes_items[_]
            for _ in self.breadth_first_search()[1:]
            if isinstance(self._nodes_items[_], Branch)
        ][::-1]:
            _.place(self)

        # 最終的な slot 配置を確定する（SubSequenceを必ず Toplevel に置くなら必要ないかも？）
        # 各アイテムのコストを更新する
        for node, item in self._nodes_items.items():
            self._tree._cost[node] = item.duration
        total_costs = self._tree.evaluate()
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
    contexts: Final[MutableSequence] = deque()


__rc__: Final[RunningConfig] = RunningConfig()


class ContextNode:
    def __init__(self) -> None:
        if len(__rc__.contexts):
            __rc__.contexts[-1].append(self)


class Item:
    def __init__(
        self,
        duration: Optional[float] = None,
        begin: Optional[float] = None,
    ) -> None:
        self._duration = duration
        self.begin: Optional[float] = begin

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        self._duration = duration

    @property
    def end(self) -> float:
        if self.begin is None or self.duration is None:
            raise ValueError("begin or duration is None")
        return self.begin + self.duration

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(duration={self.duration}, begin={self.begin})"
        )


class Padding(Item):
    def __init__(self, duration: Optional[float] = None) -> None:
        super().__init__(duration)


class Branch(Item):
    def __init__(self) -> None:
        super().__init__()
        self.duration = None
        self._next_node: Optional[int] = None
        self._root_node: Optional[int] = None

    def place(self, tree: SequenceTree) -> None:
        # 最大長を計算する
        for _ in tree.breadth_first_search(self._root_node)[1:]:
            tree._tree._cost[_] = tree._nodes_items[_].duration
        max_duration = max([_ for _ in tree._tree.evaluate(self._root_node).values()])
        # branch の duration は最大長に揃えると同時に cost も確定する
        self.duration = max_duration
        if self._next_node is None:
            raise ValueError("_next_node is None")
        tree._tree._cost[self._next_node] = self.duration

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(duration={self.duration}, begin={self.begin}, next_node={self._next_node}, root_node={self._root_node})"


class Dummy(Item):
    def __init__(self) -> None:
        super().__init__(0)

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        """Branch object cannot set duration value"""
        raise ValueError("Branch object cannot set duration value")

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
    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        self._tree = SequenceTree()
        items: List[SequenceTree | List[Any]] = []
        for item in self:
            if isinstance(item, SequenceTree):
                tree = item
                root = tree._tree._tree.root
                c = tree._tree._tree[root]
                n = tree._nodes_items
                branch = next(iter([n[_] for _ in c if isinstance(n[_], Branch)]))
                if isinstance(branch, SubSequenceBranch):
                    items.append(item)
                    continue
            if not items:
                items.append([])
            if isinstance(items[-1], SequenceTree):
                items.append([])
            items[-1].append(item)
        _items = []
        for item in items:
            if isinstance(item, SequenceTree):
                _items.append(item)
                continue
            tree = SequenceTree()
            tree.branch(SubSequenceBranch())
            SubSequence.create_tree(tree, item)
            _items.append(tree)
        for item in _items:
            # ノード番号を更新してサブツリーをローカルツリーとマージする
            # TODO この辺は tree で吸収すべき
            _tree = item
            # 全てのノードを舐めて最大のインデックスを更新する
            all_nodes = self._tree._tree._tree.all
            if all_nodes:
                offset = max(all_nodes)
            else:
                # 空なら現在のインデックスは 0
                offset = 0
            root = _tree._tree._tree.root
            # サブツリーのアイテムに対して全て
            for parent, children in _tree._tree._tree.items():
                if parent == root:
                    # ローカルツリーの active_node にサブツリーの root をぶら下げる
                    self._tree._tree._tree[self._tree._active_node] += [
                        _ + offset for _ in children
                    ]
                else:
                    # サブツリーのアイテムをローカルツリーの名前空間に変換して移動する
                    self._tree._tree._tree[parent + offset] = [
                        _ + offset for _ in children
                    ]
                self._tree._latest_node = max(self._tree._tree._tree.all)
                # children に Branch がいるか調べる
                branches = {
                    _tree._nodes_items[_]
                    for _ in children
                    if isinstance(
                        _tree._nodes_items[_],
                        Branch,
                    )
                }
                if not branches:
                    continue
                for branch in branches:
                    if not isinstance(branch, Branch):
                        continue
                    # branch の _root_node と _next_node を更新する
                    if not isinstance(branch, Branch):
                        continue
                    if branch._root_node is None:
                        raise ValueError("_root_node is None")
                    if branch._next_node is None:
                        raise ValueError("_next_node is None")
                    branch._root_node += offset
                    branch._next_node += offset
                # Series Branch なので toplevel に Branch が居たら次のアイテムはその Branch の次にぶら下げる
                # toplevel 以外なら次のアイテムの処理へ
                if parent != root:
                    continue
                # children に Branch がいるか調べる
                branches = {
                    _tree._nodes_items[_]
                    for _ in children
                    if isinstance(
                        _tree._nodes_items[_],
                        Branch,
                    )
                }
                if not branches:
                    continue
                # Series Branch なので Branch は高々一つ
                branch = next(iter(branches))
                if not isinstance(branch, Branch):
                    continue
                if branch._next_node is None:
                    raise ValueError("_next_node is None")
                self._tree._active_node = branch._next_node
            # 全てのアイテムをローカルツリーへ複製する
            for node in _tree.breadth_first_search()[1:]:
                self._tree._nodes_items[node + offset] = _tree._nodes_items[node]
                self._tree._tree._cost[node + offset] = -1
        self._validate_nodes_items()  # for debug

    def _get_tree(self) -> Tree:
        if self._tree is None:
            raise ValueError("SequenceTree is not prepared.")
        return self._tree._tree._tree

    def _get_group_items_by_target(
        self,
    ) -> Dict[str, Dict[int, MutableSequence[Waveform | TargetHolder]]]:
        nodes_items = self._tree._nodes_items
        subsequences = [
            _
            for _ in self._tree._nodes_items.values()
            if isinstance(_, SubSequenceBranch)
        ]
        return {
            _.target: {
                sub._next_node: [
                    item
                    for node, item in nodes_items.items()
                    if isinstance(item, TargetHolder)
                    if item.target == _.target
                    if node in self._tree.breadth_first_search(sub._root_node)
                ]
                for sub in subsequences
                if sub._next_node is not None
            }
            for _ in nodes_items.values()
            if isinstance(_, TargetHolder)
        }

    def _validate_nodes_items(self) -> None:
        for node, item in self._tree._nodes_items.items():
            if not isinstance(item, Branch):
                continue
            if node != item._next_node:
                raise ValueError("invalid status of item")

    def _create_gen_sampled_sequence(
        self,
        target_name: str,
        targets_items: Dict[str, Dict[int, MutableSequence[Waveform | TargetHolder]]],
        sampling_period: float = 2e-9,
    ) -> GenSampledSequence:
        items: Dict[int, MutableSequence[Waveform]] = {
            edge: [slot for slot in slots if isinstance(slot, Waveform)]
            for edge, slots in targets_items[target_name].items()
        }
        edges_items = {
            _: __
            for _, __ in self._tree._nodes_items.items()
            if isinstance(__, SubSequenceBranch)
        }
        subseq_edges = [
            edge for edge, _ in items.items() if _
        ]  # waveform を保持する subseq の edge_number
        subseqs = [
            edges_items[_]
            for _ in subseq_edges
            if isinstance(edges_items[_], SubSequenceBranch)
        ]
        nodes: List[float] = sum(
            [[0]] + [[_.begin, _.end] for _ in subseqs],
            [],
        )
        blanks = [end - begin for begin, end in zip(nodes[:-1:2], nodes[1::2])] + [None]
        sampled_subsequences = [
            GenSampledSubSequence(
                real=np.real(v),
                imag=np.imag(v),
                repeats=repeats,
                post_blank=round(blank / sampling_period)
                if blank is not None
                else None,
            )
            for (v, _, _), repeats, blank in [
                (
                    Sampler(subseq, slots).sample(
                        over_sampling_ratio=1, difference_type="center"
                    ),
                    subseq.repeats,
                    post_blank,
                )
                for subseq, slots, post_blank in zip(
                    [edges_items[_] for _ in subseq_edges],
                    [items[_] for _ in subseq_edges],
                    [_ for _ in blanks][1:],
                )
            ]
        ]
        if blanks[0] is None:
            raise ValueError("first element of blanks is None")
        return GenSampledSequence(
            target_name=target_name,
            prev_blank=round(blanks[0] / sampling_period),
            post_blank=None,
            repeats=None,
            sampling_period=sampling_period,
            sub_sequences=sampled_subsequences,
        )

    def _create_sampled_sequence(self) -> Dict[str, SampledSequenceBase]:
        targets_items = self._get_group_items_by_target()
        return {
            _: self._create_gen_sampled_sequence(_, targets_items)
            for _ in targets_items
        }

    def _create_cap_sampled_sequence(
        self,
        target_name: str,
        targets_items: Dict[str, Dict[int, MutableSequence[Waveform | TargetHolder]]],
        sampling_period: float = 2e-9,
    ) -> CapSampledSequence:
        edges_items = self._tree._nodes_items
        # waveform を保持する subseq の edge_number
        subseq_edges = [edge for edge, _ in targets_items[target_name].items() if _]
        # waveform を保持する subseq
        subseqs = [
            edges_items[_]
            for _ in subseq_edges
            # if isinstance(edges_items[_], SubSequenceBranch)
        ]
        # subseq の境界をサンプリング周期にアライメントする（負の無限大へ丸める）
        _subseqs = Utils.align_items(
            sorted(
                [_ for _ in subseqs],
                key=lambda x: x.begin,
            )
        )
        # 全体の領域長も含めて alignment した subseq のノードを抽出する
        _nodes: List[Any] = sum(
            [[0.0]] + [[_.begin, _.end] for _ in subseqs] + [[None]],
            [],
        )
        # 奇数 edge 長を blank, 偶数 edge 長を aligned subseq 長として抽出
        _blanks = [
            end - begin if end is not None and begin is not None else None
            for begin, end in zip(_nodes[:-1:2], _nodes[1::2])
        ]
        _slots = {
            subseq_edge: targets_items[target_name][subseq_edge]
            for subseq_edge in subseq_edges
        }
        return CapSampledSequence(
            target_name,
            prev_blank=round(_blanks[0] / sampling_period),
            post_blank=None,
            repeats=None,
            sub_sequences=[
                CapSampledSubSequence(
                    capture_slots=[
                        CaptureSlots(
                            duration=round(duration / sampling_period),
                            post_blank=round(blank / sampling_period)
                            if blank is not None
                            else None,
                        )
                        for duration, blank in zip(
                            *Utils._create_duration_and_blanks(__slots, _subseq)
                        )
                    ],
                    post_blank=round(blank / sampling_period)
                    if blank is not None
                    else None,
                    repeats=None,
                )
                for __slots, _subseq, blank in zip(
                    _slots.values(),
                    # [_ for _ in _subseqs if isinstance(_, SubSequenceBranch)],
                    _subseqs,
                    _blanks[1:],
                )
            ],
        )

    def _create_cap_sampled_sequence_(self) -> Dict[str, SampledSequenceBase]:
        targets_items = self._get_group_items_by_target()
        return {
            _: self._create_cap_sampled_sequence(_, targets_items)
            for _ in targets_items
        }


class Utils:
    @classmethod
    def align_items(
        cls,
        items: MutableSequence[Item],
        sampling_period: float = 2e-9,
    ) -> MutableSequence[Item]:
        dt = sampling_period

        return [
            Item(
                duration=floor(_.end, dt) - floor(_.begin, dt),
                begin=floor(_.begin, dt),
            )
            for _ in items
        ]

    @classmethod
    def _create_duration_and_blanks(
        cls,
        ranges: MutableSequence[Waveform | TargetHolder],
        frame: SubSequenceBranch,
    ) -> Tuple[MutableSequence[float], MutableSequence[Optional[float]]]:
        slots = sorted(ranges, key=lambda x: x.begin)
        _slots = Utils.align_items([_ for _ in slots if isinstance(_, Item)])
        _durations = [_.duration for _ in _slots if isinstance(_.duration, float)]
        _blanks = [
            post.begin - prev.end
            if post.begin is not None and prev.end is not None
            else None
            for prev, post in zip(_slots[:-1], _slots[1:])
        ] + [
            frame.end - _slots[-1].end
            if frame.end is not None and _slots[-1].end is not None
            else None
        ]
        return _durations, _blanks


class SubSequenceBranch(Branch):
    def __init__(self, repeats: int = 1) -> None:
        super().__init__()
        self.repeats = repeats

    @property
    def repeats(self) -> int:
        return self._repeats

    @repeats.setter
    def repeats(self, repeats: int) -> None:
        if not isinstance(repeats, int):
            raise ValueError("repeats must be int")
        self._repeats = repeats

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(duration={self.duration}, begin={self.begin}, next_node={self._next_node}, root_node={self._root_node}, repeats={self.repeats})"


class SubSequence(DequeWithContext):
    def __init__(self, repeats: int = 1) -> None:
        if not isinstance(repeats, int):
            raise ValueError("repeats must be int")
        self._repeats = repeats

    @property
    def repeats(self) -> int:
        return self._repeats

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        # このブランチ用のローカルツリーを作る
        tree = SequenceTree()
        # ツリーの根本にブランチアイテムを作る．このブランチの外のアイテムはこのブランチアイテムの次につながる
        tree.branch(SubSequenceBranch(repeats=self.repeats))
        SubSequence.create_tree(tree, self)
        # with 内の定義の所定の位置にツリーを追加
        __rc__.contexts[-1].append(tree)

    @classmethod
    def create_tree(cls, tree: SequenceTree, items: MutableSequence) -> None:
        # with 内で定義された item を舐める
        for item in items:
            if isinstance(item, Item):
                # Item ならばそのまま登録
                slot = item
                tree.append(slot)
            elif isinstance(item, SequenceTree):
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                # TODO この辺は tree で吸収すべき
                # サブツリーを見つけたらローカルツリーとマージする
                _tree = item
                # 全てのローカルノードを舐めて最大のインデックスを更新する
                all_nodes = tree._tree._tree.all
                if all_nodes:
                    offset = max(all_nodes)
                else:
                    # 空なら現在のインデックスは 0
                    offset = 0
                root = _tree._tree._tree.root
                # サブツリーのアイテムに対して全て
                for parent, children in _tree._tree._tree.items():
                    if parent == root:
                        # ローカルツリーの active_node にサブツリーの root をぶら下げる
                        tree._tree._tree[tree._active_node] += [
                            _ + offset for _ in children
                        ]
                    else:
                        # サブツリーのアイテムをローカルツリーの名前空間に変換して移動する
                        tree._tree._tree[parent + offset] = [
                            _ + offset for _ in children
                        ]
                    tree._latest_node = max(tree._tree._tree.all)
                    # children に Branch がいるか調べる
                    branches = {
                        _tree._nodes_items[_]
                        for _ in children
                        if isinstance(
                            _tree._nodes_items[_],
                            Branch,
                        )
                    }
                    if not branches:
                        continue
                    for branch in branches:
                        if not isinstance(branch, Branch):
                            continue
                        # branch の _root_node と _next_node を更新する
                        if not isinstance(branch, Branch):
                            continue
                        if branch._root_node is None:
                            raise ValueError("_root_node is None")
                        if branch._next_node is None:
                            raise ValueError("_next_node is None")
                        branch._root_node += offset
                        branch._next_node += offset
                    # Series Branch なので toplevel に Branch が居たら次のアイテムはその Branch の次にぶら下げる
                    # toplevel 以外なら次のアイテムの処理へ
                    if parent != root:
                        continue
                    # children に Branch がいるか調べる
                    branches = {
                        _tree._nodes_items[_]
                        for _ in children
                        if isinstance(
                            _tree._nodes_items[_],
                            Branch,
                        )
                    }
                    if not branches:
                        continue
                    # Series Branch なので Branch は高々一つ
                    branch = next(iter(branches))
                    if not isinstance(branch, Branch):
                        continue
                    if branch._next_node is None:
                        raise ValueError("_next_node is None")
                    tree._active_node = branch._next_node
                # 全てのアイテムをローカルツリーへ複製する
                for node in _tree.breadth_first_search()[1:]:
                    tree._nodes_items[node + offset] = _tree._nodes_items[node]
                    tree._tree._cost[node + offset] = -1
        # return tree


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
        # 外側の context にローカルツリーを渡す
        __rc__.contexts[-1].append(tree)
        # ローカルツリーのルート直下を branch して branch item を登録する
        tree.branch(SeriesBranch())
        # with 内で定義された item を舐める
        for item in self:
            if isinstance(item, Item):
                # Item ならばそのまま登録
                slot = item
                tree.append(slot)
            elif isinstance(item, SequenceTree):
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                # TODO この辺は tree で吸収すべき
                # サブツリーを見つけたらローカルツリーとマージする
                _tree = item
                # 全てのローカルノードを舐めて最大のインデックスを更新する
                all_nodes = tree._tree._tree.all
                if all_nodes:
                    offset = max(all_nodes)
                else:
                    # 空なら現在のインデックスは 0
                    offset = 0
                root = _tree._tree._tree.root
                # サブツリーのアイテムに対して全て
                for parent, children in _tree._tree._tree.items():
                    if parent == root:
                        # ローカルツリーの active_node にサブツリーの root をぶら下げる
                        tree._tree._tree[tree._active_node] += [
                            _ + offset for _ in children
                        ]
                    else:
                        # サブツリーのアイテムをローカルツリーの名前空間に変換して移動する
                        tree._tree._tree[parent + offset] = [
                            _ + offset for _ in children
                        ]
                    tree._latest_node = max(tree._tree._tree.all)
                    # children に Branch がいるか調べる
                    branches = {
                        _tree._nodes_items[_]
                        for _ in children
                        if isinstance(
                            _tree._nodes_items[_],
                            Branch,
                        )
                    }
                    if not branches:
                        continue
                    for branch in branches:
                        if not isinstance(branch, Branch):
                            continue
                        # branch の _root_node と _next_node を更新する
                        if not isinstance(branch, Branch):
                            continue
                        if branch._root_node is None:
                            raise ValueError("_root_node is None")
                        if branch._next_node is None:
                            raise ValueError("_next_node is None")
                        branch._root_node += offset
                        branch._next_node += offset
                    # Series Branch なので toplevel に Branch が居たら次のアイテムはその Branch の次にぶら下げる
                    # toplevel 以外なら次のアイテムの処理へ
                    if parent != root:
                        continue
                    # children に Branch がいるか調べる
                    branches = {
                        _tree._nodes_items[_]
                        for _ in children
                        if isinstance(
                            _tree._nodes_items[_],
                            Branch,
                        )
                    }
                    if not branches:
                        continue
                    # Series Branch なので Branch は高々一つ
                    branch = next(iter(branches))
                    if not isinstance(branch, Branch):
                        continue
                    if branch._next_node is None:
                        raise ValueError("_next_node is None")
                    tree._active_node = branch._next_node
                # 全てのアイテムをローカルツリーへ複製する
                for node in _tree.breadth_first_search()[1:]:
                    tree._nodes_items[node + offset] = _tree._nodes_items[node]
                    tree._tree._cost[node + offset] = -1


class FlushleftBranch(Branch):
    pass


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
        # ツリーの根本にブランチアイテムを作る．このブランチの外のアイテムはこのブランチアイテムの次につながる
        branch = tree.branch(FlushleftBranch())
        # with 内で定義された item を舐める
        if branch._root_node is None:
            raise ValueError("_root_node is None")
        tree._active_node = branch._root_node
        for item in self:
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
                # ツリーをマージするためにノード番号を更新する
                offset = max(tree._tree._tree.all)
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                # TODO この辺は tree で吸収すべき
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
                # 親ツリーのカウンターをアップデート
                tree._latest_node = max(_tree.breadth_first_search()) + offset
                # 親ツリーの追加先を先頭 branch の次に指定
                if branch._root_node is None:
                    raise ValueError("_root_node is None")
                tree._active_node = branch._root_node


class FlushrightBranch(Branch):
    def place(self, tree: SequenceTree) -> None:
        super().place(tree)
        # # blank が 0 の状態で最大長を計算する
        max_duration = self._duration
        # _root_node にぶら下がっている blank node を取得する
        for _ in tree._tree._tree[self._root_node]:
            # blank にぶら下がっているノードの最大長を取得する
            branch_duration = max([_ for _ in tree._tree.evaluate(_).values()])
            # 右揃えになるよう blank を調整するかつ cost も確定する
            tree._nodes_items[_].duration = max_duration - branch_duration
            tree._tree._cost[_] = tree._nodes_items[_].duration


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
        branch = tree.branch(FlushrightBranch())
        # with 内で定義された item を舐める
        for item in self:
            if isinstance(item, Item):
                # Item ならばそのまま登録
                # ただしパラレルなので注意
                tree.append(Padding(0))
                tree.append(item)
                if branch._root_node is None:
                    raise ValueError("_root_node is None")
                tree._active_node = branch._root_node
            elif isinstance(item, SequenceTree):
                # サブツリーを見つけたら親ツリーとマージする
                # サブツリーのルート直下第 1 アイテムは branch のはず
                # サブツリーを抜けたら _branch._root_node に次のアイテムをぶら下げる
                tree.append(
                    Padding(0)
                )  # flushright は特別に branch の前に Padding をつける
                _tree = item  # ローカルの SequenceTree
                offset = max(tree._tree._tree.all)
                # ノード番号を更新してサブツリーをローカルツリーとマージする
                # TODO この辺は tree で吸収すべき
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


def ceil(value: float, unit: float = 1) -> float:
    """valueの値を指定したunitの単位でその要素以上の最も近い数値に丸める（正の無限大へ丸める）

    Args:
        value (float): 対象の値
        unit (float, optional): 丸める単位. Defaults to 1.

    Returns:
        float: 丸めた値
    """
    exponent = math.floor(math.log10(unit))
    mantissa = unit * 10 ** (-exponent)
    retval = None
    if exponent < 0:
        retval = (
            math.ceil(value * 10 ** (-exponent) / mantissa) * mantissa * 10**exponent
        )
    else:
        retval = (
            math.ceil(value / 10 ** (exponent) / mantissa) * mantissa * 10**exponent
        )
    if (retval - unit) - value < 1e-16:
        return retval - unit
    else:
        return retval


def floor(value: float, unit: float = 1) -> float:
    """valueの値を指定したunitの単位でその要素以下の最も近い数値に丸める（負の無限大へ丸める）

    Args:
        value (float): 対象の値
        unit (float, optional): 丸める単位. Defaults to 1.

    Returns:
        float: 丸めた値
    """
    exponent = math.floor(math.log10(unit))
    mantissa = unit * 10 ** (-exponent)
    retval = None
    if exponent < 0:
        retval = (
            math.floor(value * 10 ** (-exponent) / mantissa) * mantissa * 10**exponent
        )
    else:
        retval = (
            math.floor(value / 10 ** (exponent) / mantissa) * mantissa * 10**exponent
        )
    if (retval + unit) - value < 1e-16:
        return retval + unit
    else:
        return retval


class TargetHolder:
    def set_target(self, target: str) -> TargetHolder:
        self.target = target
        return self


class Slot(ContextNode, Item):
    def __init__(self, duration: Optional[float] = None) -> None:
        super().__init__()
        Item.__init__(self, duration)


class Blank(Slot):
    pass


class Range(Slot, TargetHolder):
    def __init__(self, duration: Optional[float] = None) -> None:
        super().__init__(duration)


class Modifier(Slot, TargetHolder):
    def __init__(self) -> None:
        super().__init__(0)
        TargetHolder.__init__(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(begin={self.begin})"

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        """Branch object cannot set duration value"""
        raise ValueError("Branch object cannot set duration value")


class Waveform(Slot):
    def __init__(self, duration: Optional[float] = None) -> None:
        self.amplitude = 1.0
        self.phase = 0.0  # radian
        self.__iq__: Optional[NDArray] = None

        super().__init__(duration=duration)

    def func(self, t: float) -> complex:
        """正規化複素振幅 (1 + j0), ローカル時間軸 (begin=0) で iq 波形を定義したもの"""
        raise NotImplementedError()
        return 0

    def _func(self, t: float) -> complex:
        """func() に対して複素振幅とローカル時間軸を配慮したもの"""
        if self.begin is None or self.duration is None:
            raise ValueError(
                "Either or both 'begin' and 'duration' are not initialized."
            )
        if t < self.begin or self.begin + self.duration < t:
            return 0 + 0j
        return self.func(t - self.begin) * self.amplitude * np.exp(1j * self.phase)

    def ufunc(self, t: NDArray) -> NDArray:
        return np.frompyfunc(self._func, 1, 1)(t).astype(complex)


class RaisedCosFlatTop(Waveform, TargetHolder):
    """
    Raised Cosine FlatTopパルス

    Attributes
    ----------
    ampl : float
        全体にかかる振幅
    phase : float
        全体にかかる位相[rad]
    rise_time: float
        立ち上がり・立ち下がり時間[ns]
    """

    def __init__(
        self,
        duration: Optional[float] = None,
        rise_time: Optional[float] = None,
    ):
        self._rise_time = rise_time

        super().__init__(duration)

    def func(self, t: float) -> complex:
        if self.duration is None or self.rise_time is None:
            raise ValueError("duration or rise_time is None")
        flattop_duration = self.duration - self.rise_time * 2

        t1 = 0
        t2 = t1 + self.rise_time  # 立ち上がり完了時刻
        t3 = t2 + flattop_duration  # 立ち下がり開始時刻
        t4 = t3 + self.rise_time  # 立ち下がり完了時刻

        if (t1 <= t) & (t < t2):  # 立ち上がり時間領域の条件ブール値
            # 立ち上がり時間領域の値
            return (1.0 - np.cos(np.pi * (t - t1) / self.rise_time)) / 2.0
        if (t2 <= t) & (t < t3):  # 一定値領域の条件ブール値
            # 一定値領域の値
            return 1.0 + 0.0j
        if (t3 <= t) & (t < t4):  # 立ち下がり時間領域の条件ブール値
            # 立ち下がり時間領域の値
            return (1.0 - np.cos(np.pi * (t4 - t) / self.rise_time)) / 2.0
        return 0.0 + 0.0j

    @property
    def rise_time(self) -> Optional[float]:
        return self._rise_time

    @rise_time.setter
    def rise_time(self, rise_time: float) -> None:
        if not isinstance(self._rise_time, float):
            raise ValueError(f"{type(rise_time)} is invalid")
        if self.duration is None:
            raise ValueError("duration is None")
        # print(rise_time, self.duration)
        if self.duration < rise_time * 2:
            raise ValueError(f"{rise_time} is too long")

        self._rise_time = rise_time


class Rectangle(Waveform, TargetHolder):
    def __init__(self, duration: Optional[float] = None):
        super().__init__(duration)

    def func(self, t: float) -> complex:
        return 1 + 0j


class Arbit(Waveform, TargetHolder):
    """ "サンプリング点を直接与えるためのオブジェクト"""

    DEFAULT_SAMPLING_PERIOD: Final[float] = 2e-9

    def __init__(
        self,
        duration: Optional[float] = None,
        init: Optional[complex] = None,
    ):
        super().__init__(duration)
        if init is None:
            self.init = 0 + 0j
        else:
            self.init = init
        self.__iq__: Optional[NDArray] = None

    def func(self, t: float) -> complex:
        """iq データを格納している numpy array に従って iq(t) の値を返す"""
        # ローカル時間軸を返すのに注意
        if self.__iq__ is None:
            raise ValueError("__iq__ is None")
        if self.begin is None or self.duration is None:
            raise ValueError("begin or duration is None")

        d, s = self.duration, self.DEFAULT_SAMPLING_PERIOD
        if 0 <= t and t < d:
            t0 = np.arange(round(d // s) + 1) * s
            boolean = (t0 <= t) * (t - s < t0)
            return self.__iq__[boolean][0]

        return 0 + 0j

    @property
    def iq(self) -> NDArray:
        """iq データを格納している numpy array への参照を返す"""
        if self.duration is None:
            raise ValueError("duration is None")
        d, s = self.duration, self.DEFAULT_SAMPLING_PERIOD
        # 初回アクセス or 前回アクセスから duration が更新されていれば ndarray を 0 + j0 で再生成
        if self.__iq__ is None or round(d // s) + 1 != self.__iq__.shape[0]:
            self.__iq__ = np.zeros(round(d // s) + 1).astype(complex)  # iq data

        return self.__iq__


class Sampler:
    @classmethod
    def create_sampling_timing(
        cls,
        begin: float,
        duration: float,
        over_sampling_ratio: int = 1,
        difference_type: str = "back",
        endpoint: bool = False,
        sampling_period: float = DEFAULT_SAMPLING_PERIOD,
    ) -> NDArray[np.float]:
        """サンプル時系列 t 生成する。ratio 倍にオーバーサンプルする。"""

        dt = 1 * sampling_period / over_sampling_ratio
        if endpoint:
            duration += dt
        v = np.arange(
            math.ceil(begin / dt) * dt, math.ceil((begin + duration + dt) / dt) * dt, dt
        )
        if difference_type == "back":
            return v[:-1]
        elif difference_type == "center":
            return 0.5 * (v[1:] + v[:-1])
        else:
            raise ValueError(f"difference_type={difference_type} is not supported")

    @classmethod
    def _sample(
        self,
        sampling_timing: NDArray[np.float32],
        waveforms: MutableSequence[Waveform],
    ) -> NDArray[np.complex64]:
        def func(t: float) -> complex:
            for w in waveforms:
                if w.begin is None or w.duration is None:
                    raise ValueError("begin or duration is None")
                if (w.begin <= t) and (t < w.begin + w.duration):
                    return w._func(t)
            return 0 + 0j

        return np.frompyfunc(func, 1, 1)(sampling_timing).astype(complex)

    def __init__(
        self,
        branch: SubSequenceBranch,
        waveforms: MutableSequence[Waveform],
    ) -> None:
        if not isinstance(branch, SubSequenceBranch):
            raise ValueError("branch should be SubSequenceBranch")
        self._branch = branch
        self._waveforms = waveforms

    def sample(
        self,
        over_sampling_ratio: int = 1,
        difference_type: str = "back",
        sampling_period: float = DEFAULT_SAMPLING_PERIOD,
    ) -> Tuple[
        NDArray[np.complex64],
        NDArray[np.float32],
        Optional[NDArray[np.float32]],
    ]:
        begin = self._branch.begin
        duration = self._branch.duration
        if begin is None or duration is None:
            raise ValueError("begin or duration of branch is None")
        if difference_type == "center":
            ts = self.create_sampling_timing(
                begin,
                duration,
                over_sampling_ratio=over_sampling_ratio,
                difference_type="center",
                endpoint=False,
                sampling_period=sampling_period,
            )
            t = self.create_sampling_timing(
                begin,
                duration,
                over_sampling_ratio=over_sampling_ratio,
                difference_type="back",
                endpoint=True,
                sampling_period=sampling_period,
            )
            return self._sample(ts, self._waveforms), t, ts
        else:
            ts = self.create_sampling_timing(
                begin,
                duration,
                over_sampling_ratio=over_sampling_ratio,
                difference_type=difference_type,
                endpoint=False,
                sampling_period=sampling_period,
            )
            return self._sample(ts, self._waveforms), ts, None


# TODO 空の sub_sequence の処理が省かれているので追加する
# TODO subsequence の repeats 処理が曖昧なので追加する


# @dataclass
# class CapSampledSubSequence:
#     duration: float  # second
#     bost_blank: float  # second

#     def asdict(self) -> Dict:
#         return {}


@dataclass
class SampledSequenceBase:
    target_name: str
    prev_blank: int = 0  # words
    post_blank: Optional[int] = None
    repeats: Optional[int] = None
    sampling_period: float = 2e-9  # second

    def asdict(self) -> Dict:
        return asdict(self)


@dataclass
class GenSampledSequence(SampledSequenceBase):
    sub_sequences: List[GenSampledSubSequence] = field(default_factory=list)

    def asdict(self) -> Dict:
        return super().asdict() | {
            "sub_sequences": [_.asdict() for _ in self.sub_sequences],
            "class": self.__class__.__name__,
        }


@dataclass
class GenSampledSubSequence:
    real: NDArray[np.float64]
    imag: NDArray[np.float64]
    post_blank: Optional[int]  # samples
    repeats: int

    def asdict(self) -> Dict:
        return {
            "real": self.real.tolist(),
            "imag": self.imag.tolist(),
            "post_blank": self.post_blank,
            "repeats": self.repeats,
        }


@dataclass
class CapSampledSequence(SampledSequenceBase):
    sub_sequences: List[CapSampledSubSequence] = field(default_factory=list)


@dataclass
class CapSampledSubSequence:
    capture_slots: MutableSequence[CaptureSlots]
    # duration: int  # samples
    post_blank: Optional[int]  # samples
    repeats: Optional[int]

    def asdict(self) -> Dict:
        return {}


@dataclass
class CaptureSlots:
    duration: int  # samples
    post_blank: Optional[int]  # samples

    def asdict(self) -> Dict:
        return {}