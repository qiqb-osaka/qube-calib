from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Final, MutableSequence, Optional

import numpy as np
from numpy.typing import NDArray

from .tree import CostedTree, Tree

DEFAULT_SAMPLING_PERIOD: float = 2.0


@dataclass
class RunningConfig:
    contexts: Final[MutableSequence] = field(default_factory=deque)


_rc: Final[RunningConfig] = RunningConfig()


class SequenceTree:
    def __init__(self) -> None:
        self._tree = CostedTree()
        self._active_node = 0
        self._latest_node = 0
        self._nodes_items: dict[int, Item] = {}

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

    def breadth_first_search(self, start: Optional[int] = None) -> MutableSequence[int]:
        return self._tree.breadth_first_search(start)


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
        _rc.contexts.append(self)
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        _rc.contexts.pop()


class Sequence(DequeWithContext):
    def __enter__(self) -> Sequence:
        super().__enter__()
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        self._tree = SequenceTree()
        items: MutableSequence[SequenceTree | MutableSequence[Any]] = []
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
    ) -> dict[str, dict[int, MutableSequence[Slot]]]:
        nodes_items = self._tree._nodes_items
        subsequences = [
            _
            for _ in self._tree._nodes_items.values()
            if isinstance(_, SubSequenceBranch)
        ]
        nodes_by_sub = {
            sub: self._tree.breadth_first_search(sub._root_node)
            for sub in subsequences
            if sub._next_node is not None
        }
        result: dict[str, dict[int, MutableSequence[Slot]]] = {}
        for node, item in nodes_items.items():  # Sequence に属する Slot 毎に
            if not isinstance(item, Slot):
                continue
            for target in item.targets:
                if target not in result:
                    result[target] = {}
                for sub in subsequences:  # 空でない subsequence 毎に
                    if sub._next_node is None:
                        continue
                    if sub._next_node not in result[target]:
                        result[target][sub._next_node] = []
                    if (
                        isinstance(item, Slot)
                        and target in item.targets
                        and node in nodes_by_sub[sub]
                    ):
                        result[target][sub._next_node].append(item)

        return result

    def _validate_nodes_items(self) -> None:
        for node, item in self._tree._nodes_items.items():
            if not isinstance(item, Branch):
                continue
            if node != item._next_node:
                raise ValueError("invalid status of item")

    def _create_gen_sampled_sequence(
        self,
        target_name: str,
        targets_items: dict[str, dict[int, MutableSequence[Waveform | Modifier]]],
        sampling_period: float = DEFAULT_SAMPLING_PERIOD,
    ) -> GenSampledSequence:
        # edge と item の対応マップ
        items: dict[int, MutableSequence[Waveform | Modifier]] = {
            edge: [
                slot
                for slot in slots
                if isinstance(slot, Waveform) or isinstance(slot, Modifier)
            ]
            for edge, slots in targets_items[target_name].items()
        }
        # edge と subseq との対応マップ
        edges_items = {
            _: __
            for _, __ in self._tree._nodes_items.items()
            if isinstance(__, SubSequenceBranch)
        }
        # 空でない（waveform を保持する）subseq の edge_number
        subseq_edges = [edge for edge, _ in items.items() if _]
        # 空でない subseq のリスト
        subseqs = [
            edges_items[_]
            for _ in subseq_edges
            if isinstance(edges_items[_], SubSequenceBranch)
        ]
        # subseq のノード
        nodes: list[float] = sum(
            [[0]] + [[_.begin, _.end - _.post_blank] for _ in subseqs],
            [],
        )
        blanks = [end - begin for begin, end in zip(nodes[:-1:2], nodes[1::2])] + [None]
        sampled_subsequences = [
            GenSampledSubSequence(
                real=np.real(v),
                imag=np.imag(v),
                repeats=repeats,
                post_blank=(
                    round(blank / sampling_period) if blank is not None else None
                ),
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

    @classmethod
    def _is_cap_target(
        cls,
        sub_seq_edges__items: dict[int, MutableSequence[Item]],
    ) -> bool:
        # 各々の subseq 配下の items が Capture のみを含むか 空[] である
        return all(
            [
                not bool(_) or all([isinstance(__, Capture) for __ in _])
                for _ in sub_seq_edges__items.values()
            ]
        )

    @classmethod
    def _is_gen_target(
        cls,
        sub_seq_edges__items: dict[int, MutableSequence[Item]],
    ) -> bool:
        # 各々の subseq 配下の items が Waveform のみを含むか 空[] である
        return all(
            [
                not bool(_) or all([isinstance(__, Waveform) for __ in _])
                for _ in sub_seq_edges__items.values()
            ]
        )

    def _create_sampled_sequence(
        self,
    ) -> tuple[
        dict[str, GenSampledSequence],
        dict[str, CapSampledSequence],
    ]:
        group_items = self._get_group_items_by_target()
        _ = {
            target_name: {
                num: [
                    item
                    for item in items
                    if isinstance(item, Waveform) or isinstance(item, Modifier)
                ]
                for num, items in num_items.items()
            }
            for target_name, num_items in group_items.items()
        }
        __ = {
            target_name: {num: items for num, items in num_items.items() if items}
            for target_name, num_items in _.items()
        }
        targets_items_gen: dict[
            str, dict[int, MutableSequence[Waveform | Modifier]]
        ] = {
            target_name: num_items for target_name, num_items in __.items() if num_items
        }
        _ = {
            target_name: {
                num: [item for item in items if isinstance(item, Capture)]
                for num, items in num_items.items()
            }
            for target_name, num_items in group_items.items()
        }
        __ = {
            target_name: {num: items for num, items in num_items.items() if items}
            for target_name, num_items in _.items()
        }
        targets_items_cap: dict[str, dict[int, MutableSequence[Capture]]] = {
            target_name: num_items for target_name, num_items in __.items() if num_items
        }
        return (
            {
                _: self._create_gen_sampled_sequence(_, targets_items_gen)
                for _ in targets_items_gen
            },
            {
                _: self._create_cap_sampled_sequence(_, targets_items_cap)
                for _ in targets_items_cap
            },
        )

    def _create_cap_sampled_sequence(
        self,
        target_name: str,
        targets_items: dict[str, dict[int, MutableSequence[Capture]]],
        sampling_period: float = DEFAULT_SAMPLING_PERIOD,
    ) -> CapSampledSequence:
        edges_items: dict[int, Item] = self._tree._nodes_items

        # waveform を保持する（空でない） subseq の edge_number を begin に対して昇順に並べたもの
        def sort_key(x: int) -> float:
            b = edges_items[x].begin
            if b is None:
                raise ValueError("begin is None")
            return b

        subseq_edges = sorted(
            [edge for edge, _ in targets_items[target_name].items() if _],
            key=sort_key,
        )
        # waveform を保持する subseq
        subseqs: dict[int, SubSequenceBranch] = {
            edge: _
            for edge, _ in [[edge, edges_items[edge]] for edge in subseq_edges]
            if isinstance(_, SubSequenceBranch) and isinstance(edge, int)
        }
        # subseq の境界をサンプリング周期にアライメントする（負の無限大へ丸める）
        _subseqs = {
            edge: items
            for edge, items in zip(
                subseq_edges,
                Utils.align_items(
                    [
                        Item(
                            duration=_._total_duration_contents,
                            begin=_.begin,
                        )
                        for _ in [subseqs[edge] for edge in subseq_edges]
                        if isinstance(_, SubSequenceBranch)
                        if _._total_duration_contents is not None
                    ],
                ),
            )
        }
        _slots = {
            subseq_edge: Utils.align_items(
                sorted(
                    [
                        Item(duration=_.duration, begin=_.begin)
                        for _ in targets_items[target_name][subseq_edge]
                    ],
                    key=lambda x: x.begin if x.begin is not None else -math.inf,
                )
            )
            for subseq_edge in subseq_edges
        }
        # subseq 毎に slot を含んだ blank と duration の境界 node リストを生成する
        _nodes: dict[int, MutableSequence[float] | MutableSequence] = {
            _: sum(
                [[_subseqs[_].begin]]
                + [[__.begin, __.end] for __ in _slots[_]]
                + [[_subseqs[_].end]],
                [],
            )
            for _ in subseq_edges
        }
        _blanks = {
            _: [end - begin for begin, end in zip(_nodes[_][:-1:2], _nodes[_][1::2])]
            for _ in subseq_edges
        }
        _durations = {
            _: [end - begin for begin, end in zip(_nodes[_][1:-1:2], _nodes[_][2::2])]
            for _ in subseq_edges
        }
        _subseqs_original = {
            edge: items
            for edge, items in zip(
                subseq_edges,
                [
                    Item(
                        duration=_._total_duration_contents,
                        begin=_.begin,
                    )
                    for _ in [subseqs[edge] for edge in subseq_edges]
                    if isinstance(_, SubSequenceBranch)
                    if _._total_duration_contents is not None
                ],
            )
        }
        _slots_original = {
            subseq_edge: sorted(
                [
                    Item(
                        duration=item.duration,
                        begin=item.begin,
                    )
                    for item in targets_items[target_name][subseq_edge]
                ],
                key=lambda x: x.begin if x.begin is not None else -math.inf,
            )
            for subseq_edge in subseq_edges
        }
        _nodes_original: dict[int, MutableSequence[float] | MutableSequence] = {
            _: sum(
                [[_subseqs_original[_].begin]]
                + [[__.begin, __.end] for __ in _slots_original[_]]
                + [[_subseqs_original[_].end]],
                [],
            )
            for _ in subseq_edges
        }
        _blanks_original = {
            _: [
                end - begin
                for begin, end in zip(
                    _nodes_original[_][:-1:2], _nodes_original[_][1::2]
                )
            ]
            for _ in subseq_edges
        }
        _durations_original = {
            _: [
                end - begin
                for begin, end in zip(
                    _nodes_original[_][1:-1:2], _nodes_original[_][2::2]
                )
            ]
            for _ in subseq_edges
        }

        toplevel_prev_blank = 0
        toplevel_post_blank: Optional[float] = None
        return CapSampledSequence(
            target_name,
            prev_blank=round(toplevel_prev_blank / sampling_period),
            post_blank=(
                round(toplevel_post_blank / sampling_period)
                if toplevel_post_blank is not None
                else None
            ),
            original_prev_blank=toplevel_prev_blank,
            original_post_blank=toplevel_post_blank,
            repeats=None,
            sub_sequences=[
                CapSampledSubSequence(
                    capture_slots=[
                        CaptureSlots(
                            duration=round(duration / sampling_period),
                            post_blank=round(blank / sampling_period),
                            original_duration=duration_original,
                            original_post_blank=blank_original,
                        )
                        for blank, blank_original, duration, duration_original in zip(
                            blanks[1:],
                            blanks_original[1:],
                            durations,
                            durations_original,
                        )
                    ],
                    prev_blank=round(blanks[0] / sampling_period),
                    post_blank=(
                        round(subseq.post_blank / sampling_period)
                        if subseq.post_blank is not None
                        else None
                    ),
                    original_prev_blank=blanks_original[0],
                    original_post_blank=(
                        subseq.post_blank if subseq.post_blank is not None else None
                    ),
                    repeats=subseq.repeats,
                )
                for edgeid, subseq, blanks, durations, blanks_original, durations_original in [
                    [
                        _,
                        subseqs[_],
                        _blanks[_],
                        _durations[_],
                        _blanks_original[_],
                        _durations_original[_],
                    ]
                    for _ in subseq_edges
                ]
            ],
        )

    def convert_to_sampled_sequence(
        self,
    ) -> tuple[dict[str, GenSampledSequence], dict[str, CapSampledSequence]]:
        # 念の為 sequence 内の各要素を配置
        self._tree.place_slots()
        # 中間形式に変換
        return self._create_sampled_sequence()


class SubSequenceBranch(Branch):
    def __init__(
        self,
        fixed_duration: Optional[float] = None,
        repeats: int = 1,
    ) -> None:
        super().__init__()
        self.repeats = repeats
        self._fixed_duration = fixed_duration
        self._total_duration_contents: Optional[float] = None

    @property
    def repeats(self) -> int:
        return self._repeats

    @repeats.setter
    def repeats(self, repeats: int) -> None:
        if not isinstance(repeats, int):
            raise ValueError("repeats must be int")
        self._repeats = repeats

    @property
    def fixed_duration(self) -> Optional[float]:
        return self._fixed_duration

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(duration={self.duration}, begin={self.begin}, next_node={self._next_node}, root_node={self._root_node}, post_blank={self.post_blank}, repeats={self.repeats})"

    def place(self, tree: SequenceTree) -> None:
        # 最大長を計算する
        for _ in tree.breadth_first_search(self._root_node)[1:]:
            tree._tree._cost[_] = tree._nodes_items[_].duration
        max_duration = max([_ for _ in tree._tree.evaluate(self._root_node).values()])
        self._total_duration_contents = max_duration
        # branch の duration は最大長に揃えると同時に cost も確定する
        # SubSequence では全体長を指定することもできてその場合は指定値を優先する
        if self._fixed_duration is None:
            self.duration = max_duration
        else:
            if max_duration <= self._fixed_duration:
                self.duration = self._fixed_duration
            else:
                raise ValueError(
                    f"Fixed duration {self._fixed_duration} is too smaller than total duration {max_duration}."
                )
        if self._next_node is None:
            raise ValueError("_next_node is None")
        tree._tree._cost[self._next_node] = self.duration

    @property
    def post_blank(self) -> Optional[float]:
        if self.duration is None:
            raise ValueError("duration is None")
        if self._total_duration_contents is None:
            raise ValueError("place slot first")
        return self.duration - self._total_duration_contents


class SubSequence(DequeWithContext):
    def __enter__(self) -> SubSequence:
        super().__enter__()
        return self

    def __init__(self, duration: Optional[float] = None, repeats: int = 1) -> None:
        if not isinstance(repeats, int):
            raise ValueError("repeats must be int")
        self._repeats = repeats
        self._fixed_duration = duration

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
        tree.branch(
            SubSequenceBranch(
                fixed_duration=self._fixed_duration,
                repeats=self.repeats,
            )
        )
        SubSequence.create_tree(tree, self)
        # with 内の定義の所定の位置にツリーを追加
        _rc.contexts[-1].append(tree)

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
    def __enter__(self) -> Series:
        super().__enter__()
        return self

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
        _rc.contexts[-1].append(tree)
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
    def __enter__(self) -> Flushleft:
        super().__enter__()
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        # このブランチ用のサブツリーを作る
        tree = SequenceTree()
        _rc.contexts[-1].append(tree)  # with 内の定義の所定の位置にツリーを追加
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
    def __enter__(self) -> Flushright:
        super().__enter__()
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        super().__exit__(exception_type, exception_value, traceback)
        # このブランチ用のサブツリーを作る
        tree = SequenceTree()
        _rc.contexts[-1].append(tree)  # with 内の定義の所定の位置にツリーを追加
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


class Utils:
    @classmethod
    def align_items(
        cls,
        items: MutableSequence[Item],
        sampling_period: float = DEFAULT_SAMPLING_PERIOD,
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
        ranges: MutableSequence[Waveform],
        frame: SubSequenceBranch,
    ) -> tuple[MutableSequence[float], MutableSequence[Optional[float]]]:
        if [x.begin is None for x in ranges]:
            raise ValueError("begin is None")
        slots = sorted(ranges, key=lambda x: x.begin if x.begin is not None else 0)
        _slots = Utils.align_items([_ for _ in slots if isinstance(_, Item)])
        _durations = [_.duration for _ in _slots if isinstance(_.duration, float)]
        _blanks = [
            (
                post.begin - prev.end
                if post.begin is not None and prev.end is not None
                else None
            )
            for prev, post in zip(_slots[:-1], _slots[1:])
        ] + [
            (
                frame.end - _slots[-1].end
                if frame.end is not None and _slots[-1].end is not None
                else None
            )
        ]
        return _durations, _blanks


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


def padding(duration: float) -> None:
    """
    Add padding with the specified duration to the sequence.

    Parameters
    ----------
    duration : float
        Duration of the padding in ns.
    """
    if len(_rc.contexts):
        Blank(duration=duration).target()


class Slot(Item):
    """
    Slot class for the sequence.

    Parameters
    ----------
    duration : float, optional
        Duration of the slot in ns. Default is None.

    Attributes
    ----------
    duration : float
        Duration of the slot in ns.
    begin : float
        Begin time of the slot in ns.
    end : float
        End time of the slot in ns.
    targets : tuple[str]
        Target qubits.
    """

    def __init__(self, duration: Optional[float] = None) -> None:
        super().__init__(duration)

    def target(self, *targets: str) -> None:
        """
        Set the target qubits of the slot.
        """
        self.targets = targets

        # Add the slot to the context
        if len(_rc.contexts):
            _rc.contexts[-1].append(deepcopy(self))


class Blank(Slot):
    pass


class Capture(Slot):
    pass


class Modifier(Slot):
    """begin <= t の時に cmag * func(t) を返す。未定義の場合，ステップ関数として動作。"""

    def __init__(self) -> None:
        super().__init__(duration=0)
        self.cmag = 1 + 0j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(begin={self.begin})"

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        """Branch object cannot set duration value"""
        raise ValueError("Branch object cannot set duration value")

    def func(self, t: float) -> complex:
        """時間依存の Modifier (例えば frequency) を書くときにここを定義する。通常の時間非依存では 1 + 0j を返す。"""
        return 1 + 0j

    def _func(self, t: float) -> complex:
        """グローバル時間軸 (begin <= t) の時に複素振幅 (self.cmag) を，それ以前は 1 + 0j を返す。"""
        if self.begin is None or self.duration is None:
            raise ValueError(
                "Either or both 'begin' and 'duration' are not initialized."
            )
        if self.begin <= t:
            return self.cmag * self.func(t)
        else:
            return 1 + 0j
        # return self.cmag * self.func(t)

    def ufunc(self, t: NDArray) -> NDArray:
        return np.frompyfunc(self._func, 1, 1)(t).astype(complex)


class VirtualZ(Modifier):
    """
    Modify the phase of the waveform.

    Parameters
    ----------
    theta : float, optional
        Phase angle in radian. Default is 0.0. Theta is defined as the rotation angle around the z-axis following the right-handed rule.
    """

    def __init__(self, theta: float = 0.0):
        super().__init__()
        self.cmag = np.exp(-1j * theta)  # theta は z 軸方向に右ネジの回転方向を正とする


class Magnifier(Modifier):
    """
    Modify the magnitude of the waveform.

    Parameters
    ----------
    magnitude : float, optional
        Magnitude of the waveform. Default is 1.0.
    """

    def __init__(self, magnitude: float = 1.0):
        super().__init__()
        self.cmag = magnitude * (1 + 0j)


class Frequency(Modifier):
    """
    Modify the frequency of the waveform.

    Parameters
    ----------
    modulation_frequency : float, optional
        Modulation frequency in GHz. Default is 0.0.
    """

    def __init__(self, modulation_frequency: float = 0.0):
        super().__init__()
        self.modulation_frequency = modulation_frequency

    def func(self, t: float) -> complex:
        return np.exp(2j * np.pi * self.modulation_frequency * t)


class Waveform(Slot):
    """
    Waveform class for the sequence.

    Parameters
    ----------
    duration : float, optional
        Duration of the waveform in ns. Default is None.

    Attributes
    ----------
    duration : float
        Duration of the waveform in ns.
    begin : float
        Begin time of the waveform in ns.
    end : float
        End time of the waveform in ns.
    targets : tuple[str]
        Target qubits.
    cmag : complex
        Complex magnitude of the waveform.
    """

    def __init__(
        self,
        duration: Optional[float] = None,
    ) -> None:
        super().__init__(duration=duration)
        self._iq: Optional[NDArray] = None
        self.cmag = 1 + 0j

    def func(self, t: float) -> complex:
        """正規化複素振幅 (1 + j0), ローカル時間軸 (begin=0) で iq 波形を返す．継承する時はここに関数を定義する．"""
        raise NotImplementedError()

    def _func(self, t: float) -> complex:
        """func() に対して複素振幅 (self.cmag) を適用，グローバル時間軸 (t) で iq 波形を返す"""
        if self.begin is None or self.duration is None:
            raise ValueError(
                "Either or both 'begin' and 'duration' are not initialized."
            )
        if t < self.begin or self.begin + self.duration < t:
            return 0 + 0j
        return self.cmag * self.func(t - self.begin)

    def ufunc(self, t: NDArray) -> NDArray:
        return np.frompyfunc(self._func, 1, 1)(t).astype(complex)

    def scaled(self, scale: float) -> "Waveform":
        """Returns a copy of the waveform scaled by the given factor."""
        new_waveform = deepcopy(self)
        new_waveform.cmag *= scale
        return new_waveform

    def shifted(self, phase: float) -> "Waveform":
        """Returns a copy of the waveform shifted by the given phase."""
        new_waveform = deepcopy(self)
        new_waveform.cmag *= np.exp(1j * phase)
        return new_waveform


class RaisedCosFlatTop(Waveform):
    def __init__(
        self,
        duration: Optional[float] = None,
        amplitude: float = 1.0,
        rise_time: float = 0.0,
    ):
        super().__init__(duration=duration)
        self.amplitude = amplitude
        self.rise_time = rise_time

    def func(self, t: float) -> complex:
        if self.duration is None:
            raise ValueError("duration is None")

        flattop_duration = self.duration - self.rise_time * 2

        if flattop_duration < 0:
            raise ValueError("duration is too short for rise_time")

        t1 = 0
        t2 = t1 + self.rise_time  # 立ち上がり完了時刻
        t3 = t2 + flattop_duration  # 立ち下がり開始時刻
        t4 = t3 + self.rise_time  # 立ち下がり完了時刻

        if (t1 <= t) & (t < t2):  # 立ち上がり時間領域の条件ブール値
            # 立ち上がり時間領域の値
            return (
                self.amplitude * (1.0 - np.cos(np.pi * (t - t1) / self.rise_time)) / 2.0
            )
        if (t2 <= t) & (t < t3):  # 一定値領域の条件ブール値
            # 一定値領域の値
            return self.amplitude
        if (t3 <= t) & (t < t4):  # 立ち下がり時間領域の条件ブール値
            # 立ち下がり時間領域の値
            return (
                self.amplitude * (1.0 - np.cos(np.pi * (t4 - t) / self.rise_time)) / 2.0
            )
        return 0.0 + 0.0j


class Rectangle(Waveform):
    def __init__(
        self,
        duration: Optional[float] = None,
        amplitude: float = 1.0,
    ):
        super().__init__(duration)
        self.amplitude = amplitude

    def func(self, t: float) -> complex:
        if self.duration is None:
            raise ValueError("duration is None")

        if 0 <= t and t < self.duration:
            return complex(self.amplitude)
        return 0 + 0j


class Arbit(Waveform):
    """
    Arbit class for the sequence.

    Parameters
    ----------
    iq : list | NDArray
        IQ data of the waveform.
    """

    def __init__(self, iq: list | NDArray):
        duration = len(iq) * DEFAULT_SAMPLING_PERIOD
        super().__init__(duration)
        self._iq = np.array(iq).astype(complex)

    def func(self, t: float) -> complex:
        """iq データを格納している numpy array に従って iq(t) の値を返す"""
        # ローカル時間軸を返すのに注意
        if self._iq is None:
            raise ValueError("_iq is None")
        if self.begin is None or self.duration is None:
            raise ValueError("begin or duration is None")

        D, dt = self.duration, DEFAULT_SAMPLING_PERIOD
        if 0 <= t < D:
            idx = math.floor(t / dt)
            return self._iq[idx]
        else:
            return 0 + 0j

    @property
    def iq(self) -> NDArray:
        """iq データを格納している numpy array への参照を返す"""
        if self.duration is None:
            raise ValueError("duration is None")
        T, dt = self.duration, DEFAULT_SAMPLING_PERIOD
        # N = round(T // dt)
        N = math.ceil(T / dt)
        # 初回アクセス or 前回アクセスから duration が更新されていれば ndarray を 0 + j0 で再生成
        if self._iq is None or N != self._iq.shape[0]:
            self._iq = np.zeros(N).astype(complex)  # iq data

        return self._iq


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
    ) -> NDArray[np.float64]:
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
        cls,
        sampling_timing: NDArray[np.float64],
        slots: MutableSequence[Waveform | Modifier],
    ) -> NDArray[np.complex128]:
        """slots を sampling_timing でサンプリングして返す。"""
        tstart = sampling_timing[0]
        DT = sampling_timing[1] - sampling_timing[0]
        # サンプリング値を格納する配列を初期化
        np_waveform = np.zeros(sampling_timing.size).astype(complex)
        # Waveform のみを抽出
        waveforms = [o for o in slots if isinstance(o, Waveform)]
        # 各Waveform をサンプリングして適切な位置に加算
        for w in waveforms:
            if w.begin is None or w.duration is None:
                raise ValueError(f"begin or duration of {w.__class__.__name__} is None")
            B, E = math.ceil((w.begin - tstart) / DT), math.ceil((w.end - tstart) / DT)
            v = w.ufunc(sampling_timing[B:E])
            np_waveform[B:E] += v
        # Modifier 値を格納する配列を初期化
        np_modifier = np.ones(sampling_timing.size).astype(complex)
        # Modifier のみを抽出
        modifiers = [o for o in slots if isinstance(o, Modifier)]
        for m in modifiers:
            if m.begin is None:
                raise ValueError(f"begin of {m.__class__.__name__} is None")
            B = math.ceil((m.begin - tstart) / DT)
            np_modifier[B:] *= m.ufunc(sampling_timing[B:])
        # Modifier を Waveform に適用したものを返す
        return np_waveform * np_modifier

    def __init__(
        self,
        branch: SubSequenceBranch,
        waveforms: MutableSequence[Waveform | Modifier],
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
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.float64],
        Optional[NDArray[np.float64]],
    ]:
        begin = self._branch.begin
        duration = self._branch._total_duration_contents
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


@dataclass
class SampledSequenceBase:
    target_name: str
    prev_blank: int = 0  # words
    sampling_period: float = DEFAULT_SAMPLING_PERIOD
    post_blank: Optional[int] = None  # words
    repeats: Optional[int] = None
    original_prev_blank: Optional[float] = None  # ns
    original_post_blank: Optional[float] = None  # ns
    # これは本来外に出すべき
    padding: int = 0  # Sa
    modulation_frequency: Optional[float] = None  # GHz

    def asdict(self) -> dict:
        return asdict(self)


@dataclass
class GenSampledSequence(SampledSequenceBase):
    sub_sequences: MutableSequence[GenSampledSubSequence] = field(default_factory=list)
    # これは本来外に出すべき
    readout_timings: Optional[MutableSequence[list[tuple[float, float]]]] = None  # ns

    def asdict(self) -> dict:
        return super().asdict() | {
            "sub_sequences": [_.asdict() for _ in self.sub_sequences],
            "readout_timings": None,
            "class": self.__class__.__name__,
        }


@dataclass
class GenSampledSubSequence:
    real: NDArray[np.float64]
    imag: NDArray[np.float64]
    repeats: int
    post_blank: Optional[int] = None  # samples
    original_post_blank: Optional[float] = None  # ns

    def asdict(self) -> dict:
        return {
            "real": self.real.tolist(),
            "imag": self.imag.tolist(),
            "repeats": self.repeats,
            "post_blank": self.post_blank,
            "original_post_blank": self.original_post_blank,
        }


@dataclass
class CapSampledSequence(SampledSequenceBase):
    sub_sequences: MutableSequence[CapSampledSubSequence] = field(default_factory=list)
    # これは本来外に出すべき
    readin_offsets: Optional[MutableSequence[list[tuple[float, float]]]] = None  # ns

    def asdict(self) -> dict:
        return super().asdict() | {
            "sub_sequences": [_.asdict() for _ in self.sub_sequences],
            "readin_offsets": None,
            "class": self.__class__.__name__,
        }


@dataclass
class CapSampledSubSequence:
    capture_slots: MutableSequence[CaptureSlots]
    # duration: int  # samples
    prev_blank: int  # samples
    post_blank: Optional[int]  # samples
    original_prev_blank: float  # ns
    original_post_blank: Optional[float]  # ns
    repeats: Optional[int]

    def asdict(self) -> dict:
        return asdict(self)


@dataclass
class CaptureSlots:
    duration: int  # samples
    post_blank: Optional[int]  # samples
    original_duration: float  # ns
    original_post_blank: Optional[float]  # ns

    def asdict(self) -> dict:
        return {}
