from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

from pydantic import BaseModel, Field

from fzed_shunting.domain.master_data import MasterData


class PathValidationResult(BaseModel):
    is_valid: bool
    branch_codes: list[str] = Field(default_factory=list)
    total_length_m: float | None = None
    uses_l1: bool | None = None
    reverse_branch_codes: list[str] = Field(default_factory=list)
    required_reverse_clearance_m: float | None = None
    blocking_tracks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class TrackRouteMeta:
    connection_nodes: tuple[str, ...]
    terminal_branch: str | None = None
    reverse_branches: tuple[str, ...] = ()


BRANCH_ENDPOINTS: dict[str, tuple[str, str]] = {
    "大门L1": ("大门", "L1"),
    "L1-L2": ("L1", "L2"),
    "L1-L3": ("L1", "L3"),
    "L3-L4": ("L3", "L4"),
    "L2-L12": ("L2", "L12"),
    "L2-Z4": ("L2", "Z4"),
    "L4-Z4": ("L4", "Z4"),
    "L4-Z3": ("L4", "Z3"),
    "L3-L5": ("L3", "L5"),
    "L5-L6": ("L5", "L6"),
    "L5-Z2": ("L5", "Z2"),
    "L6-Z1": ("L6", "Z1"),
    "Z1-Z2": ("Z1", "Z2"),
    "Z2-Z3": ("Z2", "Z3"),
    "L6-L7": ("L6", "L7"),
    "L7-机库尽头": ("L7", "机库尽头"),
    "L7-调梁尽头": ("L7", "调梁尽头"),
    "Z1-L8": ("Z1", "L8"),
    "Z3-L13": ("Z3", "L13"),
    "Z4-L12": ("Z4", "L12"),
    "L8-L9": ("L8", "L9"),
    "L8-L11(L14)": ("L8", "L14"),
    "L12-L13": ("L12", "L13"),
    "L13-L14": ("L13", "L14"),
    "L14-L15": ("L14", "L15"),
    "L9-油漆尽头": ("L9", "油漆尽头"),
    "L9-洗罐尽头": ("L9", "洗罐尽头"),
    "L15-抛丸尽头": ("L15", "抛丸尽头"),
    "L15-L16": ("L15", "L16"),
    "L16-L17": ("L16", "L17"),
    "L16-L19": ("L16", "L19"),
    "L17-L18": ("L17", "L18"),
    "L18-修4尽头": ("L18", "修4尽头"),
    "L18-修3尽头": ("L18", "修3尽头"),
    "L17-修2尽头": ("L17", "修2尽头"),
    "L19-修1尽头": ("L19", "修1尽头"),
    "L19-卸轮尽头": ("L19", "卸轮尽头"),
}

TRACK_ENDPOINTS: dict[str, tuple[str, str]] = {
    "联6": ("279", "L1"),
    "联7": ("L15", "L16"),
    "渡1": ("L1", "L2"),
    "渡2": ("L1", "L3"),
    "渡3": ("L3", "L4"),
    "渡4": ("L6", "L7"),
    "渡5": ("L6", "Z1"),
    "渡6": ("Z1", "Z2"),
    "渡7": ("Z2", "Z3"),
    "渡8": ("L12", "L13"),
    "渡9": ("L13", "L14"),
    "渡10": ("L14", "L15"),
    "渡11": ("L16", "L19"),
    "渡12": ("L16", "L17"),
    "渡13": ("L17", "L18"),
    "临1": ("L3", "L5"),
    "临2": ("L5", "L6"),
    "临3": ("L8", "L9"),
    "临4": ("L8", "L14"),
    "存1": ("L5", "Z2"),
    "存2": ("L4", "Z3"),
    "存3": ("L4", "Z4"),
    "存4北": ("L2", "Z4"),
    "存4南": ("Z4", "L12"),
    "存5北": ("L2", "存5中"),
    "存5南": ("存5中", "L12"),
    "机库": ("L7", "机库尽头"),
    "调北": ("L7", "调棚北口"),
    "调棚": ("调棚北口", "调棚尽头"),
    "机北": ("Z1", "机棚北口"),
    "机棚": ("机棚北口", "L8"),
    "预修": ("Z3", "L13"),
    "洗北": ("L9", "洗南北口"),
    "洗南": ("洗南北口", "洗南尽头"),
    "抛": ("L15", "抛尽头"),
    "油": ("L9", "油尽头"),
    "轮": ("L19", "轮尽头"),
    "修1库外": ("L19", "修1门"),
    "修1库内": ("修1门", "修1尽头"),
    "修2库外": ("L17", "修2门"),
    "修2库内": ("修2门", "修2尽头"),
    "修3库外": ("L18", "修3门"),
    "修3库内": ("修3门", "修3尽头"),
    "修4库外": ("L18", "修4门"),
    "修4库内": ("修4门", "修4尽头"),
}


TRACK_ROUTE_META: dict[str, TrackRouteMeta] = {
    "联6": TrackRouteMeta(("L1",)),
    "联7": TrackRouteMeta(("L15", "L16")),
    "渡1": TrackRouteMeta(("L1", "L2")),
    "渡2": TrackRouteMeta(("L1", "L3")),
    "渡3": TrackRouteMeta(("L3", "L4")),
    "渡4": TrackRouteMeta(("L6", "L7")),
    "渡5": TrackRouteMeta(("L6", "Z1")),
    "渡6": TrackRouteMeta(("Z1", "Z2")),
    "渡7": TrackRouteMeta(("Z2", "Z3")),
    "渡8": TrackRouteMeta(("L12", "L13")),
    "渡9": TrackRouteMeta(("L13", "L14")),
    "渡10": TrackRouteMeta(("L14", "L15")),
    "渡11": TrackRouteMeta(("L16", "L19")),
    "渡12": TrackRouteMeta(("L16", "L17")),
    "渡13": TrackRouteMeta(("L17", "L18")),
    "临1": TrackRouteMeta(("L3", "L5"), terminal_branch="L3-L5"),
    "临2": TrackRouteMeta(("L5", "L6"), terminal_branch="L5-L6"),
    "临3": TrackRouteMeta(("L8", "L9"), terminal_branch="L8-L9"),
    "临4": TrackRouteMeta(("L8", "L14"), terminal_branch="L8-L11(L14)"),
    "存1": TrackRouteMeta(("L5", "Z2"), terminal_branch="L5-Z2"),
    "存2": TrackRouteMeta(("L4", "Z3"), terminal_branch="L4-Z3"),
    "存3": TrackRouteMeta(("L4", "Z4"), terminal_branch="L4-Z4"),
    "存4北": TrackRouteMeta(("L2", "Z4"), terminal_branch="L2-Z4"),
    "存4南": TrackRouteMeta(("Z4", "L12"), terminal_branch="Z4-L12"),
    "存5北": TrackRouteMeta(("L2",)),
    "存5南": TrackRouteMeta(("L12",)),
    "机库": TrackRouteMeta(("L7",), terminal_branch="L7-机库尽头"),
    "调北": TrackRouteMeta(("L7",)),
    "调棚": TrackRouteMeta(("L7",), terminal_branch="L7-调梁尽头"),
    "机北": TrackRouteMeta(("Z1",)),
    "机棚": TrackRouteMeta(("Z1", "L8"), terminal_branch="Z1-L8"),
    "预修": TrackRouteMeta(("Z3", "L13"), terminal_branch="Z3-L13"),
    "洗北": TrackRouteMeta(("L9",)),
    "洗南": TrackRouteMeta(("L9",), terminal_branch="L9-洗罐尽头"),
    "抛": TrackRouteMeta(("L15",), terminal_branch="L15-抛丸尽头"),
    "油": TrackRouteMeta(("L9",), terminal_branch="L9-油漆尽头"),
    "轮": TrackRouteMeta(("L19",), terminal_branch="L19-卸轮尽头"),
    "修1库外": TrackRouteMeta(("L19",)),
    "修2库外": TrackRouteMeta(("L17",)),
    "修3库外": TrackRouteMeta(("L18",)),
    "修4库外": TrackRouteMeta(("L18",)),
    "修1库内": TrackRouteMeta(("L19",), terminal_branch="L19-修1尽头"),
    "修2库内": TrackRouteMeta(("L17",), terminal_branch="L17-修2尽头"),
    "修3库内": TrackRouteMeta(("L18",), terminal_branch="L18-修3尽头"),
    "修4库内": TrackRouteMeta(("L18",), terminal_branch="L18-修4尽头"),
}


@dataclass(frozen=True)
class ResolvedRoute:
    branch_codes: list[str]
    total_length_m: float
    uses_l1: bool


class RouteOracle:
    def __init__(self, master: MasterData):
        self.master = master
        self._branch_endpoints = self._build_branch_endpoints()
        self._track_endpoints = self._build_track_endpoints()
        self._track_route_meta = self._build_track_route_meta()
        self._graph = self._build_graph()
        self._track_graph = self._build_track_graph()
        self._route_cache: dict[tuple[str, str], ResolvedRoute | None] = {}
        self._path_track_cache: dict[tuple[str, str], tuple[str, ...] | None] = {}
        self._clear_path_track_cache: dict[tuple, tuple[str, ...] | None] = {}
        self._endpoint_path_track_cache: dict[tuple, tuple[str, ...] | None] = {}
        self._node_path_cache: dict[tuple[str, str], tuple[str, ...] | None] = {}
        self._route_blockage_plan_cache: dict[tuple, object] = {}

    def validate_path(
        self,
        source_track: str,
        target_track: str,
        path_tracks: list[str],
        train_length_m: float,
        occupied_track_sequences: dict[str, list[str]] | None = None,
        expected_path_tracks: list[str] | None = None,
        route: ResolvedRoute | None = None,
        source_node: str | None = None,
        target_node: str | None = None,
    ) -> PathValidationResult:
        errors: list[str] = []
        blocking_tracks: list[str] = []
        if not path_tracks:
            errors.append("pathTracks cannot be empty")
            return PathValidationResult(is_valid=False, errors=errors)
        if path_tracks[0] != source_track or path_tracks[-1] != target_track:
            errors.append("pathTracks must start at source and end at target")
            return PathValidationResult(is_valid=False, errors=errors)
        if source_track == target_track:
            return PathValidationResult(is_valid=True)

        if expected_path_tracks is None:
            expected_path_tracks = self.resolve_clear_path_tracks(
                source_track,
                target_track,
                occupied_track_sequences=occupied_track_sequences,
                source_node=source_node,
                target_node=target_node,
            )
        if expected_path_tracks is None:
            errors.append(f"No clear track path mapping for {source_track} -> {target_track}")
            return PathValidationResult(is_valid=False, errors=errors)
        if path_tracks != expected_path_tracks:
            errors.append(
                f"pathTracks must match complete route path: {expected_path_tracks}"
            )

        if route is None:
            route = self.resolve_route_for_path_tracks(
                path_tracks,
                source_node=source_node,
                target_node=target_node,
            )
        if route is None:
            errors.append(f"No route mapping for {source_track} -> {target_track}")
            return PathValidationResult(is_valid=False, errors=errors)

        if (
            occupied_track_sequences is not None
            and self.master.business_rules.require_clear_intermediate_path_tracks
        ):
            blocking_tracks = self._blocking_tracks_for_path(
                path_tracks,
                occupied_track_sequences=occupied_track_sequences,
                source_node=source_node,
                target_node=target_node,
            )
            for track_code in blocking_tracks:
                errors.append(
                    f"Route interference on intermediate track {track_code}: "
                    f"{occupied_track_sequences.get(track_code, [])}"
                )

        for branch in route.branch_codes:
            physical_route = self.master.physical_routes.get(branch)
            if physical_route is None:
                errors.append(f"Missing physical route branch: {branch}")
                continue
            if physical_route.status != "已确认":
                errors.append(f"Route branch {branch} 状态为{physical_route.status}")

        if route.uses_l1 and train_length_m > 190:
            errors.append("Train length exceeds 190m limit when passing L1 branch")

        reverse_branch_codes: list[str] = []
        required_reverse_clearance_m = train_length_m + self.master.business_rules.loco_length_m
        reverse_branch_codes = self._resolve_reverse_branch_codes(
            path_tracks=path_tracks,
            route=route,
        )
        for reverse_branch in reverse_branch_codes:
            physical_route = self.master.physical_routes.get(reverse_branch)
            if physical_route is None:
                continue
            if physical_route.total_length_m + 1e-9 < required_reverse_clearance_m:
                errors.append(
                    f"Reverse clearance insufficient on branch {reverse_branch}: "
                    f"need {required_reverse_clearance_m:.1f}m, got {physical_route.total_length_m:.1f}m"
                )

        return PathValidationResult(
            is_valid=not errors,
            branch_codes=route.branch_codes,
            total_length_m=route.total_length_m,
            uses_l1=route.uses_l1,
            reverse_branch_codes=reverse_branch_codes,
            required_reverse_clearance_m=required_reverse_clearance_m,
            blocking_tracks=blocking_tracks if occupied_track_sequences is not None else [],
            errors=errors,
        )

    def validate_loco_access(
        self,
        *,
        loco_track: str,
        target_track: str,
        occupied_track_sequences: dict[str, list[str]] | None = None,
        carried_train_length_m: float = 0.0,
        loco_node: str | None = None,
    ) -> PathValidationResult:
        """Validate whether the locomotive can reach a hook endpoint.

        ATTACH itself is represented as a native same-track hook, but the
        locomotive still must be able to reach that track through clear portal
        tracks. If it is already carrying vehicles, the access move must also
        satisfy the same length/reverse-clearance constraints as a DETACH move.
        The endpoint may contain the vehicles being attached, so only
        intermediate tracks can block access.
        """
        source_node = loco_node
        target_node = (
            self._order_end_node(target_track)
            if occupied_track_sequences is not None
            and occupied_track_sequences.get(target_track)
            else None
        )
        path_tracks = self.resolve_clear_path_tracks(
            loco_track,
            target_track,
            occupied_track_sequences=occupied_track_sequences,
            source_node=source_node,
            target_node=target_node,
        )
        if path_tracks is None:
            blocking_tracks: list[str] = []
            default_path_tracks = self.resolve_path_tracks_for_endpoint_constraints(
                loco_track,
                target_track,
                occupied_track_sequences=occupied_track_sequences,
                source_node=source_node,
                target_node=target_node,
            )
            if default_path_tracks is None:
                default_path_tracks = self.resolve_path_tracks(loco_track, target_track)
            if (
                default_path_tracks is not None
                and occupied_track_sequences is not None
                and self.master.business_rules.require_clear_intermediate_path_tracks
            ):
                blocking_tracks = self._blocking_tracks_for_path(
                    default_path_tracks,
                    occupied_track_sequences=occupied_track_sequences,
                    source_node=source_node,
                    target_node=target_node,
                )
            errors = [
                f"No clear locomotive access path mapping for {loco_track} -> {target_track}"
            ]
            errors.extend(
                f"Route interference on intermediate track {track_code}: "
                f"{occupied_track_sequences.get(track_code, [])}"
                for track_code in blocking_tracks
            )
            return PathValidationResult(
                is_valid=False,
                blocking_tracks=blocking_tracks,
                errors=errors,
            )
        route = self.resolve_route_for_path_tracks(
            path_tracks,
            source_node=source_node,
            target_node=target_node,
        )
        if route is None and loco_track != target_track:
            return PathValidationResult(
                is_valid=False,
                errors=[f"No locomotive access route mapping for {loco_track} -> {target_track}"],
            )
        return self.validate_path(
            source_track=loco_track,
            target_track=target_track,
            path_tracks=path_tracks,
            train_length_m=carried_train_length_m,
            occupied_track_sequences=occupied_track_sequences,
            expected_path_tracks=path_tracks,
            route=route,
            source_node=source_node,
            target_node=target_node,
        )

    def resolve_route_for_path_tracks(
        self,
        path_tracks: list[str],
        *,
        source_node: str | None = None,
        target_node: str | None = None,
    ) -> ResolvedRoute | None:
        if not path_tracks:
            return None
        if len(path_tracks) == 1:
            return ResolvedRoute(branch_codes=[], total_length_m=0.0, uses_l1=False)
        route = self._resolve_route_for_concrete_track_path(
            path_tracks,
            source_node=source_node,
            target_node=target_node,
        )
        if route is not None:
            return route
        return self.resolve_route(path_tracks[0], path_tracks[-1])

    def resolve_clear_path_tracks(
        self,
        source_track: str,
        target_track: str,
        *,
        occupied_track_sequences: dict[str, list[str]] | None,
        source_node: str | None = None,
        target_node: str | None = None,
    ) -> list[str] | None:
        if (
            occupied_track_sequences is None
            or not self.master.business_rules.require_clear_intermediate_path_tracks
        ):
            return self.resolve_path_tracks(source_track, target_track)
        if source_track == target_track:
            return [source_track]
        if source_track not in self._track_endpoints or target_track not in self._track_endpoints:
            return None
        cache_key = (
            source_track,
            target_track,
            self._occupied_track_key(occupied_track_sequences),
            source_node,
            target_node,
        )
        if cache_key in self._clear_path_track_cache:
            cached = self._clear_path_track_cache[cache_key]
            return list(cached) if cached is not None else None

        queue: list[tuple[float, str, list[str]]] = [(0.0, source_track, [source_track])]
        best: dict[str, float] = {source_track: 0.0}
        while queue:
            cost, track_code, path_tracks = heappop(queue)
            if track_code == target_track:
                self._clear_path_track_cache[cache_key] = tuple(path_tracks)
                return path_tracks
            if cost > best.get(track_code, float("inf")) + 1e-9:
                continue
            for next_track in self._track_graph.get(track_code, []):
                next_track_info = self.master.tracks.get(next_track)
                if next_track_info is None:
                    continue
                candidate_path = path_tracks + [next_track]
                if self._blocking_tracks_for_path(
                    candidate_path,
                    occupied_track_sequences=occupied_track_sequences,
                    source_node=source_node,
                    target_node=target_node if next_track == target_track else None,
                ):
                    continue
                next_cost = cost + next_track_info.effective_length_m
                if next_cost >= best.get(next_track, float("inf")) - 1e-9:
                    continue
                best[next_track] = next_cost
                heappush(queue, (next_cost, next_track, candidate_path))
        self._clear_path_track_cache[cache_key] = None
        return None

    def resolve_path_tracks_for_endpoint_constraints(
        self,
        source_track: str,
        target_track: str,
        *,
        occupied_track_sequences: dict[str, list[str]] | None = None,
        source_node: str | None = None,
        target_node: str | None = None,
    ) -> list[str] | None:
        if source_track == target_track:
            return [source_track]
        if source_track not in self._track_endpoints or target_track not in self._track_endpoints:
            return None
        cache_key = (
            source_track,
            target_track,
            self._occupied_track_key(occupied_track_sequences),
            source_node,
            target_node,
        )
        if cache_key in self._endpoint_path_track_cache:
            cached = self._endpoint_path_track_cache[cache_key]
            return list(cached) if cached is not None else None
        queue: list[tuple[float, str, list[str]]] = [(0.0, source_track, [source_track])]
        best: dict[str, float] = {source_track: 0.0}
        while queue:
            cost, track_code, path_tracks = heappop(queue)
            if track_code == target_track:
                self._endpoint_path_track_cache[cache_key] = tuple(path_tracks)
                return path_tracks
            if cost > best.get(track_code, float("inf")) + 1e-9:
                continue
            for next_track in self._track_graph.get(track_code, []):
                next_track_info = self.master.tracks.get(next_track)
                if next_track_info is None:
                    continue
                candidate_path = path_tracks + [next_track]
                if self._path_endpoint_blockers(
                    candidate_path,
                    occupied_track_sequences=occupied_track_sequences,
                    source_node=source_node,
                    target_node=target_node if next_track == target_track else None,
                ):
                    continue
                next_cost = cost + next_track_info.effective_length_m
                if next_cost >= best.get(next_track, float("inf")) - 1e-9:
                    continue
                best[next_track] = next_cost
                heappush(queue, (next_cost, next_track, candidate_path))
        self._endpoint_path_track_cache[cache_key] = None
        return None

    def resolve_route(self, source_track: str, target_track: str) -> ResolvedRoute | None:
        cache_key = (source_track, target_track)
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]
        source_meta = self._track_route_meta.get(source_track)
        target_meta = self._track_route_meta.get(target_track)
        if source_meta is None or target_meta is None:
            self._route_cache[cache_key] = None
            return None

        best_route: ResolvedRoute | None = None
        for source_node in source_meta.connection_nodes:
            for target_node in target_meta.connection_nodes:
                middle = self._shortest_path(source_node, target_node)
                if middle is None and source_node != target_node:
                    continue
                branch_codes: list[str] = []
                total_length_m = 0.0
                total_length_m = self._append_branch(
                    branch_codes,
                    source_meta.terminal_branch,
                    total_length_m,
                )
                if middle is not None:
                    for branch in middle:
                        total_length_m = self._append_branch(branch_codes, branch, total_length_m)
                total_length_m = self._append_branch(
                    branch_codes,
                    target_meta.terminal_branch,
                    total_length_m,
                )
                uses_l1 = any(self._branch_uses_l1(branch) for branch in branch_codes)
                candidate = ResolvedRoute(
                    branch_codes=branch_codes,
                    total_length_m=total_length_m,
                    uses_l1=uses_l1,
                )
                if best_route is None or candidate.total_length_m < best_route.total_length_m:
                    best_route = candidate
        self._route_cache[cache_key] = best_route
        return best_route

    def _resolve_route_for_concrete_track_path(
        self,
        path_tracks: list[str],
        *,
        source_node: str | None = None,
        target_node: str | None = None,
    ) -> ResolvedRoute | None:
        if not path_tracks:
            return None
        if len(path_tracks) == 1:
            return ResolvedRoute(branch_codes=[], total_length_m=0.0, uses_l1=False)
        shared_nodes: list[str] = []
        for first_track, second_track in zip(path_tracks, path_tracks[1:]):
            shared_node = self._shared_endpoint(first_track, second_track)
            if shared_node is None:
                return None
            shared_nodes.append(shared_node)

        branch_codes: list[str] = []
        total_length_m = 0.0
        source_meta = self._track_route_meta.get(path_tracks[0])
        if source_meta is None:
            return None
        total_length_m = self._append_branch(
            branch_codes,
            source_meta.terminal_branch,
            total_length_m,
        )
        current_node = self._route_node_for_track_endpoint(
            path_tracks[0],
            source_node or shared_nodes[0],
        )
        if current_node is None:
            return None
        source_exit_node = self._route_node_for_track_endpoint(path_tracks[0], shared_nodes[0])
        if source_exit_node is None:
            return None
        total_length_m = self._append_node_path(
            branch_codes,
            current_node,
            source_exit_node,
            total_length_m,
        )
        current_node = source_exit_node

        for index in range(1, len(path_tracks) - 1):
            track_code = path_tracks[index]
            entry_node = self._route_node_for_track_endpoint(track_code, shared_nodes[index - 1])
            exit_node = self._route_node_for_track_endpoint(track_code, shared_nodes[index])
            if entry_node is None or exit_node is None:
                return None
            total_length_m = self._append_node_path(
                branch_codes,
                current_node,
                entry_node,
                total_length_m,
            )
            total_length_m = self._append_node_path(
                branch_codes,
                entry_node,
                exit_node,
                total_length_m,
            )
            current_node = exit_node

        target_track = path_tracks[-1]
        target_meta = self._track_route_meta.get(target_track)
        if target_meta is None:
            return None
        target_entry_node = self._route_node_for_track_endpoint(target_track, shared_nodes[-1])
        if target_entry_node is None:
            return None
        total_length_m = self._append_node_path(
            branch_codes,
            current_node,
            target_entry_node,
            total_length_m,
        )
        if target_node is not None:
            target_exit_node = self._route_node_for_track_endpoint(target_track, target_node)
            if target_exit_node is None:
                return None
            total_length_m = self._append_node_path(
                branch_codes,
                target_entry_node,
                target_exit_node,
                total_length_m,
            )
        total_length_m = self._append_branch(
            branch_codes,
            target_meta.terminal_branch,
            total_length_m,
        )
        uses_l1 = any(self._branch_uses_l1(branch) for branch in branch_codes)
        return ResolvedRoute(
            branch_codes=branch_codes,
            total_length_m=total_length_m,
            uses_l1=uses_l1,
        )

    def _route_node_for_track_endpoint(
        self,
        track_code: str,
        endpoint_node: str,
    ) -> str | None:
        track_meta = self._track_route_meta.get(track_code)
        if track_meta is None:
            return None
        if endpoint_node in track_meta.connection_nodes:
            return endpoint_node
        track_endpoints = self._track_endpoints.get(track_code, ())
        if endpoint_node in track_endpoints:
            endpoint_index = track_endpoints.index(endpoint_node)
            if endpoint_index < len(track_meta.connection_nodes):
                return track_meta.connection_nodes[endpoint_index]
        if len(track_meta.connection_nodes) == 1:
            return track_meta.connection_nodes[0]
        if track_meta.terminal_branch is not None:
            branch_endpoints = self._branch_endpoints.get(track_meta.terminal_branch)
            if branch_endpoints is not None:
                for branch_endpoint in branch_endpoints:
                    if branch_endpoint in track_meta.connection_nodes:
                        return branch_endpoint
        return None

    def _append_node_path(
        self,
        branch_codes: list[str],
        source_node: str,
        target_node: str,
        total_length_m: float,
    ) -> float:
        if source_node == target_node:
            return total_length_m
        middle = self._shortest_path(source_node, target_node)
        if middle is None:
            return total_length_m
        for branch in middle:
            total_length_m = self._append_branch(branch_codes, branch, total_length_m)
        return total_length_m

    def resolve_path_tracks(self, source_track: str, target_track: str) -> list[str] | None:
        cache_key = (source_track, target_track)
        if cache_key in self._path_track_cache:
            cached = self._path_track_cache[cache_key]
            return list(cached) if cached is not None else None
        if source_track == target_track:
            return [source_track]
        if source_track not in self._track_endpoints or target_track not in self._track_endpoints:
            self._path_track_cache[cache_key] = None
            return None

        queue: list[tuple[float, str, list[str]]] = [(0.0, source_track, [source_track])]
        best: dict[str, float] = {source_track: 0.0}
        while queue:
            cost, track_code, path_tracks = heappop(queue)
            if track_code == target_track:
                self._path_track_cache[cache_key] = tuple(path_tracks)
                return path_tracks
            if cost > best.get(track_code, float("inf")) + 1e-9:
                continue
            for next_track in self._track_graph.get(track_code, []):
                next_track_info = self.master.tracks.get(next_track)
                if next_track_info is None:
                    continue
                next_cost = cost + next_track_info.effective_length_m
                if next_cost >= best.get(next_track, float("inf")) - 1e-9:
                    continue
                best[next_track] = next_cost
                heappush(queue, (next_cost, next_track, path_tracks + [next_track]))
        self._path_track_cache[cache_key] = None
        return None

    def _blocking_tracks_for_path(
        self,
        path_tracks: list[str],
        *,
        occupied_track_sequences: dict[str, list[str]],
        source_node: str | None = None,
        target_node: str | None = None,
    ) -> list[str]:
        blockers: list[str] = []
        if len(path_tracks) <= 1:
            return blockers
        for track_code in path_tracks[1:-1]:
            if occupied_track_sequences.get(track_code):
                blockers.append(track_code)
        source_track = path_tracks[0]
        if source_node is not None and occupied_track_sequences.get(source_track):
            exit_node = self._shared_endpoint(source_track, path_tracks[1])
            if exit_node is None or exit_node != source_node:
                blockers.append(source_track)
        target_track = path_tracks[-1]
        if target_node is not None and occupied_track_sequences.get(target_track):
            entry_node = self._shared_endpoint(path_tracks[-2], target_track)
            if entry_node is None or entry_node != target_node:
                blockers.append(target_track)
        elif target_node is not None and self._target_endpoint_requires_direct_entry(
            target_track,
            target_node,
        ):
            entry_node = self._shared_endpoint(path_tracks[-2], target_track)
            if entry_node is None or entry_node != target_node:
                blockers.append(target_track)
        return list(dict.fromkeys(blockers))

    def _occupied_track_key(
        self,
        occupied_track_sequences: dict[str, list[str]] | None,
    ) -> tuple[str, ...]:
        if not occupied_track_sequences:
            return ()
        return tuple(
            sorted(
                track_code
                for track_code, seq in occupied_track_sequences.items()
                if seq
            )
        )

    def _path_endpoint_blockers(
        self,
        path_tracks: list[str],
        *,
        occupied_track_sequences: dict[str, list[str]] | None = None,
        source_node: str | None = None,
        target_node: str | None = None,
    ) -> list[str]:
        if len(path_tracks) <= 1:
            return []
        blockers: list[str] = []
        if (
            source_node is not None
            and occupied_track_sequences is not None
            and occupied_track_sequences.get(path_tracks[0])
        ):
            source_track = path_tracks[0]
            exit_node = self._shared_endpoint(source_track, path_tracks[1])
            if exit_node is None or exit_node != source_node:
                blockers.append(source_track)
        if (
            target_node is not None
            and occupied_track_sequences is not None
            and occupied_track_sequences.get(path_tracks[-1])
        ):
            target_track = path_tracks[-1]
            entry_node = self._shared_endpoint(path_tracks[-2], target_track)
            if entry_node is None or entry_node != target_node:
                blockers.append(target_track)
        elif target_node is not None:
            target_track = path_tracks[-1]
            if self._target_endpoint_requires_direct_entry(target_track, target_node):
                entry_node = self._shared_endpoint(path_tracks[-2], target_track)
                if entry_node is None or entry_node != target_node:
                    blockers.append(target_track)
        return blockers

    def _target_endpoint_requires_direct_entry(
        self,
        track_code: str,
        endpoint_node: str,
    ) -> bool:
        track_meta = self._track_route_meta.get(track_code)
        if track_meta is None:
            return True
        return endpoint_node not in track_meta.connection_nodes

    def order_end_node(self, track_code: str) -> str | None:
        return self._order_end_node(track_code)

    def track_has_node(self, track_code: str, node: str | None) -> bool:
        return node is not None and node in self._track_endpoints.get(track_code, ())

    def path_entry_node(self, path_tracks: list[str]) -> str | None:
        if len(path_tracks) < 2:
            return self._order_end_node(path_tracks[0]) if path_tracks else None
        return self._shared_endpoint(path_tracks[-2], path_tracks[-1])

    def _shared_endpoint(self, first_track: str, second_track: str) -> str | None:
        first = set(self._track_endpoints.get(first_track, ()))
        second = set(self._track_endpoints.get(second_track, ()))
        shared = sorted(first & second)
        return shared[0] if shared else None

    def _order_end_node(self, track_code: str) -> str | None:
        endpoints = self._track_endpoints.get(track_code)
        if not endpoints:
            return None
        return endpoints[0]

    def _build_graph(self) -> dict[str, list[tuple[str, str, float]]]:
        graph: dict[str, list[tuple[str, str, float]]] = {}
        for branch, (left, right) in self._branch_endpoints.items():
            route = self.master.physical_routes.get(branch)
            if route is None:
                continue
            graph.setdefault(left, []).append((right, branch, route.total_length_m))
            graph.setdefault(right, []).append((left, branch, route.total_length_m))
        return graph

    def _build_branch_endpoints(self) -> dict[str, tuple[str, str]]:
        endpoints: dict[str, tuple[str, str]] = {}
        for branch_code, route in self.master.physical_routes.items():
            if route.left_node and route.right_node:
                endpoints[branch_code] = (route.left_node, route.right_node)
            elif branch_code in BRANCH_ENDPOINTS:
                endpoints[branch_code] = BRANCH_ENDPOINTS[branch_code]
        return endpoints

    def _build_track_endpoints(self) -> dict[str, tuple[str, str]]:
        endpoints: dict[str, tuple[str, str]] = {}
        for track_code, track in self.master.tracks.items():
            if len(track.endpoint_nodes) == 2:
                endpoints[track_code] = (
                    track.endpoint_nodes[0],
                    track.endpoint_nodes[1],
                )
            elif track_code in TRACK_ENDPOINTS:
                endpoints[track_code] = TRACK_ENDPOINTS[track_code]
        return endpoints

    def _build_track_route_meta(self) -> dict[str, TrackRouteMeta]:
        route_meta: dict[str, TrackRouteMeta] = {}
        for track_code, track in self.master.tracks.items():
            if track.connection_nodes or track.terminal_branch is not None:
                route_meta[track_code] = TrackRouteMeta(
                    connection_nodes=track.connection_nodes,
                    terminal_branch=track.terminal_branch,
                    reverse_branches=track.reverse_branches,
                )
            elif track_code in TRACK_ROUTE_META:
                route_meta[track_code] = TRACK_ROUTE_META[track_code]
            elif len(track.endpoint_nodes) == 2:
                route_meta[track_code] = TrackRouteMeta(
                    connection_nodes=tuple(track.endpoint_nodes),
                )
        return route_meta

    def _build_track_graph(self) -> dict[str, list[str]]:
        by_node: dict[str, set[str]] = {}
        for track_code, endpoints in self._track_endpoints.items():
            for node in endpoints:
                by_node.setdefault(node, set()).add(track_code)

        graph: dict[str, set[str]] = {track_code: set() for track_code in self._track_endpoints}
        for tracks in by_node.values():
            for track_code in tracks:
                graph.setdefault(track_code, set()).update(other for other in tracks if other != track_code)
        return {track_code: sorted(neighbors) for track_code, neighbors in graph.items()}

    def _shortest_path(self, source_node: str, target_node: str) -> list[str] | None:
        cache_key = (source_node, target_node)
        if cache_key in self._node_path_cache:
            cached = self._node_path_cache[cache_key]
            return list(cached) if cached is not None else None
        if source_node == target_node:
            return []
        queue: list[tuple[float, str, list[str]]] = [(0.0, source_node, [])]
        best: dict[str, float] = {source_node: 0.0}
        while queue:
            cost, node, branch_codes = heappop(queue)
            if node == target_node:
                self._node_path_cache[cache_key] = tuple(branch_codes)
                return branch_codes
            if cost > best.get(node, float("inf")) + 1e-9:
                continue
            for next_node, branch, branch_length in self._graph.get(node, []):
                next_cost = cost + branch_length
                if next_cost >= best.get(next_node, float("inf")) - 1e-9:
                    continue
                best[next_node] = next_cost
                heappush(queue, (next_cost, next_node, branch_codes + [branch]))
        self._node_path_cache[cache_key] = None
        return None

    def _append_branch(
        self,
        branch_codes: list[str],
        branch: str | None,
        total_length_m: float,
    ) -> float:
        if branch is None:
            return total_length_m
        if branch_codes and branch_codes[-1] == branch:
            return total_length_m
        branch_codes.append(branch)
        physical_route = self.master.physical_routes.get(branch)
        if physical_route is not None:
            total_length_m += physical_route.total_length_m
        return total_length_m

    def _branch_uses_l1(self, branch: str) -> bool:
        endpoints = self._branch_endpoints.get(branch)
        if endpoints is None:
            return False
        return "L1" in endpoints

    def _resolve_reverse_branch_codes(
        self,
        path_tracks: list[str],
        route: ResolvedRoute,
    ) -> list[str]:
        reverse_branch_codes: list[str] = []
        for index, track_code in enumerate(path_tracks):
            track_meta = self._track_route_meta.get(track_code)
            if track_meta is None:
                continue
            candidate_branches: list[str] = []
            if track_meta.terminal_branch is not None and index in {0, len(path_tracks) - 1}:
                candidate_branches.append(track_meta.terminal_branch)
            candidate_branches.extend(track_meta.reverse_branches)
            for branch_code in candidate_branches:
                if branch_code not in route.branch_codes:
                    continue
                if branch_code in reverse_branch_codes:
                    continue
                reverse_branch_codes.append(branch_code)
        return reverse_branch_codes
