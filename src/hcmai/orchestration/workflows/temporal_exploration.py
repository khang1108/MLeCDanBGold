"""Own one in-memory temporal exploration branch and its revision history.

This module retains only one selected video's immutable score copy. It does
not own API sessions, persistence, full-corpus caches, or answer submission.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from threading import RLock
from typing import Literal

from hcmai.orchestration.workflows.temporal_search import (
    DecoderConfigSnapshot,
    TemporalSearchService,
)
from hcmai.retrieval.plan import KISRetrievalPlan
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.constraints import (
    Conditions,
    Interval,
    build_mask,
    merge_intervals,
    validate_interval,
)
from hcmai.temporal.dp import AlignedPath

ExplorationStatus = Literal[
    "ok",
    "contradictory_conditions",
    "no_indexed_frames",
    "no_valid_path",
]


@dataclass(frozen=True, slots=True)
class QueryBinding:
    """Bind an exploration branch to one immutable scoring generation."""

    retrieval_plan: KISRetrievalPlan
    semantic_revision: int
    event_version: str
    use_dense: bool
    use_bm25: bool
    scoring_revision: str


@dataclass(frozen=True, slots=True)
class ExplorationView:
    """Present one immutable snapshot of branch conditions and results."""

    revision: int
    event_version: str
    video_id: str
    conditions: Conditions
    status: ExplorationStatus
    paths: tuple[AlignedPath, ...]
    changed_event_indices: tuple[int, ...]
    comparison_available: bool
    can_undo: bool


class ExplorationConflict(ValueError):
    """Report a stale exploration revision or binding generation."""


class ExplorationUnavailable(RuntimeError):
    """Report an unopened branch or unavailable scoring dependency."""


class TemporalExploration:
    """Retain feedback and cached scores for one selected video."""

    def __init__(self, temporal: TemporalSearchService) -> None:
        """Bind the temporal scoring service."""

        self._temporal = temporal
        self._binding: QueryBinding | None = None
        self._video: VideoEventScores | None = None
        self._decoder_config: DecoderConfigSnapshot | None = None
        self._conditions: Conditions | None = None
        self._history: list[Conditions] = []
        self._revision = 0
        self._view: ExplorationView | None = None
        # One lock makes guard, evaluation, and publication one local transaction.
        self._lock = RLock()

    def open(
        self,
        binding: QueryBinding,
        video_id: str,
        window: Interval,
    ) -> ExplorationView:
        """Open the branch and acquire its selected-video score matrix."""

        with self._lock:
            if self._binding is not None:
                raise ValueError("exploration branch is already active")

            _validate_binding(binding)
            _validate_nonblank(video_id, "video_id")
            validate_interval(window)
            conditions = Conditions(
                window=(int(window[0]), int(window[1])),
                confirmed=(None,) * len(binding.retrieval_plan.events),
                rejected=((),) * len(binding.retrieval_plan.events),
            )

            try:
                scores, _ = self._temporal.score_videos(
                    binding.retrieval_plan.canonical_texts,
                    retrieval_events=(
                        binding.retrieval_plan.dense_texts
                        if binding.use_dense else binding.retrieval_plan.canonical_texts
                    ),
                    caption_events=(
                        binding.retrieval_plan.bm25_texts if binding.use_bm25 else None
                    ),
                    use_dense=binding.use_dense,
                    use_bm25=binding.use_bm25,
                )
                selected = next(
                    (video for video in scores if video.video_id == video_id),
                    None,
                )
                if selected is None:
                    raise ExplorationUnavailable(
                        "selected video scoring is unavailable"
                    )
                frozen = _freeze_video(selected)
                snapshot = getattr(
                    self._temporal, "snapshot_decoder_config", None
                )
                decoder_config = snapshot() if callable(snapshot) else None
                status, paths = self._evaluate_video(
                    frozen,
                    conditions,
                    decoder_config,
                )
            except OSError as error:
                raise ExplorationUnavailable(
                    "temporal exploration scoring is unavailable"
                ) from error

            revision = self._revision + 1
            view = self._make_view(
                binding=binding,
                video_id=video_id,
                revision=revision,
                conditions=conditions,
                status=status,
                paths=paths,
                can_undo=False,
                previous=None,
            )

            # Publish only after score acquisition and initial evaluation succeed.
            self._binding = binding
            self._video = frozen
            self._decoder_config = decoder_config
            self._conditions = conditions
            self._revision = revision
            self._view = view
            return view

    def apply(
        self,
        *,
        expected_revision: int,
        event_version: str,
        scoring_revision: str,
        action: Literal["confirm", "reject", "window", "unknown"],
        event_index: int | None = None,
        interval: Interval | None = None,
    ) -> ExplorationView:
        """Apply one guarded feedback action."""

        with self._lock:
            binding, video, old, previous = self._require_active()
            self._guard(
                expected_revision=expected_revision,
                event_version=event_version,
                scoring_revision=scoring_revision,
            )
            new = self._updated_conditions(
                old,
                action=action,
                event_index=event_index,
                interval=interval,
            )
            if new is old or new == old:
                return previous

            # Evaluate before mutation so infrastructure failures keep history intact.
            status, paths = self._evaluate_active(new)
            revision = self._revision + 1
            view = self._make_view(
                binding=binding,
                video_id=video.video_id,
                revision=revision,
                conditions=new,
                status=status,
                paths=paths,
                can_undo=True,
                previous=previous,
            )

            self._history.append(old)
            self._conditions = new
            self._revision = revision
            self._view = view
            return view

    def undo(
        self,
        *,
        expected_revision: int,
        event_version: str,
        scoring_revision: str,
    ) -> ExplorationView:
        """Restore the last condition snapshot under a new revision."""

        with self._lock:
            binding, video, _, previous = self._require_active()
            self._guard(
                expected_revision=expected_revision,
                event_version=event_version,
                scoring_revision=scoring_revision,
            )
            if not self._history:
                return previous

            restored = self._history[-1]
            status, paths = self._evaluate_active(restored)
            revision = self._revision + 1
            view = self._make_view(
                binding=binding,
                video_id=video.video_id,
                revision=revision,
                conditions=restored,
                status=status,
                paths=paths,
                can_undo=len(self._history) > 1,
                previous=previous,
            )

            self._conditions = restored
            self._revision = revision
            self._view = view
            # Pop after evaluation so an unavailable undo keeps its checkpoint.
            self._history.pop()
            return view

    def close(self, *, expected_revision: int) -> None:
        """Close the branch after checking its revision."""

        with self._lock:
            self._require_active()
            if not _same_revision(expected_revision, self._revision):
                raise ExplorationConflict("stale exploration revision")

            self._binding = None
            self._video = None
            self._decoder_config = None
            self._conditions = None
            self._history.clear()
            self._view = None

    def current(self) -> ExplorationView:
        """Return the current immutable branch snapshot."""

        with self._lock:
            return self._require_active()[3]

    def _require_active(
        self,
    ) -> tuple[QueryBinding, VideoEventScores, Conditions, ExplorationView]:
        """Return complete active state or report an unavailable session."""

        if (
            self._binding is None
            or self._video is None
            or self._conditions is None
            or self._view is None
        ):
            raise ExplorationUnavailable("temporal exploration is not open")
        return self._binding, self._video, self._conditions, self._view

    def _guard(
        self,
        *,
        expected_revision: int,
        event_version: str,
        scoring_revision: str,
    ) -> None:
        """Reject stale state and scoring generations before payload work."""

        binding, _, _, _ = self._require_active()
        # Stale guards run before validation so old UI work cannot affect history.
        if not _same_revision(expected_revision, self._revision):
            raise ExplorationConflict("stale exploration revision")
        if event_version != binding.event_version:
            raise ExplorationConflict("stale event version")
        if scoring_revision != binding.scoring_revision:
            raise ExplorationConflict("stale scoring revision")

    def _updated_conditions(
        self,
        old: Conditions,
        *,
        action: str,
        event_index: int | None,
        interval: Interval | None,
    ) -> Conditions:
        """Validate one action and return its immutable condition update."""

        if not isinstance(action, str) or action not in {
            "confirm",
            "reject",
            "window",
            "unknown",
        }:
            raise ValueError("unsupported temporal exploration action")

        event_action = action in {"confirm", "reject", "unknown"}
        if event_action:
            if (
                isinstance(event_index, bool)
                or not isinstance(event_index, Integral)
                or not 0 <= event_index < len(old.confirmed)
            ):
                raise ValueError("event action requires a valid event_index")
            event_index = int(event_index)
        elif event_index is not None:
            raise ValueError("window action does not accept event_index")

        if action == "unknown":
            if interval is not None:
                raise ValueError("unknown action does not accept interval")
            return old
        if interval is None:
            raise ValueError(f"{action} action requires interval")
        validate_interval(interval)
        normalized = (int(interval[0]), int(interval[1]))

        if action == "window":
            return replace(old, window=normalized)
        if action == "confirm":
            confirmed = list(old.confirmed)
            confirmed[event_index] = normalized
            return replace(old, confirmed=tuple(confirmed))

        rejected = list(old.rejected)
        rejected[event_index] = merge_intervals(
            rejected[event_index] + (normalized,)
        )
        return replace(old, rejected=tuple(rejected))

    def _evaluate_video(
        self,
        video: VideoEventScores,
        conditions: Conditions,
        decoder_config: DecoderConfigSnapshot | None,
    ) -> tuple[ExplorationStatus, tuple[AlignedPath, ...]]:
        """Build the mask and decode one supplied selected-video snapshot."""

        allowed, status = build_mask(video.timestamps_ms, conditions)
        if status != "ready":
            return status, ()
        if decoder_config is None:
            paths = self._temporal.decode_video(video, allowed=allowed)
        else:
            paths = self._temporal.decode_video(
                video,
                allowed=allowed,
                decoder_config=decoder_config,
            )
        return ("ok", paths) if paths else ("no_valid_path", ())

    def _evaluate_active(
        self,
        conditions: Conditions,
    ) -> tuple[ExplorationStatus, tuple[AlignedPath, ...]]:
        """Evaluate active cached scores without changing branch state."""

        _, video, _, _ = self._require_active()
        try:
            return self._evaluate_video(video, conditions, self._decoder_config)
        except OSError as error:
            raise ExplorationUnavailable(
                "temporal exploration evaluation is unavailable"
            ) from error

    @staticmethod
    def _make_view(
        *,
        binding: QueryBinding,
        video_id: str,
        revision: int,
        conditions: Conditions,
        status: ExplorationStatus,
        paths: tuple[AlignedPath, ...],
        can_undo: bool,
        previous: ExplorationView | None,
    ) -> ExplorationView:
        """Build a view with a canonical best-path comparison when possible."""

        changed, comparison_available = _path_diff(previous, paths)

        return ExplorationView(
            revision=revision,
            event_version=binding.event_version,
            video_id=video_id,
            conditions=conditions,
            status=status,
            paths=paths,
            changed_event_indices=changed,
            comparison_available=comparison_available,
            can_undo=can_undo,
        )


def _path_diff(
    previous: ExplorationView | None,
    paths: tuple[AlignedPath, ...],
) -> tuple[tuple[int, ...], bool]:
    """Compare best paths by canonical frame identity and timestamp pairs."""

    if previous is None:
        return (), False
    comparison_available = bool(previous.paths and paths)
    # Empty transitions have no comparable hypothesis, not an unchanged path.
    if not comparison_available:
        return (), False

    before = previous.paths[0]
    after = paths[0]
    changed = tuple(
        event_index
        for event_index, (before_pair, after_pair) in enumerate(
            zip(
                zip(before.frame_ids, before.timestamps_ms, strict=True),
                zip(after.frame_ids, after.timestamps_ms, strict=True),
                strict=True,
            )
        )
        if before_pair != after_pair
    )
    return changed, True


def _freeze_video(video: VideoEventScores) -> VideoEventScores:
    """Return a selected-video copy whose arrays cannot be mutated in place."""

    arrays = {}
    for name in ("frame_ids", "frame_idx", "timestamps_ms", "scores"):
        # Copy before freezing so scorer-owned arrays stay writable and reusable.
        value = getattr(video, name).copy()
        value.setflags(write=False)
        arrays[name] = value
    return replace(video, **arrays)


def _validate_binding(binding: QueryBinding) -> None:
    """Validate a complete normalized scoring snapshot before acquisition."""

    if not isinstance(binding, QueryBinding):
        raise ValueError("binding must be a QueryBinding")
    _validate_nonblank(binding.event_version, "event_version")
    _validate_nonblank(binding.scoring_revision, "scoring_revision")
    if (
        not isinstance(binding.semantic_revision, int)
        or isinstance(binding.semantic_revision, bool)
        or binding.semantic_revision < 1
    ):
        raise ValueError("semantic_revision must be a positive integer")
    if not isinstance(binding.retrieval_plan, KISRetrievalPlan):
        raise ValueError("binding requires a KISRetrievalPlan")
    binding.retrieval_plan.validate_text_sources(
        use_dense=binding.use_dense, use_bm25=binding.use_bm25,
    )


def _validate_nonblank(value: str, name: str) -> None:
    """Require one nonblank string without rewriting its identity."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")


def _same_revision(expected: object, current: int) -> bool:
    """Compare revisions without accepting booleans as integer revisions."""

    return (
        isinstance(expected, Integral)
        and not isinstance(expected, bool)
        and int(expected) == current
    )
