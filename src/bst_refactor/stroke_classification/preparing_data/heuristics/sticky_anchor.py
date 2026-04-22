"""``sticky_anchor`` heuristic: per-slot EMA + fixed-prior tracking.

Picks one detection per slot (Top, Bottom) by proximity to a weighted
anchor. The anchor for slot ``s`` is:

    effective_anchor[s] = prior_weight * halfcourt_centre[s]
                        + (1 - prior_weight) * ema[s]

``halfcourt_centre`` is derived per-clip from the homography borders and
collapses to ``(0.5, 0.25)`` and ``(0.5, 0.75)`` for ShuttleSet's
canonical rectangle. The EMA starts at ``halfcourt_centre`` and updates
with the slot's picked court position each frame, gated on that pick
landing inside the court by ``update_gate_eps`` (pollution guard).

Per-frame flow:

A. Score-filter raw detections, project each bbox-bottom-centre to
   normalised court coords, drop NaN projections.
B. Compute effective anchors and the (candidate, slot) distance matrix.
C. For each slot in order (Bottom first, then Top):
   - Drop candidates past ``sanity_ceiling``.
   - Drop candidates closer to the OTHER slot's anchor (Voronoi partition
     against cross-half capture).
   - For Top, drop whichever candidate Bottom assigned.
   - ``argmin`` D on survivors. If any survivor is within
     ``tiebreaker_tol`` of the winner's D: drop sitting candidates
     (body-frame knee/torso projection), pick largest bbox area among
     what remains. Fall back to the original ``argmin`` if the sitting
     filter drops everyone.
D. Rally-presence check: if both slots picked but neither pick lands
   inside ``[-generous_margin, 1 + generous_margin]`` on both axes, zero
   both (catches cutaway / pure-bystander frames).
E. Write outputs. EMA resets to ``halfcourt_centre[s]`` on zero; updates
   by ``ema_alpha`` on a picked slot whose pick is within the
   ``update_gate_eps`` in-court test.

Full spec with rationale: "Sticky_anchor design, finalised (2026-04-22)"
section of ``scratch/architecture_notes/mmpose_heuristic/mmpose_heuristic_investigation.md``.
"""
from __future__ import annotations

import numpy as np

from .base import ClipContext, HeuristicOutput, RawClip

J = 17
SLOT_TOP = 0
SLOT_BOTTOM = 1
SLOT_ORDER = (SLOT_BOTTOM, SLOT_TOP)  # pick order: Bottom first, Top second

# COCO keypoint indices used by the sitting test.
SHOULDER_L, SHOULDER_R = 5, 6
HIP_L, HIP_R = 11, 12
KNEE_L, KNEE_R = 13, 14


def _compute_halfcourt_centres(court_info: dict) -> np.ndarray:
    """Halfcourt centres in normalised [0, 1] coords, returned as (2, 2).

    Row 0 = Top (y = 0.25 for ShuttleSet), row 1 = Bottom (y = 0.75).
    Computed from the homography borders rather than hardcoded so the
    code stays correct for any future homography that maps to a
    non-ShuttleSet canonical rectangle. For ShuttleSet the formula
    collapses to the constants above.
    """
    from pipeline.court_utils import normalize_position  # noqa: PLC0415

    bL, bR = court_info["border_L"], court_info["border_R"]
    bU, bD = court_info["border_U"], court_info["border_D"]

    cx = (bL + bR) / 2
    y_top = bU + (bD - bU) / 4
    y_bot = bU + 3 * (bD - bU) / 4

    raw = np.array([[cx, cx], [y_top, y_bot]], dtype=np.float64)  # (2, 2)
    return normalize_position(raw, court_info).T  # (2, 2); row = slot


def _project_bbox_bottom_centre(
    bboxes: np.ndarray, ctx: ClipContext,
) -> np.ndarray:
    """Project (n, 4) pixel-space bboxes to (n, 2) normalised court coords.

    Uses bbox bottom-centre ``((x1+x2)/2, y2)`` as the foot proxy.
    """
    from pipeline.court_utils import normalize_position, to_court_coordinate  # noqa: PLC0415

    bottom_centres = np.stack(
        [(bboxes[:, 0] + bboxes[:, 2]) / 2, bboxes[:, 3]], axis=0,
    )  # (2, n)
    court = to_court_coordinate(
        bottom_centres, ctx.vid, ctx.all_court_info, ctx.res_df,
    )  # (2, n)
    normalised = normalize_position(court, ctx.all_court_info[ctx.vid])  # (2, n)
    return normalised.T  # (n, 2)


def _is_sitting(kp: np.ndarray, sitting_threshold: float) -> bool:
    """Body-frame sitting test.

    Projects the knee-offset-from-hip onto the hip-to-shoulder axis. A
    standing / airborne player has knees in the body-down direction
    (ratio around -0.7 to -0.9); a sitting person has knees roughly
    perpendicular to the body axis (ratio near 0). Returns True when the
    ratio exceeds ``sitting_threshold`` (default -0.3).
    """
    sh = (kp[SHOULDER_L] + kp[SHOULDER_R]) / 2
    hp = (kp[HIP_L] + kp[HIP_R]) / 2
    kn = (kp[KNEE_L] + kp[KNEE_R]) / 2
    body_up = sh - hp
    torso_len_sq = float(body_up @ body_up)
    if torso_len_sq < 1e-6:
        return False  # Degenerate pose; defer to anchor distance.
    knee_vec = kn - hp
    ratio = float((knee_vec @ body_up) / torso_len_sq)
    return ratio > sitting_threshold


def _in_generous_court(pos: np.ndarray, margin: float) -> bool:
    return bool(
        -margin <= pos[0] <= 1 + margin and -margin <= pos[1] <= 1 + margin
    )


def apply(raw: RawClip, ctx: ClipContext, **hyperparams) -> HeuristicOutput:
    """Apply the sticky_anchor heuristic to a raw clip."""
    from preparing_data.prepare_train_on_shuttleset import (  # noqa: PLC0415
        normalize_joints,
    )

    prior_weight = hyperparams.get("prior_weight", 0.75)
    ema_alpha = hyperparams.get("ema_alpha", 0.1)
    sanity_ceiling = hyperparams.get("sanity_ceiling", 0.6)
    generous_margin = hyperparams.get("generous_margin", 0.15)
    score_filter = hyperparams.get("score_filter", 0.2)
    tiebreaker_tol = hyperparams.get("tiebreaker_tol", 0.05)
    sitting_threshold = hyperparams.get("sitting_threshold", -0.3)
    update_gate_eps = hyperparams.get("update_gate_eps", 0.01)

    court_info = ctx.all_court_info[ctx.vid]
    halfcourt_centre = _compute_halfcourt_centres(court_info)  # (2, 2)

    num_frames = raw.kps.shape[0]
    failed = np.zeros(num_frames, dtype=bool)
    pos = np.zeros((num_frames, 2, 2), dtype=np.float64)
    joints = np.zeros((num_frames, 2, J, 2), dtype=np.float64)

    # Per-slot EMA, initialised to halfcourt_centre.
    ema = halfcourt_centre.copy()

    for f in range(num_frames):
        n = int(raw.ndet[f])
        if n == 0:
            failed[f] = True
            ema[:] = halfcourt_centre
            continue

        # Step A: score filter on real detections.
        scores_f = raw.scores[f, :n]
        pass_score = scores_f > score_filter
        if not pass_score.any():
            failed[f] = True
            ema[:] = halfcourt_centre
            continue

        keep_idx = np.nonzero(pass_score)[0]
        bboxes_f = raw.bboxes[f, keep_idx].astype(np.float64)  # (k, 4)
        kps_f = raw.kps[f, keep_idx].astype(np.float64)  # (k, J, 2)

        court_base_pos = _project_bbox_bottom_centre(bboxes_f, ctx)  # (k, 2)
        valid = ~np.isnan(court_base_pos).any(axis=1)
        if not valid.any():
            failed[f] = True
            ema[:] = halfcourt_centre
            continue

        bboxes_f = bboxes_f[valid]
        kps_f = kps_f[valid]
        court_base_pos = court_base_pos[valid]
        k = court_base_pos.shape[0]

        # Step B: effective anchors + full distance matrix.
        effective_anchor = (
            prior_weight * halfcourt_centre
            + (1 - prior_weight) * ema
        )  # (2, 2); row = slot
        distances = np.linalg.norm(
            court_base_pos[:, None, :] - effective_anchor[None, :, :],
            axis=-1,
        )  # (k, 2); distances[c, s]

        # Precompute per-candidate sitting + bbox area (tiebreaker only).
        is_sitting = np.array(
            [_is_sitting(kps_f[i], sitting_threshold) for i in range(k)],
            dtype=bool,
        )
        bbox_areas = (bboxes_f[:, 2] - bboxes_f[:, 0]) * (
            bboxes_f[:, 3] - bboxes_f[:, 1]
        )

        # Step C: process slots Bottom first, then Top.
        picks: list[int] = [-1, -1]
        for s in SLOT_ORDER:
            other = SLOT_TOP if s == SLOT_BOTTOM else SLOT_BOTTOM

            eligible = np.ones(k, dtype=bool)
            eligible &= distances[:, s] <= sanity_ceiling
            # Closer-to-own-anchor rule: keep candidates where own slot's
            # anchor is no further than the other slot's anchor.
            eligible &= distances[:, other] >= distances[:, s]
            if picks[other] >= 0:
                eligible[picks[other]] = False

            if not eligible.any():
                continue

            eligible_idx = np.nonzero(eligible)[0]
            winner = int(eligible_idx[np.argmin(distances[eligible_idx, s])])
            winner_d = distances[winner, s]

            # Tiebreaker: any other eligible within tiebreaker_tol.
            tied = eligible & (
                np.abs(distances[:, s] - winner_d) < tiebreaker_tol
            )
            if tied.sum() > 1:
                standing_tied = tied & ~is_sitting
                if standing_tied.any():
                    st_idx = np.nonzero(standing_tied)[0]
                    winner = int(st_idx[np.argmax(bbox_areas[st_idx])])
                # Else: sitting dropped everyone; revert to original argmin.

            picks[s] = winner

        # Step D: rally-presence check.
        if picks[SLOT_TOP] >= 0 and picks[SLOT_BOTTOM] >= 0:
            top_p = court_base_pos[picks[SLOT_TOP]]
            bot_p = court_base_pos[picks[SLOT_BOTTOM]]
            if not _in_generous_court(top_p, generous_margin) and not _in_generous_court(
                bot_p, generous_margin
            ):
                picks = [-1, -1]

        # Step E: write outputs + update EMAs.
        frame_has_zero = False
        for s in (SLOT_TOP, SLOT_BOTTOM):
            if picks[s] < 0:
                frame_has_zero = True
                ema[s] = halfcourt_centre[s]
                continue
            cbp = court_base_pos[picks[s]]
            pos[f, s] = cbp
            joints[f, s] = normalize_joints(
                arr=kps_f[picks[s]][None, :, :],
                bbox=bboxes_f[picks[s]][None, :],
                v_height=None,
                center_align=True,
            )[0]
            if _in_generous_court(cbp, update_gate_eps):
                ema[s] = ema_alpha * cbp + (1 - ema_alpha) * ema[s]

        failed[f] = frame_has_zero

    return HeuristicOutput(pos=pos, joints=joints, failed=failed)
