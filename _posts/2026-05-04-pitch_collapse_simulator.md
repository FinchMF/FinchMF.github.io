# Pitch, Quantization, and What Gets Lost

*A note is not a frequency. It is a position in a representational system — and which system you choose determines which distinctions survive.*

---

When you move from a finer pitch space to a coarser one, you are not simplifying. You are projecting. The two are not the same thing.

The simulator takes a melody written in 19-tone equal temperament — 19 steps per octave, each about 63 cents wide — and projects it into the familiar 12-tone chromatic scale. The rule is nearest-neighbor rounding: each 19-TET pitch lands on whichever 12-TET semitone is closest. The result is that several distinct source positions collapse to the same destination. Seven of the twelve target bins absorb more than one source pitch. About 37% of the original pitch distinctions are gone: not rounded, and not recoverable without prior context.

![Dimensional Collapse: 19-TET to 12-TET. Each arrow is a projection; each orange × marks a collision — two sources, one destination.](/assets/blog_pitch_logic_final_assets/pitch_collapse_3d_1.png)

*19-TET → 12-TET. Purple and orange arrows show each note's collapse trajectory. Orange × = aliasing event. Color intensity encodes information loss magnitude.*

The per-note damage is not uniform. Some pitches land cleanly within 5¢ of a 12-TET target; others miss by 47¢ — more than a quarter-tone, more than a half-step within the source system.

![Per-note quantization error. Green = tolerable (< 10¢). Purple = noticeable (10–25¢). Orange = categorical loss (> 25¢, exceeds a 19-TET half-step).](/assets/blog_pitch_logic_final_assets/pitch_info_loss_bars_1.png)

![Projection collision matrix. Each row is one 12-TET target. Rows marked 2→1 receive two distinct 19-TET sources — irrecoverable degeneracy distributed uniformly across the octave.](/assets/blog_pitch_logic_final_assets/pitch_projection_matrix_1.png)

This is the structure of pitch collapse: a many-to-one projection, non-uniform information loss, no recovery path without external context. It is not specific to tuning systems. The same pattern shows up anywhere a continuous field gets committed to a discrete grid.

---

## The Ensemble Problem

A second experiment introduces a live ensemble and asks what happens when different instruments commit to different grids — or to no grid at all.

Five archetypes:

| Instrument | Mode | JI Blend | Memory |
|------------|------|----------|--------|
| Piano | Fixed ET | 0% | 100% |
| Guitar | Fixed ET | 0% | 100% |
| Winds | Semi-flex | 45% JI / 55% ET | 35% |
| Strings | Adaptive JI | 100% | 25% |
| Voice | Adaptive JI | 100% | 15% |

The memory parameter models inertia: an instrument with 100% memory stays exactly where it was; an instrument with 15% memory updates almost immediately to each new harmonic target. The JI blend determines how much of the correction force it draws from just-intonation ratios versus equal-tempered anchors.

Geometrically, Piano and Guitar stay on the ET lattice — a discrete, rigid vertical plane — while Strings and Voice move through the JI field, a continuous surface shaped by harmonic context. Winds occupy the gap.

![The Dual Hyperplane View. Right plane: ET Lattice (discrete). Left plane: JI Field (continuous). Colored arrows show correction force on adaptive instruments.](/assets/blog_pitch_logic_final_assets/pitch_dual_hyperplane_1.png)

*The two representational geometries in one frame. Piano never moves from the lattice face. Voice traverses the full field.*

The contrast is clearest when you isolate individual instruments. Piano's "trajectory" across a seven-chord progression is a vertical line: one pitch-deviation coordinate across all seven chords.

![Piano (fixed ET, memory 100%). A straight vertical line. The lattice does not react to harmonic context.](/assets/blog_pitch_logic_final_assets/pitch_piano_trajectory_1.png)

Voice charts a wide arc through the JI field, most displaced at the remote chords.

![Voice (adaptive JI, memory 15%). Wide, harmonic-context-sensitive trajectory through the JI field.](/assets/blog_pitch_logic_final_assets/pitch_voice_trajectory_1.png)

---

## Where the Spread Becomes a Problem

The chord progression is Cmaj7 → E7 → Ab7 → Dbmaj7 → F#maj7 → C#maj7 → Cmaj7. It starts at home, moves through maximally remote harmonic territory, and returns. For fixed instruments, this journey is invisible. For adaptive ones, it accumulates.

![Per-instrument intonation drift across the progression. Piano and Guitar hold at 0¢. Strings and Voice peak near 850¢ at Ab7.](/assets/blog_pitch_logic_final_assets/pitch_tuning_drift_1.png)

By Ab7, the voice has moved roughly 857¢ from the ET anchor. That is not a tuning error; it is what faithful JI correction looks like when the harmonic context has moved nearly as far from the starting key as possible.

![Ensemble snapshot at Ab7 (Step 3/7). Piano: +0.0¢. Winds: +358.9¢. Voice: +857.0¢.](/assets/blog_pitch_logic_final_assets/pitch_chord_ab7_1.png)

*Piano hears Ab7 one way. The voice is singing it from nearly an octave away in tuning space. They are playing the same chord.*

The maximum inter-instrument spread at Ab7 approaches 1200¢ — a full octave of disagreement between the most rigid and most adaptive actors.

![Maximum intonation spread at each chord step. Peaks near 1200¢ at Ab7.](/assets/blog_pitch_logic_final_assets/pitch_max_spread_1.png)

![Unison beating risk (Hz proxy). Peaks at Ab7 — the geometric apex is also the acoustic conflict apex.](/assets/blog_pitch_logic_final_assets/pitch_beating_risk_1.png)

The ensemble is not in disagreement about what chord they are playing. They are in disagreement about where, in pitch space, that chord lives.

---

## Coherence and What a Field Can Preserve

There is a structural parallel here to quantum decoherence. This is not metaphorical; it is formally similar. Both situations involve a system that starts in a well-defined state, accumulates noise from competing constraints, and decays toward maximum ambiguity.

In the ensemble:
- dephasing noise → intonation drift pressure between instruments
- weak measurement/backaction → continuous listening and correction
- lattice-only control → rigid ET anchoring, no adaptation
- lattice + field control → adaptive tuning, hybrid negotiation

Applied to the Bloch sphere — a geometric representation of a two-state system — the two control strategies leave very different trajectories:

![Bloch sphere trajectories. Left: lattice-only (ET) — erratic, ends far from coherence target. Right: hybrid (ET+JI adaptive) — disciplined arc, converges near target.](/assets/blog_pitch_logic_final_assets/pitch_bloch_sphere_1.png)

*Axes: X = pitch drift, Y = phase jitter, Z = ET/JI stability. Lattice-only control spirals chaotically. Hybrid control traces a clean arc.*

Both strategies decay toward the same long-run level of ambiguity (fidelity = 0.5, the maximum-entropy state). The difference is transient: the hybrid degrades smoothly during the critical window when the ensemble is negotiating remote terrain, while lattice-only control oscillates violently through that same window.

![Coherence fidelity under noise. Both strategies converge toward 0.5. The hybrid does so without the transient oscillations of the lattice-only approach.](/assets/blog_pitch_logic_final_assets/pitch_fidelity_decay_1.png)

In musical terms: rigid ET anchoring does not prevent eventual pitch divergence, it just makes the path there more disruptive.

---

## The Geometry of Return

Compressing the full intonation state of all five instruments into three principal components gives a navigable record of the progression. The ensemble starts at Cmaj7, moves outward, and returns.

![PCA manifold of ensemble intonation state. Each node is one chord step. Cmaj7_2 nearly, but not exactly, coincides with Cmaj7.](/assets/blog_pitch_logic_final_assets/pitch_pca_manifold_1.png)

It does not return to the same point.

![Distance from tonic state in PCA space. Peaks at F#maj7, collapses on return. The residual gap at Cmaj7_2 is the unrecoverable fraction.](/assets/blog_pitch_logic_final_assets/pitch_pca_drift_1.png)

The gap between Cmaj7 and Cmaj7_2 is not an experiment artifact. It is the portion of the original projection that field correction cannot restore. Harmonic memory and adaptive intonation recover most of the structure; they do not recover all of it. What is gone was lost at the moment of projection — when the first note committed to a grid that had fewer degrees of freedom than the field it was drawn from.

The broader collapse experiments show the same pattern: coarse projection preserves large-scale structure, destroys fine-grained distinction, and reconstruction recovers only what is implied by the remaining redundancy. Pitch collapse is not a single event. It is a family of projections with the same structure:

1. Choose a measurement basis.
2. Annihilate off-basis distinctions.
3. Infer what can be recovered from priors and constraints.

---

## The Enharmonic Case

The sharpest test of this is the enharmonic equivalence of G# and Ab. In equal temperament, they are the same pitch — the same byte in a fixed lookup table. In adaptive just intonation, they are not.

![Adaptive JI instruments at C#maj7: G# and Ab diverge by ~2¢ — a harmonic identity distinction, not a tuning error.](/assets/blog_pitch_logic_final_assets/pitch_enharmonic_adaptive_1.png)

![Fixed ET instruments: G# and Ab collapse to identical pitch. Not a per-chord decision — a categorical suppression inherent to the representation.](/assets/blog_pitch_logic_final_assets/pitch_enharmonic_et_1.png)

The ET instrument does not play G# and Ab the same because it is imprecise. It plays them the same because its representation has no variable in which to store the distinction. This is what a representational commitment means in practice: it determines not only what you can express, but also what you can know.

---

## The Practical Resolution

Neither pole works alone.

A lattice with no field yields stability without expressiveness — technically reproducible, harmonically rigid, slowly draining warmth in remote keys. A field with no lattice yields expressiveness without stability — adaptive, contextually rich, and liable to drift without a fixed reference.

The useful regime is hybrid: discrete grid for addressability and repeatability, continuous field for expressive correction, memory parameter to tune adaptation speed against stability.

That hybrid is not a compromise. It is a control system. The grid is the commitment. The field is the conversation. Pitch collapse is what remains when one of them disappears.

---

*Pitch quantization is a projection — from a field of possible intonations into a coarser lattice. What survives the projection depends on the basis you choose. What can be recovered afterward depends on how well you understand what you destroyed.*