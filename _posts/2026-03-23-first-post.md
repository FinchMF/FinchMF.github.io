---
layout: post
title: "Neural Networks as System Identification Engines"
date: 2026-03-23
---

## Idea

Instead of treating neural networks as black-box predictors,
we can treat them as **learned coordinate systems** for a system.

Once trained:
- weights encode structure
- activations encode state
- layers encode transformations

## Interpretation

This aligns with:
- dynamical systems
- Koopman operator ideas
- latent mode decomposition

## Direction

Future work:
- extract modes
- analyze stability (Jacobian eigenvalues)
- simulate perturbations