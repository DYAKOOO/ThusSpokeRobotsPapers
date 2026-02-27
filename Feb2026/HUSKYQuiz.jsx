'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { ChevronLeft, ChevronRight, RefreshCw, BookOpen, Trophy, Clock, CheckCircle, XCircle, GripVertical, Zap, AlertTriangle } from 'lucide-react';

/**
 * HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control
 * Interactive Comprehension Quiz
 *
 * Source: Han et al. (2026), arXiv:2602.03205v1
 * Coverage: Sections I–VI + Appendices A–H
 * Learning Cycle: Intuition → Technical → Implementation → Validation
 */

const quizData = [
  // ============================================================
  // ABSTRACT & INTRO — 10%
  // ============================================================
  {
    id: 1,
    question: "What is the core novel contribution that distinguishes HUSKY from prior whole-body control frameworks?",
    questionType: "single-choice",
    options: [
      "It uses model predictive control (MPC) for real-time trajectory optimization on a humanoid",
      "It integrates physics-aware modeling of the humanoid–skateboard coupling with a learning-based whole-body controller",
      "It is the first framework to use reinforcement learning for any legged locomotion task",
      "It uses a convolutional network to process visual observations from onboard cameras"
    ],
    answer: 1,
    explanation: {
      intuition: "Most humanoid controllers assume the robot stands on solid, static ground — like a dancer on a fixed stage. HUSKY is different because the 'stage' is a skateboard: a passive, underactuated platform that moves and tilts in response to the robot's own weight. The key insight is that you can't ignore this coupling — you have to model it explicitly and let it guide learning.",
      math: "From the Abstract and Section I (page 1): 'we propose HUSKY, a learning-based framework that integrates humanoid-skateboard system modeling and physics-aware whole-body control.' The three novelties are: (1) explicit tilt–steering constraint modeling (Equation 1), (2) AMP-based pushing, (3) physics-guided steering with trajectory-planned transitions. Traditional MPC approaches (cited as [32, 33, 42]) are noted to fail due to high-dimensional non-convex optimization costs.",
      computation: `# Conceptual pseudocode of HUSKY's high-level pipeline
# Source: Section III (pages 3–5)

def HUSKY_framework(state_t, command):
    # Phase 1: System modeling (Appendix A)
    gamma = board.tilt_angle          # board tilt
    sigma = compute_steering(gamma)   # Eq. 1: tan σ = tan λ · sin γ

    # Phase 2: Phase detection
    phase = detect_phase(state_t)     # pushing / steering / transition

    # Phase 3: Phase-specific policy
    if phase == "pushing":
        action = amp_pushing_policy(state_t, command.velocity)
    elif phase == "steering":
        gamma_ref = compute_tilt_ref(command.heading)  # Eq. 10
        action = steering_policy(state_t, gamma_ref)
    else:
        trajectory = bezier_slerp_plan(state_t, next_phase_ref)  # Eq. 12–13
        action = transition_policy(state_t, trajectory)

    return action`,
      connection: "The ablation in Table II (Section IV-B) directly validates each component: removing any one of AMP, tilt guidance, or trajectory transitions causes measurable degradation. Tracking-based pushing alone achieves only 11.12% success rate — confirming the importance of the integrated approach."
    },
    topic: "CORE_CONTRIBUTION",
    section: "Abstract & Section I",
    difficulty: "Easy",
    category: "conceptual",
    pageReference: "Abstract, Page 1; Section I, Pages 1–2",
  },

  {
    id: 2,
    question: "Select ALL challenges that make humanoid skateboarding harder than standard whole-body control tasks:",
    questionType: "multi-select",
    options: [
      "The skateboard is an underactuated, passive wheeled platform subject to non-holonomic constraints",
      "The robot must control its own moving support base indirectly through foot contacts",
      "The task requires mastery of hybrid contact dynamics with discrete phase transitions",
      "The humanoid must solve a vision problem to detect skateboard position",
      "Sim-to-real discrepancies are amplified by the tight human–board coupling"
    ],
    answers: [0, 1, 2, 4],
    explanation: {
      intuition: "Imagine learning to skateboard blindfolded. The board shifts under your feet, wheels only roll forward (non-holonomic), and sometimes one foot is on the ground, sometimes both are on the board. Each mode requires completely different balance strategies. Now imagine teaching a robot this — the board's response to its every movement feeds back into the next state. It's a control nightmare!",
      math: "From Section I (page 2): 'This integrated system is governed by non-holonomic constraints and tightly coupled human-object interactions.' The paper explicitly states three co-occurring challenges: (a) hybrid contact dynamics (Section II-B), (b) underactuated wheeled platform, (c) sim-to-real discrepancies (Section III-E). Vision is listed as a future work limitation (Section VI, page 9): 'The limited camera field of view prevents reliable observation.' It is NOT part of the current system.",
      computation: `# Non-holonomic constraint demonstration
# Source: Section II-A (page 2)

import numpy as np

# Skateboard can only move in its forward direction
# Lateral velocity = 0 (no sideslip by wheel constraint)
def non_holonomic_constraint(v_board, heading_angle):
    """
    At any instant, the skateboard velocity must be
    along its heading direction — wheels cannot slip laterally.
    """
    v_lateral = v_board[0] * np.sin(heading_angle) - v_board[1] * np.cos(heading_angle)
    print(f"Lateral velocity (must be ~0): {v_lateral:.4f} m/s")
    return np.abs(v_lateral) < 1e-3  # constraint satisfied?`,
      connection: "Vision-based perception is explicitly listed as a future limitation (Section VI, page 9), so option D is incorrect. The other four challenges are directly stated in Section I–II and motivate the three-part HUSKY design."
    },
    topic: "PROBLEM_FORMULATION",
    section: "Section I–II",
    difficulty: "Medium",
    category: "conceptual",
    pageReference: "Section I, Page 2; Section VI, Page 9",
  },

  // ============================================================
  // RELATED WORK — 5%
  // ============================================================
  {
    id: 3,
    question: "According to the Related Work, why do traditional model-based approaches (MPC) struggle with humanoid skateboarding?",
    questionType: "single-choice",
    options: [
      "MPC requires large neural networks that are too slow for real-time control",
      "MPC cannot handle any form of contact dynamics",
      "High-dimensional non-convex optimization is too costly for real-time use, and simplified models fail to capture non-holonomic underactuated dynamics",
      "MPC-based methods have not been tested on any legged robot platform"
    ],
    answer: 2,
    explanation: {
      intuition: "Imagine trying to predict the next 10 seconds of a skateboard run by solving a giant math puzzle from scratch every millisecond. MPC does exactly this — it's like a chess grandmaster replanning every move. This works for slow, simple problems. But on a skateboard, the physics are so nonlinear and the timing so tight that the puzzle becomes unsolvable in real time.",
      math: "From Section I (page 2): 'the high computational cost of solving high-dimensional, non-convex optimization problems often precludes the real-time responsiveness required for dynamic skateboarding. Moreover, simplified models are insufficient to capture the complex non-holonomic and underactuated dynamics inherent to skateboarding.' References [32, 33, 42] are cited as MPC-based examples that face these limitations.",
      computation: `# Why MPC is expensive for this problem
# Source: Section I discussion, Page 2

# MPC must solve at each timestep (50 Hz):
# min  sum_{k=0}^{N} cost(x_k, u_k)
# s.t. x_{k+1} = f(x_k, u_k)   <-- nonlinear skateboard dynamics
#      g(x_k, u_k) <= 0          <-- non-holonomic + contact constraints

# State dimension: 23 joints + skateboard (3 joints) + base (6DoF) = ~35
# The non-convexity from contact modes makes this NP-hard in general.
# RL sidesteps this by amortizing the optimization over millions of episodes.`,
      connection: "This motivates the Deep RL approach adopted in HUSKY (Section I, page 2), and is validated by the success rate results in Table II where the RL-based HUSKY achieves 100% vs the implicit-model baselines."
    },
    topic: "RELATED_WORK",
    section: "Section I",
    difficulty: "Easy",
    category: "conceptual",
    pageReference: "Section I, Page 2; Related Work Section V, Page 8",
  },

  // ============================================================
  // METHOD — 40%
  // ============================================================
  {
    id: 4,
    question: "The lean-to-steer constraint (Equation 1) states: tan σ = tan λ sin γ. Match each variable to its physical meaning:",
    questionType: "matching",
    pairs: [
      { item: "σ (sigma)", match: "Truck steering angle — how much the front/rear trucks rotate" },
      { item: "λ (lambda)", match: "Rake angle — a fixed geometric property of the skateboard trucks" },
      { item: "γ (gamma)", match: "Board tilt angle — how much the deck is tilted sideways by the humanoid" },
      { item: "tan σ ≈ tan λ sin γ", match: "Larger tilts → proportionally greater steering deflections" }
    ],
    explanation: {
      intuition: "A skateboard steers like bicycle handle-bars hidden in the trucks. When you tilt the board sideways (γ), the truck geometry converts that lean into a turning angle (σ). The rake angle (λ) is fixed by the board's design and acts like a mechanical gear ratio. Lean more → turn more. This is exactly what allows a skater to steer without using their hands!",
      math: "From Section II-A (page 2), Equation 1: tan σ = tan λ sin γ, where λ is the 'constant rake angle of the skateboard' and σ is the 'resulting truck steering angle.' The full derivation is in Appendix A (pages 11–12), using a two-stage rotation sequence: η-rotation about kingpin axis BC (Eq. 16–17) followed by γ-rotation about x-axis (Eq. 18–23), constrained by the wheel-ground contact condition (Eq. 24–26).",
      computation: `import numpy as np

# Lean-to-steer relationship
# Source: Appendix A, Equation 30 (page 12)

def compute_steering_angle(gamma_deg, lambda_deg):
    """
    Args:
        gamma_deg: Board tilt angle in degrees
        lambda_deg: Rake angle (fixed hardware property)
    Returns:
        sigma_deg: Truck steering angle in degrees
    """
    gamma = np.deg2rad(gamma_deg)
    lam = np.deg2rad(lambda_deg)

    # Equation 1: tan(σ) = tan(λ) · sin(γ)
    tan_sigma = np.tan(lam) * np.sin(gamma)
    sigma = np.arctan(tan_sigma)
    return np.rad2deg(sigma)

# Test: standard skateboard rake angle ~65°, tilt 10°
rake_angle = 65  # degrees (typical kingpin angle)
for tilt in [0, 5, 10, 15, 20]:
    steer = compute_steering_angle(tilt, rake_angle)
    print(f"Tilt {tilt:3d}° → Steer {steer:.2f}°")`,
      connection: "The paper validates this model in Figure 4(a) (page 7): 'Omitting the equality constraint in Eq.(1) prevents board tilting from inducing truck steering, leaving the skateboard able only to glide straight forward with negligible turning capability.' This directly confirms the constraint is physically correct and essential."
    },
    topic: "SYSTEM_MODELING",
    section: "Section II-A",
    difficulty: "Medium",
    category: "theoretical",
    pageReference: "Section II-A, Page 2, Equation 1; Appendix A, Pages 11–12",
  },

  {
    id: 5,
    question: "In HUSKY's problem formulation (Section III-A), arrange these components of the proprioceptive state vector o_t^prop in the order they appear in Equation 2:",
    questionType: "ordering",
    options: [
      "Command c_t = [v_cmd, ψ] — target velocity and heading",
      "Previous action a_{t-1} — last motor command",
      "Base angular velocity ω_t",
      "Joint angles θ_t and velocities θ̇_t",
      "Phase variable Φ = (t mod H) / H"
    ],
    answer: [0, 2, 3, 3, 1, 4],
    explanation: {
      intuition: "Think of the proprioceptive state as a robot's 'body awareness' — what it can feel without looking. The command tells it where to go, the angular velocity and gravity tell it how it's tilting, joint angles/velocities describe limb positions, the previous action provides motor memory, and the phase variable acts like a metronome telling the policy where it is in the push-glide cycle.",
      math: "From Section III-A (page 3), Equation 2: o_t^prop = [c_t, ω_t, g_t, θ_t, θ̇_t, a_{t-1}, Φ] ∈ ℝ^78. Note that g_t (projected gravity) is in the full equation but was collapsed into the joint/velocity group in this question for clarity. The phase variable Φ = (t mod H)/H ∈ [0,1) where H is the cycle duration. Total dimension: c_t(2) + ω_t(3) + g_t(3) + θ_t(23) + θ̇_t(23) + a_{t-1}(23) + Φ(1) = 78.",
      computation: `import numpy as np

# Proprioceptive state construction
# Source: Section III-A, Equation 2 (page 3)

def build_proprioceptive_obs(robot, command, phase_t, H, prev_action):
    """
    Returns o_t^prop ∈ ℝ^78
    """
    c_t    = np.array([command.v_cmd, command.psi])     # 2
    omega  = robot.base_angular_velocity                  # 3
    g      = robot.projected_gravity                      # 3
    theta  = robot.joint_angles                           # 23
    dtheta = robot.joint_velocities                       # 23
    a_prev = prev_action                                   # 23
    phi    = (phase_t % H) / H                            # 1

    obs = np.concatenate([c_t, omega, g, theta, dtheta, a_prev, [phi]])
    assert obs.shape[0] == 78
    return obs`,
      connection: "The phase variable Φ is critical — Figure 5 (page 7) shows policies trained WITHOUT transition guidance never learn to switch phases even with 50/50 initialization. Φ serves as the temporal scaffold that enables phase-specific behavior."
    },
    topic: "PROBLEM_FORMULATION",
    section: "Section III-A",
    difficulty: "Medium",
    category: "implementation",
    pageReference: "Section III-A, Page 3, Equation 2",
  },

  {
    id: 6,
    question: "The AMP discriminator loss (Equation 5) uses which formulation, and why is the gradient penalty term included?",
    questionType: "single-choice",
    options: [
      "Binary cross-entropy loss; the gradient penalty prevents gradient explosion in the policy",
      "Least-squares GAN loss; the gradient penalty on reference data promotes training stability",
      "Wasserstein distance; gradient penalty enforces 1-Lipschitz constraint on the discriminator",
      "Hinge loss; gradient penalty prevents the discriminator from memorizing reference motions"
    ],
    answer: 1,
    explanation: {
      intuition: "The discriminator is like a motion judge who scores whether the robot moves like a human skater. The least-squares formulation is gentler than binary cross-entropy — it doesn't saturate, so gradients keep flowing even when the discriminator is very confident. The gradient penalty is like telling the judge: 'Don't become so extreme in your opinion that tiny motion changes cause wild reward swings.' It keeps training stable.",
      math: "From Section III-B (page 4), Equation 5:\narg max_ϕ E_{τ~M}[(D_ϕ(τ)−1)²] + E_{τ~P}[(D_ϕ(τ)+1)²] + (α_d/2)·E_{τ~M}[‖∇_ϕ D_ϕ(τ)‖²]\nThe first term pushes expert motion (from dataset M) toward +1, the second pushes policy rollouts (from P) toward −1. The gradient penalty αd/2·‖∇ϕDϕ‖² on reference data M is noted to 'promote training stability.' The style reward (Eq. 6): r^style = α·max(0, 1−(1/4)(d−1)²), where d = D_ϕ(τ_t).",
      computation: `import torch
import torch.nn as nn

# AMP Discriminator Loss
# Source: Section III-B, Equation 5 (page 4)

def amp_discriminator_loss(D_phi, expert_batch, agent_batch, alpha_d=10.0):
    """
    Least-squares GAN loss + gradient penalty on expert data.
    """
    # Expert motions should be classified as +1
    d_expert = D_phi(expert_batch)
    loss_expert = ((d_expert - 1) ** 2).mean()

    # Agent motions should be classified as -1
    d_agent = D_phi(agent_batch)
    loss_agent = ((d_agent + 1) ** 2).mean()

    # Gradient penalty on expert data (training stability)
    expert_batch.requires_grad_(True)
    d_expert_gp = D_phi(expert_batch)
    grad = torch.autograd.grad(d_expert_gp.sum(), expert_batch, create_graph=True)[0]
    grad_penalty = (alpha_d / 2) * (grad.norm(2) ** 2).mean()

    return loss_expert + loss_agent + grad_penalty

def compute_style_reward(d, alpha=1.0):
    # Source: Equation 6
    return alpha * torch.clamp(1.0 - 0.25 * (d - 1)**2, min=0.0)`,
      connection: "Table V (Appendix C, page 13) shows the AMP style reward has the highest weight (5.0) among all pushing phase rewards, reflecting its central importance. Table II ablation confirms: removing AMP (Gait-Based baseline) increases contact error (E_contact) from 0.001 to 0.130, showing AMP produces more coordinated foot-ground interactions."
    },
    topic: "AMP_PUSHING",
    section: "Section III-B",
    difficulty: "Hard",
    category: "theoretical",
    pageReference: "Section III-B, Page 4, Equations 5–7",
  },

  {
    id: 7,
    question: "The physics-guided tilt reference (Equation 10) computes γ_ref as: γ_ref = arcsin(L·Δψ / (v·Δt·tan λ)). Select ALL assumptions made in this derivation:",
    questionType: "multi-select",
    options: [
      "The skateboard follows a planar bicycle-model approximation (yaw only, no lateral slip)",
      "The yaw rate is assumed constant: ψ̇ ≈ Δψ/Δt over the steering horizon",
      "The board tilt angle and truck steering are related by Equation 1",
      "The robot's center of mass is always directly above the board center",
      "The forward velocity v is clipped to a minimum threshold for numerical stability"
    ],
    answers: [0, 1, 2, 4],
    explanation: {
      intuition: "Imagine steering a bicycle: lean left to go left. The lean angle needed depends on how fast you're going and how sharp the turn is. HUSKY does the same math — it figures out 'how much should the robot lean' from the desired heading change. The assumptions are: (1) treat the board like a simple bicycle, (2) assume we want a smooth constant-rate turn, (3) use our tilt-to-steer conversion from Equation 1. The velocity clipping is purely numerical — a near-zero speed makes the equation blow up.",
      math: "From Section III-C (page 4–5): (a) bicycle model: 'we adopt a bicycle-model approximation commonly used in vehicle dynamics [44], which captures the yaw motion of the skateboard while neglecting lateral slip and vertical dynamics' → Equation 8: ψ̇ = (v/L)·tan σ. (b) Constant yaw rate: 'we assume a constant yaw rate ψ̇ ≈ Δψ/Δt to achieve a smooth and gradual turn.' (c) Combining Eq.1 into Eq.8 → Eq.9: ψ̇ = (v/L)·tan λ·sin γ → Eq.10: γ_ref = arcsin(L·Δψ/(v·Δt·tan λ)). (d) Velocity clipping: 'we clip v to a minimum threshold' — no CMass assumption is stated.",
      computation: `import numpy as np

# Physics-guided tilt reference
# Source: Section III-C, Equations 8–10 (pages 4–5)

def compute_gamma_ref(psi_target, psi_board, v_board, L, lam,
                      delta_t=1.0, v_min=0.3, gamma_max=0.2):
    """
    Args:
        psi_target: desired heading [rad]
        psi_board:  current skateboard heading [rad]
        v_board:    current skateboard speed [m/s]
        L:          wheelbase [m]
        lam:        rake angle [rad]
        delta_t:    steering horizon [s]
    Returns:
        gamma_ref: desired board tilt [rad], clipped to feasible range
    """
    delta_psi = psi_target - psi_board  # heading error

    # Clip velocity for numerical stability
    v = max(v_board, v_min)

    # Equation 10
    arg = (L * delta_psi) / (v * delta_t * np.tan(lam))
    arg = np.clip(arg, -1.0, 1.0)  # arcsin domain
    gamma_ref = np.arcsin(arg)

    # Clip to feasible tilt range
    return np.clip(gamma_ref, -gamma_max, gamma_max)`,
      connection: "Figure 4(b) (page 7) validates this: 'Without tilt guidance, the achievable heading range is narrow. In contrast, incorporating tilt guidance produces smooth turning trajectories and enables the humanoid to reach a substantially wider range of headings.' The w/o Tilt Guidance ablation in Table II confirms a 12% relative increase in heading error (0.233 vs 0.208 rad)."
    },
    topic: "PHYSICS_GUIDED_STEERING",
    section: "Section III-C",
    difficulty: "Hard",
    category: "mathematical",
    pageReference: "Section III-C, Pages 4–5, Equations 8–10",
  },

  {
    id: 8,
    question: "In HUSKY's trajectory-guided phase transition mechanism, Bézier curves are used for __ and SLERP is used for __. Which statement correctly fills both blanks?",
    questionType: "single-choice",
    options: [
      "Bézier: body orientation interpolation; SLERP: Cartesian position planning",
      "Bézier: Cartesian position planning of key bodies; SLERP: quaternion orientation interpolation",
      "Bézier: reward shaping over the transition; SLERP: action-space interpolation",
      "Bézier: generating human-like pushing motions; SLERP: converting between Euler angles and quaternions"
    ],
    answer: 1,
    explanation: {
      intuition: "Imagine planning how to gracefully step onto a skateboard. Your feet need to trace a smooth curved path in 3D space (Bézier handles this — it's like drawing a smooth S-curve through waypoints). Meanwhile your body needs to rotate from 'pushing sideways stance' to 'both-feet-on-board steering stance'. Rotations live on a sphere mathematically, so SLERP (Spherical Linear intERPolation) ensures the shortest, smoothest rotation path.",
      math: "From Section III-D (page 5), Equation 12 (Bézier): p^K(t) = Σ_{i=0}^n C(n,i)·(1−s)^{n-i}·s^i·p^K_i, where s=(t−t0)/(tf−t0), p^K_0 = p^K_end, p^K_n = p^K_ref. Equation 13 (SLERP): q^K(t) = [sin((1−s)Ω)/sinΩ]·q^K_end + [sin(sΩ)/sinΩ]·q^K_ref, where Ω = arccos(⟨q^K_end, q^K_ref⟩). The paper explicitly states: 'For body translations... Bézier curve' and 'Boy orientations are interpolated using spherical linear interpolation (slerp) between quaternions.'",
      computation: `import numpy as np

# Trajectory Planning for Phase Transition
# Source: Section III-D, Equations 12–13 (page 5)

def bezier_position(p_end, p_ref, s, control_pts=None):
    """Bézier curve for Cartesian key-body positions (Eq. 12)"""
    if control_pts is None:
        # Linear (n=1) for simplicity
        return (1 - s) * p_end + s * p_ref
    # General n-th order: use de Casteljau's algorithm
    pts = [p_end] + list(control_pts) + [p_ref]
    n = len(pts) - 1
    while len(pts) > 1:
        pts = [(1-s)*pts[i] + s*pts[i+1] for i in range(len(pts)-1)]
    return pts[0]

def slerp_orientation(q_end, q_ref, s):
    """SLERP for quaternion orientations (Eq. 13)"""
    dot = np.clip(np.dot(q_end, q_ref), -1.0, 1.0)
    Omega = np.arccos(dot)
    if np.abs(Omega) < 1e-6:
        return q_end  # nearly identical
    return (np.sin((1-s)*Omega)/np.sin(Omega)) * q_end + \
           (np.sin(s*Omega)/np.sin(Omega)) * q_ref`,
      connection: "Figure 6 (page 7) shows representative transition trajectories, described as 'smooth, coordinated whole-body motions with seamless transitions.' The Translation-only ablation in Table II (E_contact = 0.038 vs 0.001) shows that omitting SLERP-based orientation guidance causes poor boarding posture."
    },
    topic: "TRAJECTORY_TRANSITION",
    section: "Section III-D",
    difficulty: "Medium",
    category: "implementation",
    pageReference: "Section III-D, Page 5, Equations 12–13",
  },

  {
    id: 9,
    question: "Arrange the correct sequence of steps in HUSKY's skateboard physical identification procedure:",
    questionType: "ordering",
    options: [
      "Perturb the skateboard in roll and release it — observe free-decay oscillation",
      "Compute logarithmic decrement δ and damping ratio ζ from successive peak amplitudes",
      "Calculate undamped natural frequency ωn from damped frequency and ζ",
      "Estimate torsional stiffness k = Iωn² using rigid-body roll inertia approximation",
      "Compute damping coefficient d = 2ζ√(kI) and embed (k, d) in simulation"
    ],
    answer: [0, 1, 2, 3, 4],
    explanation: {
      intuition: "Think of the skateboard trucks as a spring-and-damper system — tilt the board, and the bushing springs try to push it back while friction slows the oscillation. To model this in simulation, you perform a simple experiment: tilt the board, let go, and film how it wiggles back and forth. From two successive peaks, you can extract how 'springy' and how 'dampy' the system is — then embed these numbers into MuJoCo.",
      math: "From Section III-E.1 (page 5), Equations 14–15. Step 1: Free-decay roll response. Step 2: δ = ln(ϕ(t)/ϕ(t+T)), ζ = δ/√(4π²+δ²) (Eq. 14). Step 3: ωn = ωd/√(1−ζ²) where ωd = 2π/T (Eq. 15). Step 4: k = Iωn² (text after Eq. 15). Step 5: d = 2ζ√(kI). Appendix G (page 14) gives concrete values: Board 1: T=0.107s, peaks (0.614, 0.0108), Kp=34.835, Kd=0.540.",
      computation: `import numpy as np

# Skateboard System Identification
# Source: Section III-E.1, Equations 14–15 (page 5); Appendix G (page 14)

def identify_skateboard(theta1, theta2, T, I_board):
    """
    Args:
        theta1, theta2: successive roll peak amplitudes [rad]
        T: observed oscillation period [s]
        I_board: roll moment of inertia (rectangular prism) [kg·m²]
    Returns:
        k: torsional stiffness [Nm/rad]
        d: damping coefficient [Nm·s/rad]
    """
    # Step 2: Logarithmic decrement and damping ratio (Eq. 14)
    delta = np.log(theta1 / theta2)
    zeta = delta / np.sqrt(4 * np.pi**2 + delta**2)

    # Step 3: Natural frequency (Eq. 15)
    omega_d = 2 * np.pi / T
    omega_n = omega_d / np.sqrt(1 - zeta**2)

    # Step 4: Torsional stiffness
    k = I_board * omega_n**2

    # Step 5: Damping coefficient
    d = 2 * zeta * np.sqrt(k * I_board)
    return k, d

# Appendix G values for Board 1:
k1, d1 = identify_skateboard(0.614, 0.0108, 0.107, 7.15e-3)
print(f"Board 1: k={k1:.3f} Nm/rad, d={d1:.3f} Nm·s/rad")
# Expected: Kp ≈ 34.835, Kd ≈ 0.540`,
      connection: "Figure 8 (page 8) validates the critical importance of accurate identification: cross-applying parameters (compliant→stiff and stiff→compliant) causes mounting failure and steering instability, respectively. This confirms the five-step procedure must be performed per board."
    },
    topic: "SIM_TO_REAL",
    section: "Section III-E.1",
    difficulty: "Hard",
    category: "implementation",
    pageReference: "Section III-E.1, Page 5, Equations 14–15; Appendix G, Page 14",
  },

  {
    id: 10,
    question: "The overall reward function (Equation 4) uses binary phase indicators. Match each reward component to its phase indicator:",
    questionType: "matching",
    pairs: [
      { item: "AMP style reward + velocity tracking", match: "𝕀^push (pushing phase)" },
      { item: "Board tilt tracking + heading reward", match: "𝕀^steer (steering phase)" },
      { item: "Key-body position and orientation tracking", match: "𝕀^trans (transition phase)" },
      { item: "Joint torque penalties + action smoothness", match: "r^reg (always active)" }
    ],
    explanation: {
      intuition: "The reward is like three separate coaches who each yell instructions only at their designated part of the routine — the pushing coach only speaks during push-offs, the steering coach only during gliding turns, and the transition manager only during foot-mounting. But the safety officer (regularization) never shuts up — they're always penalizing dangerous joint movements and jerky actions!",
      math: "From Section III-A (page 3), Equation 4: r_t = 𝕀^push·r_t^push + 𝕀^steer·r_t^steer + 𝕀^trans·r_t^trans + r_t^reg. Table V (Appendix C, page 13) lists full reward terms. Pushing includes: velocity tracking (3.0), yaw alignment (1.0), feet air time (3.0), ankle parallel (0.5), AMP style (5.0). Steering includes: steer feet contact (3.0), joint deviation (1.5), heading (5.0), board tilt (4.0), feet markers (1.0). Transition includes: keybody position (10.0) and orientation (10.0). Regularization includes: joint limits (−5.0), velocity (−1e-3), torque (−1e-6), etc.",
      computation: `# Reward structure from Table V (Appendix C)
# Source: Equation 4 (page 3), Table V (page 13)

class HUSKYReward:
    def compute(self, state, phase):
        r_reg = self.regularization(state)  # always active

        if phase == "push":
            return self.pushing_reward(state) + r_reg
        elif phase == "steer":
            return self.steering_reward(state) + r_reg
        elif phase == "trans":
            return self.transition_reward(state) + r_reg

    def transition_reward(self, state):
        # Highest weight components in the whole system!
        r_pos = 10.0 * self.keybody_pos_tracking(state)
        r_rot = 10.0 * self.keybody_rot_tracking(state)
        return r_pos + r_rot  # total weight up to 20`,
      connection: "The transition reward has the highest per-component weight (10.0 each × 2 = 20) in the entire system — confirming the paper's claim that 'Trajectory guidance is essential for phase transitions' (Section IV-B). The AMP style reward (weight 5.0) is the second-highest, reflecting the importance of human-like motion quality."
    },
    topic: "REWARD_DESIGN",
    section: "Section III-A",
    difficulty: "Medium",
    category: "implementation",
    pageReference: "Section III-A, Page 3, Equation 4; Appendix C, Page 13, Table V",
  },

  // ============================================================
  // EXPERIMENTS — 30%
  // ============================================================
  {
    id: 11,
    question: "According to Table II (simulation results), which baseline achieves the worst (highest) contact error E_contact, and what does this reveal about that design choice?",
    questionType: "single-choice",
    options: [
      "HUSKY-Tracking-Based (E_contact = 0.015); strict reference tracking causes foot slip",
      "HUSKY-Gait-Based (E_contact = 0.130); lacking human motion priors leads to poor phase coordination",
      "HUSKY-AMP-Transition (E_contact = 0.394); relying on style rewards alone for transitions fails to guide phase switching",
      "HUSKY-w/o-Tilt-Guidance (E_contact = 0.002); no tilt reference means the feet cannot find the board"
    ],
    answer: 2,
    explanation: {
      intuition: "The AMP Transition baseline tries to guide mounting/dismounting using style rewards — basically saying 'copy this reference motion of a human stepping onto the board.' But style rewards are soft guidance; they don't enforce the specific foot-board contact patterns required. It's like trying to learn to skateboard by watching YouTube videos without anyone actually correcting your foot placement.",
      math: "From Table II (Section IV, page 7): HUSKY-AMP Transition achieves E_contact = 0.394 ± 0.015 — the highest by far. Section IV-B (page 6) explains: 'the AMP Transition baseline, which relies solely on full reference motions, achieves moderate success rates but incurs significant contact errors, as the robot fails to explore proper phase switching.' For comparison: HUSKY (ours) = 0.001, HUSKY-Gait-Based = 0.130, HUSKY-Tracking-Based = 0.015.",
      computation: `# Table II: Contact Error Comparison
# Source: Table II, Section IV-B (pages 6–7)

results = {
    "HUSKY (ours)":             {"Esucc": 100.00, "Econtact": 0.001},
    "HUSKY-Tracking-Based":     {"Esucc": 11.12,  "Econtact": 0.015},
    "HUSKY-Gait-Based":         {"Esucc": 82.38,  "Econtact": 0.130},
    "HUSKY-w/o-Tilt Guidance":  {"Esucc": 96.72,  "Econtact": 0.002},
    "HUSKY-AMP Transition":     {"Esucc": 85.12,  "Econtact": 0.394},  # ← worst
    "HUSKY-Translation-only":   {"Esucc": 89.55,  "Econtact": 0.038},
}

# E_contact = per-step violation of foot-board contact pattern
# (single-foot during pushing, double-foot during steering)
worst = max(results, key=lambda k: results[k]["Econtact"])
print(f"Worst contact error: {worst} → {results[worst]['Econtact']:.3f}")`,
      connection: "Figure 5 (page 7) provides further evidence: without transition guidance, 'the steering contact reward remains low, indicating persistent incorrect foot–board contact patterns' even when using 50/50 phase initialization. This validates the necessity of explicit trajectory-guided transitions."
    },
    topic: "EXPERIMENTAL_RESULTS",
    section: "Section IV-B",
    difficulty: "Easy",
    category: "experimental",
    pageReference: "Section IV-B, Pages 6–7; Table II, Page 7",
  },

  {
    id: 12,
    question: "Select ALL evaluation metrics used in HUSKY's simulation experiments, and identify the metric that specifically measures contact pattern correctness:",
    questionType: "multi-select",
    options: [
      "E_succ: Task success rate — percentage of episodes completed without termination",
      "E_vel: Velocity tracking error — mean absolute error of v_cmd vs v_board during pushing",
      "E_yaw: Heading tracking error — heading error during steering phase",
      "E_smth: Motion smoothness — aggregated joint angle variations across consecutive control steps",
      "E_contact: Contact error — per-step violation of foot–board contact pattern"
    ],
    answers: [0, 1, 2, 3, 4],
    explanation: {
      intuition: "The five metrics cover the full skateboarding performance picture: safety (did it fall?), propulsion accuracy (how well did it hit the speed target?), navigation accuracy (did it turn correctly?), naturalness (were the motions smooth?), and contact correctness (did the feet land on the board at the right times?). Each metric targets a different aspect of the multi-phase task.",
      math: "From Section IV-A (page 5–6): All five metrics are explicitly defined. E_contact is specifically described as: 'Defined as the per-step violation of the foot–board contact pattern, i.e., single-foot contact during pushing and double-foot contact during steering.' This directly measures phase-specific contact compliance. Note: no separate metric for sim-to-real transfer quality is reported — this is assessed qualitatively through Figure 8.",
      computation: `# Metric computation pseudocode
# Source: Section IV-A.1, Pages 5–6

def compute_metrics(episodes):
    n = len(episodes)
    Esucc  = sum(not ep.terminated for ep in episodes) / n * 100
    Evel   = mean(|ep.vcmd - ep.vboard| for ep in episodes if ep.in_push_phase)
    Eyaw   = mean(|ep.psi_target - ep.psi_board| for ep in episodes if ep.in_steer_phase)
    Esmth  = mean(sum(|theta_t - theta_{t-1}| for t) for ep in episodes)
    # E_contact: phase-specific contact pattern violation
    def contact_violation(ep):
        violations = []
        for t, phase in zip(ep.timesteps, ep.phases):
            if phase == "push":
                # Expect single foot on board
                violations.append(not ep.is_single_foot_on_board(t))
            elif phase == "steer":
                # Expect both feet on board
                violations.append(not ep.is_both_feet_on_board(t))
        return mean(violations)
    Econtact = mean(contact_violation(ep) for ep in episodes)
    return Esucc, Evel, Eyaw, Esmth, Econtact`,
      connection: "All five metrics are reported in Table II (page 7). HUSKY achieves perfect Esucc (100%) and near-zero Econtact (0.001), while all baselines show degradation in at least one metric, validating the necessity of each framework component."
    },
    topic: "EXPERIMENTAL_SETUP",
    section: "Section IV-A",
    difficulty: "Easy",
    category: "experimental",
    pageReference: "Section IV-A.1, Pages 5–6",
  },

  {
    id: 13,
    question: "Figure 4(a) validates the skateboard equality constraint. What happens when the constraint tan σ = tan λ sin γ is omitted from the simulator?",
    questionType: "single-choice",
    options: [
      "The board steers in the opposite direction, causing the robot to follow a mirror-image trajectory",
      "The simulator crashes due to unsatisfied kinematic constraints",
      "Board tilting no longer induces truck steering, leaving the board able only to glide straight forward",
      "The robot's feet slip off the board due to incorrect friction modeling"
    ],
    answer: 2,
    explanation: {
      intuition: "If you disconnect the linkage between a skateboard's deck tilt and its wheel-steering mechanism, the trucks become rigid and the board can only go straight — like a sled. No matter how much the robot leans, the wheels stay pointed forward. HUSKY's steering relies entirely on exploiting this mechanical coupling, so removing it completely eliminates turning capability.",
      math: "From Section IV-C, Skateboard Modeling analysis (page 6): 'Similar to prior simplified models [20], omitting the equality constraint in Eq.(1) prevents board tilting from inducing truck steering, leaving the skateboard able only to glide straight forward with negligible turning capability.' This is visualized in Figure 4(a) (page 7) where the 'w/o modeling' trajectories go straight while HUSKY's follow curved paths.",
      computation: `# What happens without the constraint
# Source: Section IV-C, Figure 4(a) (page 6–7)

def simulate_step_without_constraint(gamma, lambda_angle):
    """
    Without Equation 1: truck angle is NOT updated by tilt
    """
    sigma = 0.0  # trucks remain centered regardless of gamma!
    # Result: yaw_rate = (v/L) * tan(sigma) = 0
    # Board only goes straight
    return sigma

def simulate_step_with_constraint(gamma, lambda_angle):
    """
    With Equation 1: tilt drives steering
    """
    sigma = np.arctan(np.tan(lambda_angle) * np.sin(gamma))
    # Result: yaw_rate = (v/L) * tan(sigma) != 0
    return sigma

# For gamma = 15° lean, lambda = 65°:
import numpy as np
print(f"Without: sigma = {simulate_step_without_constraint(0.26, 1.13):.3f} rad")
print(f"With:    sigma = {simulate_step_with_constraint(0.26, 1.13):.3f} rad")`,
      connection: "This ablation directly confirms that the physics-informed modeling in Section II-A is not just a mathematical exercise — it is foundational to the entire steering capability. Without it, the heading tracking error E_yaw would be unbounded as the board cannot respond to lean."
    },
    topic: "EXPERIMENTAL_ANALYSIS",
    section: "Section IV-C",
    difficulty: "Medium",
    category: "experimental",
    pageReference: "Section IV-C, Pages 6–7; Figure 4, Page 7",
  },

  {
    id: 14,
    question: "Domain randomization (Table I) varies robot and skateboard parameters. Arrange these DR parameters in order of their range magnitude (smallest to largest physical scale):",
    questionType: "ordering",
    options: [
      "Default Joint Position: U(−0.01, 0.01) rad",
      "Robot/Skateboard Center of Mass offset: U(−2.5, 2.5) cm",
      "Default Root Position: U(−2.0, 2.0) cm",
      "Push Robot Base velocity: U(−0.5, 0.5) m/s"
    ],
    answer: [0, 2, 1, 3],
    explanation: {
      intuition: "Domain randomization is like training with noisy equipment — sometimes the skateboard is slightly heavier, sometimes the robot joints start in slightly different positions. This makes the policy robust to real-world imperfections. The ranges tell you how much each parameter varies: joint positions vary by ±0.01 rad (tiny), positions by ±2cm (small), and external push velocities by ±0.5 m/s (large).",
      math: "From Table I (Section IV, page 6), all DR parameters: Robot CoM = U(2.5, 2.5) cm, Skateboard CoM = U(2.5, 2.5) cm, Default Root Position = U(2.0, 2.0) cm, Default Joint Position = U(0.01, 0.01) rad, Push Robot Base = U(−0.5, 0.5) m/s, Robot Body Friction = U(0.3, 1.6), Skateboard Deck Friction = U(0.8, 2.0). Converting to common units: 0.01 rad < 2.0 cm < 2.5 cm < 0.5 m/s (as velocities). Note: friction ranges are dimensionless and their 'magnitude' is harder to compare — the ordering uses physical displacement/velocity scale.",
      computation: `# Domain Randomization configuration
# Source: Table I, Section IV (page 6)

import numpy as np

domain_randomization = {
    "robot_com_offset_cm":    (-2.5, 2.5),   # Center of mass
    "board_com_offset_cm":    (-2.5, 2.5),   # Board CoM
    "root_position_cm":       (-2.0, 2.0),   # Initial root
    "joint_position_rad":     (-0.01, 0.01), # Joint init
    "push_velocity_m_s":      (-0.5, 0.5),   # Perturbation
    "robot_body_friction":    (0.3, 1.6),    # Friction
    "board_deck_friction":    (0.8, 2.0),    # Friction
}

def apply_dr(nominal_params):
    randomized = {}
    for key, (lo, hi) in domain_randomization.items():
        randomized[key] = nominal_params[key] + np.random.uniform(lo, hi)
    return randomized`,
      connection: "Section IV-D.2 (page 8) notes that DR was specifically increased for friction to 'improve sim-to-real transfer of foot-mounting and body-alignment behaviors' — addressing the high friction of real skateboard decks that differs from simulation defaults."
    },
    topic: "SIM_TO_REAL",
    section: "Section III-E.2",
    difficulty: "Medium",
    category: "implementation",
    pageReference: "Section III-E.2; Table I, Page 6",
  },

  {
    id: 15,
    question: "Figure 5 shows training curves for episode length and steering contact reward. What does the persistent low steering contact reward (without transition guidance) indicate, even when mixed-phase initialization is used?",
    questionType: "single-choice",
    options: [
      "The RL agent converges faster with mixed initialization, but the reward scale is lower",
      "Without explicit transition guidance, the policy collapses to pushing-only behavior and cannot explore the steering contact pattern",
      "The steering contact reward is defined incorrectly — it should include ground contact bonuses",
      "Mixed initialization causes the robot to fall more often, reducing total reward"
    ],
    answer: 1,
    explanation: {
      intuition: "Imagine learning to do a handstand by randomly starting halfway up sometimes. Even with these helpful starting positions, if nobody guides you through the tricky middle part, you'll just keep falling and never learn the full motion. The RL agent is the same — even with 50/50 phase initialization, it never learns to transition because there's no reward signal to guide it through the foot-mounting sequence.",
      math: "From Section IV-C, Phase Exploration (page 6–7): 'In both settings, episode length increases rapidly during early training, yet the steering contact reward remains low, indicating persistent incorrect foot–board contact patterns. This suggests that policies trained without transition guidance fail to learn phase transitions, collapsing to a trivial pushing-only behavior.' Figure 5 (page 7) shows 'Ours' rapidly achieves high steering contact reward, while 'w/o Transition Guidance' and 'Mixed Initialization' both plateau near zero.",
      computation: `# Training dynamics analysis
# Source: Section IV-C, Figure 5 (page 7)

# Observed pattern from Figure 5:
# - ALL three curves: episode length increases rapidly (policy learns to stay alive)
# - ONLY "Ours": steering contact reward increases after ~4000 steps
# - "w/o Transition" + "Mixed Init": steering contact reward ≈ 0 throughout

# This is a local optima problem in RL:
# Without transition guidance, pushing (one foot ground) is rewarded
# The policy optimizes: "keep pushing, never step on the board"
# Result: pushes indefinitely, never explores steering contact pattern

# Transition reward (weight 10.0 + 10.0 = 20.0) breaks this by
# providing explicit gradient toward foot-mounting behaviors`,
      connection: "Reference [49] (WoCoCo, page 9) is cited: 'Policies trained without explicit transition guidance often fail to explore new contact phases, which can lead to convergence to local optima.' HUSKY's trajectory guidance directly addresses this known problem in sequential contact learning."
    },
    topic: "PHASE_EXPLORATION",
    section: "Section IV-C",
    difficulty: "Medium",
    category: "experimental",
    pageReference: "Section IV-C, Pages 6–7; Figure 5, Page 7",
  },

  // ============================================================
  // DISCUSSION & LIMITATIONS — 15%
  // ============================================================
  {
    id: 16,
    question: "Select ALL limitations that are explicitly stated in HUSKY's Conclusion/Future Work section:",
    questionType: "multi-select",
    options: [
      "Limited camera field of view prevents reliable observation of board and wheel–ground interactions",
      "The framework has only been tested on flat indoor surfaces and cannot handle stairs",
      "Current experiments are on relatively simple terrains; human skateboarders perform in complex skateparks",
      "The AMP discriminator requires large motion capture datasets that are expensive to collect",
      "Extending to complex environments will require richer motion priors and terrain-adaptive control"
    ],
    answers: [0, 2, 4],
    explanation: {
      intuition: "The paper honestly identifies two main gaps: (1) the robot is essentially blind to the board and wheel-ground contact during real-world use — it cannot see its own feet; (2) it has only been tested on simple flat ground, not actual skatepark obstacles. The paper proposes solutions for both but doesn't pursue them. The dataset cost and stair limitations are not mentioned.",
      math: "From Section VI, Conclusion and Future Work (page 9): 'Onboard Vision. The limited camera field of view prevents reliable observation of the board and wheel–ground interactions, restricting perception-driven feedback in the control loop.' And: 'Complex Terrains. Current experiments are conducted on relatively simple terrains, whereas human skateboarders routinely perform in complex environments such as skateparks while executing acrobatic maneuvers. Extending our framework to such scenarios will require richer motion priors and terrain-adaptive control strategies.' Stairs and dataset cost are NOT mentioned.",
      computation: `# Limitations summary
# Source: Section VI, Page 9

limitations = {
    "STATED": [
        "Vision: Limited FOV prevents board/wheel-ground perception",
        "Terrain: Only flat surfaces; no complex terrain adaptation"
    ],
    "UNSTATED_BUT_NOTABLE": [
        "No quantitative sim-to-real gap measurement (only qualitative Fig. 8)",
        "Generalization tested on only 2 board types (Fig. 12, Appendix H)",
        "Human motion dataset M not described (size, source, collection method)",
        "No energy efficiency metrics",
        "No formal stability proof for the hybrid dynamical system",
        "SLERP for orientations noted as 'Boy orientations' — likely typo in paper"
    ]
}`,
      connection: "The two stated limitations directly motivate future directions: visual state estimation for board tracking, and terrain-adaptive motion priors. Figure 12 (page 13) shows generalization to 'a variety of skateboard platforms' — but notably all appear to be on flat ground, supporting the terrain limitation claim."
    },
    topic: "LIMITATIONS",
    section: "Section VI",
    difficulty: "Easy",
    category: "critical",
    pageReference: "Section VI, Page 9",
  },

  {
    id: 17,
    question: "Critical analysis: The paper reports 100% success rate for HUSKY in simulation across 1,000 episodes. Which methodological concern is most valid?",
    questionType: "single-choice",
    options: [
      "1,000 episodes is too few — the standard in robotics RL is at least 10,000 episodes",
      "Success rate alone is insufficient because the termination criterion and maximum episode length are not clearly defined in the main paper",
      "The 100% result is suspicious and likely the result of reward hacking in the AMP discriminator",
      "The results are invalid because the evaluation uses the same random seeds as training"
    ],
    answer: 1,
    explanation: {
      intuition: "A 100% success rate sounds great — but 'success' is only meaningful if we know what 'failure' looks like. If episodes are very short or the termination condition is too easy, 100% is trivial to achieve. The paper doesn't clearly define in the main text what exactly causes episode termination — this information is buried or missing, making it hard to assess how challenging the bar actually is.",
      math: "The paper states (Section IV-A, page 5): 'E_succ: Defined as the percentage of episodes completed without termination.' But the termination conditions themselves are not enumerated in the main paper. From training details (Appendix D, page 12): 'each episode lasting 20 seconds.' The joint limits penalty (Table V: −5.0 for out-of-range joints) and collision penalty (−10.0) presumably trigger termination, but this is inferred from reward design, not stated explicitly. For a rigorous reproducibility assessment: the exact termination conditions and whether they match human-level challenge should be specified.",
      computation: `# Reproducibility assessment
# Source: Sections IV-A, IV-D; Appendix D

reproducibility = {
    "Clearly specified": [
        "1,000 rollout episodes per experiment",
        "5 random seeds (results as mean ± std)",
        "Policy deployed at 50 Hz on real robot",
        "500 Hz PD controller for joint tracking",
        "Episode length: 20 seconds",
        "4,096 parallel environments",
        "PPO with full hyperparameters in Table VI",
    ],
    "Missing or unclear": [
        "Exact termination conditions not listed in main paper",
        "Human motion dataset M: source, size, capture protocol not described",
        "Skateboard rake angle λ value not given in main paper",
        "Steering horizon Δt value not specified",
        "σ_γ tolerance in tilt reward not specified",
        "Whether 'ours' evaluation environments differ from training",
    ]
}`,
      connection: "The paper's reproducibility is partially good: PPO hyperparameters are fully provided in Table VI, hardware specs are clear (RTX 5080, ~20 hours training). However, key physics constants (λ, Δt, σ_γ) and motion dataset information are absent from the main paper, limiting independent reproduction."
    },
    topic: "CRITICAL_ANALYSIS",
    section: "Section IV",
    difficulty: "Hard",
    category: "critical",
    pageReference: "Section IV-A, Page 5; Appendix D, Page 12",
  },

  {
    id: 18,
    question: "The asymmetric actor–critic framework uses different observations for actor and critic. Why is privileged information given only to the critic during training?",
    questionType: "single-choice",
    options: [
      "The critic has more compute available, so it can process larger state vectors",
      "The actor uses privileged info during deployment and critic is cheaper to compute",
      "The critic guides policy learning with fuller state knowledge in simulation, while the actor must rely only on deployable proprioceptive sensors — enabling zero-shot real-world transfer",
      "Privileged information contains ground-truth physics that cannot be expressed as a neural network input"
    ],
    answer: 2,
    explanation: {
      intuition: "During training, the critic is like a coach watching the game on TV with full replays and statistics — it can see the board's exact velocity, tilt angle, and all contact forces. The actor (robot's brain) only feels what its body sensors report, just like in the real world. By training this way, the actor learns to make great decisions with limited sensing, because the critic was there to tell it if those decisions were good during training.",
      math: "From Section III-A (page 3): 'We utilize an asymmetric actor–critic framework for policy training, where the actor observes only proprioceptive information, while the critic has access to additional privileged information.' Actor: o_t^prop ∈ ℝ^78 (Equation 2). Critic: o_t^priv (Equation 3) = [v_t, p_t^b, r_t^b, v_t^b, ω_t^b, θ_t^b, f_t^g, f_t^b] — includes skateboard state and contact forces not measurable by onboard sensors.",
      computation: `# Asymmetric Actor-Critic
# Source: Section III-A, Equations 2–3 (page 3)

class ActorCritic:
    def __init__(self):
        # Actor: small MLP on deployable proprioception only
        self.actor = MLP([512, 256, 128], input_dim=78 * 5)  # 5-step history

        # Critic: larger MLP with privileged simulation data
        priv_dim = 3 + 3 + 6 + 3 + 3 + 3 + 6 + 6  # Eq. 3 dimensions
        self.critic = MLP([512, 256, 128], input_dim=78*5 + priv_dim)

    def act(self, obs_prop_history):
        """Deploy on real robot — only uses proprioception"""
        return self.actor(obs_prop_history.flatten())

    def evaluate(self, obs_prop_history, obs_priv):
        """Training only — critic sees everything"""
        full_obs = torch.cat([obs_prop_history.flatten(), obs_priv])
        return self.critic(full_obs)`,
      connection: "This design is validated by real-world deployment (Section IV-D): the actor policy is transferred zero-shot to the Unitree G1 at 50 Hz, using only proprioceptive sensors available on the physical robot. The critic never runs on the real robot."
    },
    topic: "POLICY_ARCHITECTURE",
    section: "Section III-A",
    difficulty: "Medium",
    category: "implementation",
    pageReference: "Section III-A, Page 3, Equations 2–3",
  },

  {
    id: 19,
    question: "Arrange the four key steps of HUSKY's complete training pipeline from first to last:",
    questionType: "ordering",
    options: [
      "Deploy actor policy on Unitree G1 at 50Hz — zero-shot sim-to-real transfer",
      "Model the humanoid-skateboard system: derive tan σ = tan λ sin γ, identify spring-damper parameters (k, d)",
      "Train policy using PPO across 4,096 parallel environments with phase-specific rewards + AMP discriminator",
      "Design domain randomization and transition trajectory planning mechanism"
    ],
    answer: [1, 3, 2, 0],
    explanation: {
      intuition: "HUSKY follows a clear engineering sequence: first understand the physics (model the board), then prepare the training environment (design rewards and transitions), then actually train, then transfer to reality. It's like building a skateboard, setting up the training course, practicing for weeks, then going to the skate park.",
      math: "From the paper structure: Section II (pages 2–3) covers system modeling. Section III-A/B/C/D (pages 3–5) covers problem formulation, AMP design, steering strategy, and trajectory planning — all design choices before training. Section IV-A (pages 5–6) describes the training setup: 'All trainings... implemented in mjlab... PPO algorithm across 4,096 parallel environments... each episode lasting 20 seconds... roughly 20 hours in total' (Appendix D). Section IV-D covers real-world deployment at 50 Hz.",
      computation: `# HUSKY training pipeline
# Source: Sections II–IV (pages 2–8); Appendix D (page 12)

# Step 1: Physics modeling (Section II)
lambda_angle = identify_rake_angle(skateboard)  # fixed geometry
k, d = identify_skateboard(oscillation_data)    # Eq. 14–15

# Step 2: Training setup (Section III + DR)
rewards = define_phase_rewards()               # Table V
transitions = define_bezier_slerp_plans()      # Eq. 12–13
dr_config = configure_domain_randomization()   # Table I

# Step 3: PPO training (Section IV-A, Appendix D)
# Hardware: NVIDIA RTX 5080, ~20 hours
policy = train_ppo(
    envs=4096, episode_len=20, algorithm="PPO",
    rewards=rewards, dr_config=dr_config
)

# Step 4: Real-world deployment (Section IV-D)
deploy(policy, robot="Unitree G1",
       control_rate=50,   # Hz
       joint_control_rate=500)  # Hz PD controller`,
      connection: "Appendix D (page 12) confirms training details: 6-second skateboarding cycle (40% pushing, 10% mount, 45% steering, 5% dismount), RTX 5080 server, ~2–3 seconds per iteration, ~20 hours total training. The four-stage pipeline is the key distinguishing structure of HUSKY versus end-to-end RL approaches."
    },
    topic: "TRAINING_PIPELINE",
    section: "Sections II–IV",
    difficulty: "Medium",
    category: "implementation",
    pageReference: "Sections II–IV, Pages 2–8; Appendix D, Page 12",
  },

  {
    id: 20,
    question: "What does the system identification experiment in Figure 8 reveal about the sensitivity of HUSKY's sim-to-real transfer to skateboard parameters?",
    questionType: "single-choice",
    options: [
      "Parameter mismatch has minimal effect — the policy is robust due to domain randomization",
      "Only the stiffness parameter k matters; the damping d can be approximated without measurement",
      "Applying mismatched parameters in either direction causes critical failures: mounting failure (compliant→stiff) or over-leaning/instability (stiff→compliant)",
      "The identified parameters are interchangeable across all skateboard platforms tested"
    ],
    answer: 2,
    explanation: {
      intuition: "The experiment is like swapping car suspension settings between a sports car and a pickup truck. If you put the stiff sports-car settings in the bouncy truck, it will wallow on corners. If you put the soft truck settings in the sports car, it will handle terribly. For HUSKY, the 'stiff compliant mismatch' is even more dramatic because the whole mounting strategy depends on whether the board tilts when you step on it.",
      math: "From Section IV-D.3 (page 8): Case (a) — using compliant parameters on stiff board: 'the robot fails to mount. In simulation, the board tilts under the robot's stepping motion, allowing the policy to exploit this compliance for mounting, whereas the real stiff board remains nearly flat, breaking this assumption and preventing successful mounting.' Case (b) — using stiff parameters on compliant board: 'causes excessive leaning and loss of stability during steering, since the policy is not trained for such compliant dynamics.' The identified parameters are specific: Board 1: Kp=34.835, Kd=0.540; Board 2: Kp=14.677, Kd=0.402 (Appendix G, page 14).",
      computation: `# System identification sensitivity analysis
# Source: Section IV-D.3 (page 8); Appendix G (page 14)

boards = {
    "stiff_black":   {"Kp": 34.835, "Kd": 0.540, "T": 0.107},
    "compliant_pink": {"Kp": 14.677, "Kd": 0.402, "T": 0.185},
}

# Correct deployment (matched parameters):
# stiff_black  + stiff_params   → success ✓
# compliant_pink + compliant_params → success ✓

# Cross-application (mismatched):
# stiff_black  + compliant_params → FAILS (board doesn't tilt on step)
# compliant_pink + stiff_params   → FAILS (over-leans during steering)

# Key insight: DR alone (Table I) is insufficient to cover
# the gap between Kp=34 vs Kp=14 (~2.4× difference in stiffness)`,
      connection: "This finding reveals an important limitation: domain randomization was NOT sufficient to bridge the gap between two commercially available skateboards with ~2.4× stiffness difference. Per-board system identification is necessary, which increases deployment overhead. This is an implicit limitation not discussed in Section VI."
    },
    topic: "SIM_TO_REAL",
    section: "Section IV-D.3",
    difficulty: "Hard",
    category: "critical",
    pageReference: "Section IV-D.3, Page 8; Figure 8, Page 8; Appendix G, Page 14",
  }
];

// ─────────────────────────────────────────────────────────────
// TIMER HOOK
// ─────────────────────────────────────────────────────────────
const useTimer = () => {
  const [timeSpent, setTimeSpent] = useState(0);
  const [isActive, setIsActive] = useState(false);

  useEffect(() => {
    let interval = null;
    if (isActive) {
      interval = setInterval(() => setTimeSpent(t => t + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [isActive]);

  return {
    timeSpent,
    start: () => setIsActive(true),
    pause: () => setIsActive(false),
    reset: () => { setTimeSpent(0); setIsActive(false); }
  };
};

const formatTime = (s) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;

// ─────────────────────────────────────────────────────────────
// MAIN QUIZ COMPONENT
// ─────────────────────────────────────────────────────────────
const Quiz = () => {
  const [screen, setScreen] = useState('welcome');
  const [qIdx, setQIdx] = useState(0);
  const [answers, setAnswers] = useState(Array(quizData.length).fill(null));
  const [selected, setSelected] = useState([]);
  const [showExp, setShowExp] = useState(false);
  const [reviewMode, setReviewMode] = useState(false);
  const [draggedItem, setDraggedItem] = useState(null);
  const [shuffledMatches, setShuffledMatches] = useState([]);
  const { timeSpent, start, pause, reset: resetTimer } = useTimer();

  const q = quizData[qIdx];

  useEffect(() => {
    if (screen === 'quiz' && !showExp && !reviewMode) start();
    else pause();
  }, [screen, showExp, reviewMode, qIdx]);

  useEffect(() => {
    if (screen === 'quiz' && q?.questionType === 'matching' && !showExp && !reviewMode) {
      setShuffledMatches([...q.pairs.map((p, i) => ({ text: p.match, idx: i }))].sort(() => Math.random() - 0.5));
    }
  }, [qIdx, screen, showExp, reviewMode]);

  useEffect(() => {
    if (screen === 'quiz' && q?.questionType === 'ordering' && selected.length === 0 && !showExp && !reviewMode) {
      setSelected([...Array(q.options.length).keys()]);
    }
    if (screen === 'quiz' && q?.questionType === 'matching' && selected.length === 0 && !showExp && !reviewMode) {
      setSelected(Array(q.pairs.length).fill(null));
    }
  }, [qIdx, screen]);

  const isCorrect = useCallback((question, ans) => {
    if (ans === null || ans === undefined) return false;
    if (question.questionType === 'multi-select')
      return JSON.stringify([...(ans || [])].sort()) === JSON.stringify([...question.answers].sort());
    if (question.questionType === 'ordering')
      return JSON.stringify(ans) === JSON.stringify(question.answer);
    if (question.questionType === 'matching')
      return question.pairs.every((_, i) => (ans || [])[i] === i);
    return ans === question.answer;
  }, []);

  const handleSubmit = () => {
    const newAns = [...answers];
    const qType = q.questionType;
    if (qType === 'multi-select') newAns[qIdx] = [...selected].sort();
    else if (qType === 'ordering' || qType === 'matching') newAns[qIdx] = [...selected];
    else newAns[qIdx] = selected[0] ?? null;
    setAnswers(newAns);
    setShowExp(true);
  };

  const goNext = () => {
    if (qIdx < quizData.length - 1) {
      setQIdx(qIdx + 1);
      const nextAns = answers[qIdx + 1];
      setShowExp(reviewMode || nextAns !== null);
      setSelected([]);
    } else {
      setScreen('summary');
    }
  };

  const goPrev = () => {
    if (qIdx > 0) {
      setQIdx(qIdx - 1);
      setShowExp(answers[qIdx - 1] !== null);
      setSelected([]);
    }
  };

  const handleDragStart = (e, idx) => { setDraggedItem(idx); e.dataTransfer.effectAllowed = 'move'; };
  const handleDragOver = (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; };
  const handleDrop = (e, dropIdx) => {
    e.preventDefault();
    if (draggedItem === null) return;
    const arr = [...selected];
    const [item] = arr.splice(draggedItem, 1);
    arr.splice(dropIdx, 0, item);
    setSelected(arr);
    setDraggedItem(null);
  };

  // ── WELCOME ──────────────────────────────────────────────────
  if (screen === 'welcome') return (
    <div style={{ minHeight: '100vh', background: '#0a0a0f', fontFamily: "'Rajdhani', 'Segoe UI', sans-serif", padding: '2rem 1rem' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
        * { box-sizing: border-box; }
        :root { --accent: #f97316; --accent2: #fb923c; --surface: #111118; --surface2: #1a1a25; --border: #2a2a3a; --text: #e2e8f0; --muted: #64748b; }
        .glow { box-shadow: 0 0 20px rgba(249,115,22,0.15), 0 0 40px rgba(249,115,22,0.05); }
        .btn-primary { background: linear-gradient(135deg, #f97316, #ea580c); color: #fff; border: none; cursor: pointer; font-family: inherit; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; transition: all 0.2s; }
        .btn-primary:hover { filter: brightness(1.15); transform: translateY(-1px); }
        .tag { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
      `}</style>
      <div style={{ maxWidth: 800, margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          <div style={{ display: 'inline-block', background: 'rgba(249,115,22,0.1)', border: '1px solid rgba(249,115,22,0.3)', borderRadius: 8, padding: '4px 14px', marginBottom: '1rem', color: '#f97316', fontSize: '0.75rem', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.1em' }}>
            arXiv:2602.03205v1 · cs.RO · 3 Feb 2026
          </div>
          <h1 style={{ fontSize: 'clamp(2rem, 5vw, 3.2rem)', fontWeight: 700, color: '#f1f5f9', margin: '0 0 0.5rem', letterSpacing: '-0.01em', lineHeight: 1.1 }}>
            HUSKY
          </h1>
          <p style={{ color: '#f97316', fontSize: '1.05rem', fontWeight: 600, letterSpacing: '0.05em', marginBottom: '0.25rem' }}>
            HUMANOID SKATEBOARDING SYSTEM
          </p>
          <p style={{ color: '#64748b', fontSize: '0.9rem' }}>via Physics-Aware Whole-Body Control</p>
        </div>

        <div style={{ background: '#111118', border: '1px solid #2a2a3a', borderRadius: 12, padding: '1.5rem', marginBottom: '1.5rem' }}>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem', margin: 0, lineHeight: 1.6 }}>
            <strong style={{ color: '#f97316' }}>Authors:</strong> Han, Wang, Zhang, Liu, Luo, Bai, Li &nbsp;·&nbsp;
            <strong style={{ color: '#f97316' }}>Institution:</strong> TeleAI, China Telecom (+ SJTU, USTC, ShanghaiTech, HKU)
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
          {[['20', 'Questions'], ['4', 'Question Types'], ['Sections I–VI', 'Full Coverage']].map(([n, l]) => (
            <div key={l} style={{ background: '#111118', border: '1px solid #2a2a3a', borderRadius: 10, padding: '1rem', textAlign: 'center' }}>
              <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#f97316', fontFamily: "'JetBrains Mono', monospace" }}>{n}</div>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: 2 }}>{l}</div>
            </div>
          ))}
        </div>

        <div style={{ background: '#111118', border: '1px solid rgba(249,115,22,0.2)', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
          <p style={{ color: '#94a3b8', fontSize: '0.82rem', margin: 0, lineHeight: 1.7 }}>
            <span style={{ color: '#f97316', fontWeight: 600 }}>📚 Learning Cycle:</span> Every answer follows&nbsp;
            🧠 Intuition → 📐 Technical Formulation (w/ citations) → 💻 Implementation → 🔄 Validation
          </p>
        </div>

        <button className="btn-primary glow" onClick={() => { setScreen('quiz'); setSelected([]); }}
          style={{ width: '100%', padding: '1rem', borderRadius: 10, fontSize: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
          Start Quiz <Zap size={18} />
        </button>
      </div>
    </div>
  );

  // ── QUIZ ────────────────────────────────────────────────────
  if (screen === 'quiz') {
    const progress = ((qIdx + 1) / quizData.length) * 100;
    const answered = answers[qIdx] !== null;
    const correct = isCorrect(q, answers[qIdx]);
    const canSubmit = selected.length > 0 || (q.questionType === 'matching' && selected.some(s => s !== null));

    const renderOptions = () => {
      if (q.questionType === 'single-choice') {
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            {q.options.map((opt, i) => {
              const sel = selected.includes(i);
              const isC = showExp && i === q.answer;
              const isW = showExp && sel && i !== q.answer;
              return (
                <button key={i} disabled={showExp || reviewMode} onClick={() => !showExp && setSelected([i])}
                  style={{ background: showExp ? (isC ? 'rgba(34,197,94,0.1)' : isW ? 'rgba(239,68,68,0.1)' : '#111118') : sel ? 'rgba(249,115,22,0.12)' : '#111118', border: `2px solid ${showExp ? (isC ? '#22c55e' : isW ? '#ef4444' : '#2a2a3a') : sel ? '#f97316' : '#2a2a3a'}`, borderRadius: 8, padding: '0.85rem 1rem', textAlign: 'left', cursor: showExp ? 'default' : 'pointer', transition: 'all 0.15s', display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                  <span style={{ flexShrink: 0, width: 26, height: 26, borderRadius: '50%', background: showExp ? (isC ? '#22c55e' : isW ? '#ef4444' : '#1e1e2e') : sel ? '#f97316' : '#1e1e2e', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', fontWeight: 700, color: (sel || isC || isW) ? '#fff' : '#64748b', fontFamily: "'JetBrains Mono', monospace" }}>
                    {String.fromCharCode(65 + i)}
                  </span>
                  <span style={{ color: '#c8d3e0', fontSize: '0.9rem', lineHeight: 1.5, flex: 1 }}>{opt}</span>
                </button>
              );
            })}
          </div>
        );
      }

      if (q.questionType === 'multi-select') {
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            {q.options.map((opt, i) => {
              const sel = selected.includes(i);
              const isC = showExp && q.answers.includes(i);
              const isW = showExp && sel && !q.answers.includes(i);
              return (
                <button key={i} disabled={showExp || reviewMode}
                  onClick={() => { if (showExp) return; const ns = [...selected]; const xi = ns.indexOf(i); xi > -1 ? ns.splice(xi, 1) : ns.push(i); setSelected(ns.sort()); }}
                  style={{ background: showExp ? (isC ? 'rgba(34,197,94,0.1)' : isW ? 'rgba(239,68,68,0.1)' : '#111118') : sel ? 'rgba(249,115,22,0.12)' : '#111118', border: `2px solid ${showExp ? (isC ? '#22c55e' : isW ? '#ef4444' : '#2a2a3a') : sel ? '#f97316' : '#2a2a3a'}`, borderRadius: 8, padding: '0.85rem 1rem', textAlign: 'left', cursor: showExp ? 'default' : 'pointer', display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                  <span style={{ flexShrink: 0, width: 20, height: 20, borderRadius: 4, border: `2px solid ${isC ? '#22c55e' : isW ? '#ef4444' : sel ? '#f97316' : '#3f3f5a'}`, background: (sel || isC) ? (isC ? '#22c55e' : '#f97316') : 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    {(sel || isC) && <span style={{ color: '#fff', fontSize: '0.65rem', fontWeight: 700 }}>✓</span>}
                  </span>
                  <span style={{ color: '#c8d3e0', fontSize: '0.9rem', lineHeight: 1.5, flex: 1 }}>{opt}</span>
                </button>
              );
            })}
          </div>
        );
      }

      if (q.questionType === 'ordering') {
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {(selected.length > 0 ? selected : [...Array(q.options.length).keys()]).map((oi, pos) => {
              const correctPos = showExp && q.answer[pos] === oi;
              return (
                <div key={oi} draggable={!showExp && !reviewMode}
                  onDragStart={(e) => handleDragStart(e, pos)}
                  onDragOver={handleDragOver}
                  onDrop={(e) => handleDrop(e, pos)}
                  style={{ background: showExp ? (correctPos ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)') : '#111118', border: `2px solid ${showExp ? (correctPos ? '#22c55e' : '#ef4444') : '#2a2a3a'}`, borderRadius: 8, padding: '0.85rem 1rem', display: 'flex', alignItems: 'center', gap: 10, cursor: showExp ? 'default' : 'grab', opacity: draggedItem === pos ? 0.4 : 1, transition: 'opacity 0.15s' }}>
                  <GripVertical size={16} color="#3f3f5a" style={{ flexShrink: 0 }} />
                  <span style={{ width: 24, height: 24, borderRadius: '50%', background: '#1e1e2e', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.7rem', fontWeight: 700, color: '#f97316', flexShrink: 0, fontFamily: "'JetBrains Mono', monospace" }}>{pos + 1}</span>
                  <span style={{ color: '#c8d3e0', fontSize: '0.88rem', lineHeight: 1.5, flex: 1 }}>{q.options[oi]}</span>
                  {showExp && !correctPos && <span style={{ fontSize: '0.7rem', color: '#ef4444', flexShrink: 0 }}>→ pos {q.answer.indexOf(oi) + 1}</span>}
                </div>
              );
            })}
          </div>
        );
      }

      if (q.questionType === 'matching') {
        const selArr = selected.length > 0 ? selected : Array(q.pairs.length).fill(null);
        return (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div>
              <p style={{ color: '#64748b', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>Items</p>
              {q.pairs.map((p, i) => (
                <div key={i} style={{ background: '#0d0d15', border: '1px solid #2a2a3a', borderRadius: 8, padding: '0.75rem', marginBottom: '0.5rem', color: '#94a3b8', fontSize: '0.82rem', lineHeight: 1.4 }}>{p.item}</div>
              ))}
            </div>
            <div>
              <p style={{ color: '#64748b', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '0.5rem' }}>Matches</p>
              {showExp ? q.pairs.map((p, i) => {
                const userMatch = (selArr[i] !== null && selArr[i] !== undefined) ? q.pairs[selArr[i]]?.match : null;
                const correct2 = selArr[i] === i;
                return (
                  <div key={i} style={{ background: correct2 ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)', border: `2px solid ${correct2 ? '#22c55e' : '#ef4444'}`, borderRadius: 8, padding: '0.75rem', marginBottom: '0.5rem' }}>
                    <div style={{ color: '#c8d3e0', fontSize: '0.82rem', lineHeight: 1.4 }}>{userMatch || '(no match)'}</div>
                    {!correct2 && <div style={{ color: '#22c55e', fontSize: '0.75rem', marginTop: 4 }}>✓ {p.match}</div>}
                  </div>
                );
              }) : (
                <>
                  <div style={{ background: '#0d0d15', border: '1px solid rgba(249,115,22,0.2)', borderRadius: 8, padding: '0.5rem', marginBottom: '0.5rem' }}>
                    <p style={{ color: '#94a3b8', fontSize: '0.72rem', margin: '0 0 0.4rem' }}>Drag to assign:</p>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      {shuffledMatches.map((m, mi) => {
                        const used = selArr.includes(m.idx);
                        return (
                          <div key={mi} draggable={!used}
                            onDragStart={(e) => e.dataTransfer.setData('matchIdx', m.idx.toString())}
                            style={{ background: used ? '#1a1a25' : '#1e1e2e', border: `1px solid ${used ? '#2a2a3a' : '#3f3f5a'}`, borderRadius: 5, padding: '3px 8px', fontSize: '0.72rem', color: used ? '#3f3f5a' : '#94a3b8', cursor: used ? 'default' : 'grab', transition: 'all 0.15s' }}>
                            {m.text}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  {q.pairs.map((_, i) => (
                    <div key={i}
                      onDragOver={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = '#f97316'; }}
                      onDragLeave={(e) => { e.currentTarget.style.borderColor = selArr[i] !== null ? '#f97316' : '#2a2a3a'; }}
                      onDrop={(e) => {
                        e.preventDefault();
                        e.currentTarget.style.borderColor = '#f97316';
                        const mi = parseInt(e.dataTransfer.getData('matchIdx'));
                        const ns = [...selArr];
                        const existing = ns.indexOf(mi);
                        if (existing !== -1) ns[existing] = null;
                        ns[i] = mi;
                        setSelected([...ns]);
                      }}
                      style={{ background: '#0d0d15', border: `2px dashed ${selArr[i] !== null ? '#f97316' : '#2a2a3a'}`, borderRadius: 8, padding: '0.75rem', marginBottom: '0.5rem', minHeight: 44, display: 'flex', alignItems: 'center', justifyContent: 'space-between', transition: 'border-color 0.15s' }}>
                      {selArr[i] !== null && selArr[i] !== undefined ? (
                        <>
                          <span style={{ color: '#c8d3e0', fontSize: '0.8rem' }}>{q.pairs[selArr[i]]?.match}</span>
                          <button onClick={() => { const ns = [...selArr]; ns[i] = null; setSelected([...ns]); }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#ef4444', padding: 2 }}><XCircle size={14} /></button>
                        </>
                      ) : (
                        <span style={{ color: '#3f3f5a', fontSize: '0.75rem' }}>Drop here</span>
                      )}
                    </div>
                  ))}
                </>
              )}
            </div>
          </div>
        );
      }
      return null;
    };

    const renderExplanation = () => (
      <div style={{ background: correct ? 'rgba(34,197,94,0.05)' : 'rgba(239,68,68,0.05)', border: `1px solid ${correct ? 'rgba(34,197,94,0.25)' : 'rgba(239,68,68,0.25)'}`, borderRadius: 10, padding: '1.25rem', marginTop: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '1rem' }}>
          {correct ? <CheckCircle size={20} color="#22c55e" /> : <XCircle size={20} color="#ef4444" />}
          <span style={{ fontWeight: 700, color: correct ? '#22c55e' : '#ef4444', fontSize: '0.95rem' }}>
            {correct ? 'Correct!' : "Not quite — here's the full breakdown:"}
          </span>
        </div>
        {[
          ['🧠', 'INTUITION', q.explanation.intuition, '#111118'],
          ['📐', 'TECHNICAL', q.explanation.math, '#0a0a12'],
          ['💻', 'IMPLEMENTATION', q.explanation.computation, '#0a0f0a'],
          ['🔄', 'VALIDATION', q.explanation.connection, '#0f0a0a'],
        ].map(([icon, label, content, bg]) => (
          <div key={label} style={{ background: bg, border: '1px solid #1e1e2e', borderRadius: 8, padding: '0.85rem', marginBottom: '0.6rem' }}>
            <div style={{ fontWeight: 700, color: '#f97316', fontSize: '0.75rem', letterSpacing: '0.08em', marginBottom: '0.4rem', fontFamily: "'JetBrains Mono', monospace" }}>{icon} {label}</div>
            {label === 'IMPLEMENTATION' ? (
              <pre style={{ color: '#86efac', fontSize: '0.75rem', lineHeight: 1.6, fontFamily: "'JetBrains Mono', monospace", whiteSpace: 'pre-wrap', overflowX: 'auto', margin: 0 }}>{content}</pre>
            ) : (
              <p style={{ color: '#94a3b8', fontSize: '0.83rem', lineHeight: 1.6, margin: 0, fontFamily: label === 'TECHNICAL' ? "'JetBrains Mono', monospace" : 'inherit', whiteSpace: label === 'TECHNICAL' ? 'pre-wrap' : 'normal' }}>{content}</p>
            )}
          </div>
        ))}
        <p style={{ color: '#3f3f5a', fontSize: '0.72rem', margin: 0, fontFamily: "'JetBrains Mono', monospace" }}>
          📍 {q.pageReference}
        </p>
      </div>
    );

    return (
      <div style={{ minHeight: '100vh', background: '#0a0a0f', fontFamily: "'Rajdhani', 'Segoe UI', sans-serif", padding: '1.5rem 1rem' }}>
        <style>{`@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap'); * { box-sizing: border-box; }`}</style>
        <div style={{ maxWidth: 820, margin: '0 auto' }}>
          {/* Header */}
          <div style={{ marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8, marginBottom: '0.75rem' }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
                <span style={{ background: 'rgba(249,115,22,0.12)', color: '#f97316', padding: '3px 8px', borderRadius: 5, fontSize: '0.72rem', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>{q.section}</span>
                <span style={{ background: q.difficulty === 'Easy' ? 'rgba(34,197,94,0.12)' : q.difficulty === 'Medium' ? 'rgba(234,179,8,0.12)' : 'rgba(239,68,68,0.12)', color: q.difficulty === 'Easy' ? '#22c55e' : q.difficulty === 'Medium' ? '#eab308' : '#ef4444', padding: '3px 8px', borderRadius: 5, fontSize: '0.72rem', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>{q.difficulty}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ color: '#3f3f5a', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: 4, fontFamily: "'JetBrains Mono', monospace" }}><Clock size={14} />{formatTime(timeSpent)}</span>
                <span style={{ color: '#3f3f5a', fontSize: '0.8rem', fontFamily: "'JetBrains Mono', monospace" }}>{qIdx + 1}/{quizData.length}</span>
              </div>
            </div>
            <div style={{ width: '100%', background: '#1a1a25', height: 4, borderRadius: 2, overflow: 'hidden' }}>
              <div style={{ width: `${progress}%`, height: '100%', background: 'linear-gradient(90deg, #f97316, #fb923c)', transition: 'width 0.4s ease', borderRadius: 2 }} />
            </div>
          </div>

          {/* Question card */}
          <div style={{ background: '#111118', border: '1px solid #2a2a3a', borderRadius: 12, padding: '1.5rem' }}>
            <h2 style={{ color: '#e2e8f0', fontSize: 'clamp(0.95rem, 2.5vw, 1.1rem)', fontWeight: 600, lineHeight: 1.55, marginBottom: '0.4rem' }}>{q.question}</h2>
            {q.questionType !== 'single-choice' && (
              <p style={{ color: '#3f3f5a', fontSize: '0.78rem', fontStyle: 'italic', marginBottom: '1rem' }}>
                {q.questionType === 'multi-select' && '✓ Select all that apply'}
                {q.questionType === 'ordering' && '⇅ Drag to arrange in correct order'}
                {q.questionType === 'matching' && '⟷ Drag items to match'}
              </p>
            )}
            <div style={{ marginBottom: '1rem' }}>{renderOptions()}</div>
            {showExp && renderExplanation()}
          </div>

          {/* Navigation */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem', gap: 8 }}>
            <button disabled={qIdx === 0} onClick={goPrev}
              style={{ background: '#111118', border: '1px solid #2a2a3a', color: qIdx === 0 ? '#2a2a3a' : '#94a3b8', borderRadius: 8, padding: '0.65rem 1.2rem', cursor: qIdx === 0 ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', gap: 6, fontFamily: 'inherit', fontSize: '0.85rem', fontWeight: 600 }}>
              <ChevronLeft size={16} /> Prev
            </button>
            {!showExp && !reviewMode ? (
              <button disabled={!canSubmit} onClick={handleSubmit}
                style={{ background: canSubmit ? 'linear-gradient(135deg, #f97316, #ea580c)' : '#1a1a25', border: 'none', color: canSubmit ? '#fff' : '#3f3f5a', borderRadius: 8, padding: '0.65rem 1.5rem', cursor: canSubmit ? 'pointer' : 'not-allowed', fontFamily: 'inherit', fontSize: '0.9rem', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                Submit Answer
              </button>
            ) : (
              <button onClick={goNext}
                style={{ background: qIdx === quizData.length - 1 ? 'linear-gradient(135deg, #22c55e, #16a34a)' : 'linear-gradient(135deg, #f97316, #ea580c)', border: 'none', color: '#fff', borderRadius: 8, padding: '0.65rem 1.5rem', cursor: 'pointer', fontFamily: 'inherit', fontSize: '0.9rem', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 6 }}>
                {qIdx === quizData.length - 1 ? <><Trophy size={16} /> Finish</> : <>Next <ChevronRight size={16} /></>}
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ── SUMMARY ──────────────────────────────────────────────────
  if (screen === 'summary') {
    const score = answers.filter((a, i) => isCorrect(quizData[i], a)).length;
    const pct = Math.round((score / quizData.length) * 100);

    const topics = {};
    quizData.forEach((q2, i) => {
      const t = q2.topic || 'OTHER';
      if (!topics[t]) topics[t] = { total: 0, correct: 0 };
      topics[t].total++;
      if (isCorrect(q2, answers[i])) topics[t].correct++;
    });

    return (
      <div style={{ minHeight: '100vh', background: '#0a0a0f', fontFamily: "'Rajdhani', 'Segoe UI', sans-serif", padding: '2rem 1rem' }}>
        <style>{`@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap'); * { box-sizing: border-box; }`}</style>
        <div style={{ maxWidth: 760, margin: '0 auto' }}>
          <h1 style={{ textAlign: 'center', color: '#f1f5f9', fontSize: 'clamp(1.8rem, 5vw, 2.5rem)', fontWeight: 700, marginBottom: '1.5rem' }}>Quiz Complete</h1>

          <div style={{ background: 'linear-gradient(135deg, rgba(249,115,22,0.1), rgba(251,146,60,0.05))', border: '1px solid rgba(249,115,22,0.2)', borderRadius: 14, padding: '2rem', textAlign: 'center', marginBottom: '1.5rem' }}>
            <div style={{ fontSize: 'clamp(3rem, 10vw, 4.5rem)', fontWeight: 700, color: '#f97316', fontFamily: "'JetBrains Mono', monospace", lineHeight: 1 }}>{pct}%</div>
            <div style={{ color: '#94a3b8', fontSize: '1.1rem', marginTop: 8 }}>{score} / {quizData.length} correct</div>
            <div style={{ color: '#3f3f5a', fontSize: '0.85rem', marginTop: 4, fontFamily: "'JetBrains Mono', monospace" }}>Time: {formatTime(timeSpent)}</div>
          </div>

          <div style={{ background: '#111118', border: '1px solid #2a2a3a', borderRadius: 12, padding: '1.25rem', marginBottom: '1.5rem' }}>
            <h3 style={{ color: '#f1f5f9', fontSize: '1rem', marginBottom: '1rem', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Performance by Topic</h3>
            {Object.entries(topics).map(([topic, stats]) => {
              const tp = Math.round((stats.correct / stats.total) * 100);
              return (
                <div key={topic} style={{ marginBottom: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>{topic.replace(/_/g, ' ')}</span>
                    <span style={{ color: tp >= 80 ? '#22c55e' : tp >= 50 ? '#eab308' : '#ef4444', fontSize: '0.8rem', fontFamily: "'JetBrains Mono', monospace" }}>{stats.correct}/{stats.total}</span>
                  </div>
                  <div style={{ background: '#1a1a25', height: 5, borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ width: `${tp}%`, height: '100%', background: tp >= 80 ? '#22c55e' : tp >= 50 ? '#eab308' : '#ef4444', borderRadius: 3, transition: 'width 0.6s ease' }} />
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <button onClick={() => { setReviewMode(true); setQIdx(0); setShowExp(true); setScreen('quiz'); }}
              style={{ background: '#111118', border: '1px solid rgba(249,115,22,0.3)', color: '#f97316', borderRadius: 10, padding: '0.85rem', cursor: 'pointer', fontFamily: 'inherit', fontSize: '0.9rem', fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
              <BookOpen size={16} /> Review
            </button>
            <button onClick={() => { setAnswers(Array(quizData.length).fill(null)); setQIdx(0); setShowExp(false); setReviewMode(false); setSelected([]); resetTimer(); setScreen('welcome'); }}
              style={{ background: 'linear-gradient(135deg, #f97316, #ea580c)', border: 'none', color: '#fff', borderRadius: 10, padding: '0.85rem', cursor: 'pointer', fontFamily: 'inherit', fontSize: '0.9rem', fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
              <RefreshCw size={16} /> Restart
            </button>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default Quiz;
