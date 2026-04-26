"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SURAKSHA — Credit Risk Assessment Simulator              ║
║                                                                              ║
║  A standalone Python desktop application that simulates a bank environment   ║
║  where a customer applies for a loan. Uses Fuzzy Logic (skfuzzy) for        ║
║  preliminary risk assessment and an MLP Neural Network (sklearn) for the     ║
║  final loan decision.                                                        ║
║                                                                              ║
║  Soft Computing Concepts Used:                                               ║
║  1. Fuzzy Logic — Models uncertainty in human-like decision making           ║
║  2. MLP (Multi-Layer Perceptron) — Learns patterns via backpropagation       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pygame
import sys
import math
import random
import numpy as np
import asyncio
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")  # Suppress sklearn convergence warnings

# =============================================================================
# INITIALIZE PYGAME
# =============================================================================
pygame.init()
pygame.font.init()

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
WIDTH, HEIGHT = 1200, 700
FPS = 60

# --- Premium Dark Color Palette ---
BG_DARK       = (12, 15, 28)        # Deep navy background
BG_PANEL      = (20, 24, 45)        # Panel background
BG_CARD       = (28, 33, 62)        # Card/input area background
ACCENT_GOLD   = (255, 200, 60)      # Gold accent
ACCENT_TEAL   = (0, 220, 200)       # Teal accent
ACCENT_BLUE   = (60, 130, 255)      # Blue accent
TEXT_WHITE     = (240, 240, 250)     # Primary text
TEXT_DIM       = (140, 145, 170)     # Dimmed text
TEXT_DARK      = (40, 40, 60)        # Dark text
GREEN_APPROVE  = (40, 210, 120)      # Approved green
RED_REJECT     = (240, 60, 80)       # Rejected red
SLIDER_TRACK   = (50, 55, 80)       # Slider track color
SLIDER_FILL    = (60, 130, 255)     # Slider filled portion
NODE_COLOR     = (100, 180, 255)    # Neural net node
EDGE_COLOR     = (60, 80, 120)      # Neural net edge
GAUGE_BG       = (40, 44, 70)       # Gauge background arc
SKIN_COLOR     = (240, 200, 160)    # Character skin
SHIRT_COLOR    = (60, 130, 255)     # Character shirt
PANTS_COLOR    = (40, 50, 80)       # Character pants
DESK_COLOR     = (100, 70, 45)      # Bank desk
TELLER_SHIRT   = (180, 50, 50)      # Teller shirt

# --- State Machine ---
STATE_INPUT       = 0
STATE_WALKING     = 1
STATE_CALCULATING = 2
STATE_RESULT      = 3

# --- Fonts ---
FONT_TITLE   = pygame.font.SysFont("Segoe UI", 38, bold=True)
FONT_HEADING = pygame.font.SysFont("Segoe UI", 26, bold=True)
FONT_BODY    = pygame.font.SysFont("Segoe UI", 20)
FONT_SMALL   = pygame.font.SysFont("Segoe UI", 16)
FONT_LABEL   = pygame.font.SysFont("Segoe UI", 18, bold=True)
FONT_VALUE   = pygame.font.SysFont("Segoe UI", 22, bold=True)
FONT_STAMP   = pygame.font.SysFont("Impact", 72, bold=True)
FONT_GAUGE   = pygame.font.SysFont("Segoe UI", 48, bold=True)
FONT_BTN     = pygame.font.SysFont("Segoe UI", 22, bold=True)
FONT_BANNER  = pygame.font.SysFont("Segoe UI", 14)

# =============================================================================
# FUZZY LOGIC MODULE
# =============================================================================
"""
FUZZY LOGIC — Soft Computing Concept #1
========================================
Fuzzy Logic extends classical Boolean logic by allowing truth values between
0 and 1 instead of just True/False. This models the inherent vagueness in
human reasoning (e.g., "high income" is not a sharp boundary).

KEY CONCEPTS:
  • Membership Function (MF): Maps a crisp input value to a degree of
    membership [0, 1] in a fuzzy set. We use triangular MFs here:
        μ(x) = max(0, min((x-a)/(b-a), (c-x)/(c-b)))
    where [a, b, c] define the triangle's left foot, peak, and right foot.

  • Fuzzy Rules: IF-THEN rules that combine fuzzy sets using AND (min)
    and OR (max) operators.

  • Defuzzification: Converts the fuzzy output back to a crisp number.
    We use the Centroid method:
        z* = ∫ μ(z)·z dz  /  ∫ μ(z) dz
"""

def build_fuzzy_system():
    """
    Constructs the fuzzy inference system with three input variables
    (Income, Debt, Age) and one output variable (Risk Index).
    """
    # --- Define Universe of Discourse (range of each variable) ---
    income = ctrl.Antecedent(np.arange(0, 20000001, 100000), 'income')
    debt   = ctrl.Antecedent(np.arange(0, 100001, 1000), 'debt')
    age    = ctrl.Antecedent(np.arange(18, 81, 1), 'age')
    risk   = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    # --- Triangular Membership Functions ---
    # Income: Low (0-80k peak at 0), Medium (40k-160k peak at 100k), High (120k-200k peak at 200k)
    income['low']    = fuzz.trimf(income.universe, [0, 0, 8000000])
    income['medium'] = fuzz.trimf(income.universe, [4000000, 10000000, 16000000])
    income['high']   = fuzz.trimf(income.universe, [12000000, 20000000, 20000000])

    # Debt: Safe (0-40k peak at 0), Moderate (20k-70k peak at 45k), Risky (50k-100k peak at 100k)
    debt['safe']     = fuzz.trimf(debt.universe, [0, 0, 40000])
    debt['moderate'] = fuzz.trimf(debt.universe, [20000, 45000, 70000])
    debt['risky']    = fuzz.trimf(debt.universe, [50000, 100000, 100000])

    # Age: Young (18-35 peak at 18), Middle (28-60 peak at 44), Senior (50-80 peak at 80)
    age['young']     = fuzz.trimf(age.universe, [18, 18, 35])
    age['middle']    = fuzz.trimf(age.universe, [28, 44, 60])
    age['senior']    = fuzz.trimf(age.universe, [50, 80, 80])

    # Risk: Low (0-40 peak at 0), Medium (20-70 peak at 50), High (60-100 peak at 100)
    risk['low']      = fuzz.trimf(risk.universe, [0, 0, 40])
    risk['medium']   = fuzz.trimf(risk.universe, [20, 50, 80])
    risk['high']     = fuzz.trimf(risk.universe, [60, 100, 100])

    # --- Fuzzy Rules ---
    # These rules encode expert banking knowledge:
    rules = [
        # High income + safe debt → low risk (ideal borrower)
        ctrl.Rule(income['high'] & debt['safe'], risk['low']),
        # High income + risky debt → medium risk (earns well but over-leveraged)
        ctrl.Rule(income['high'] & debt['risky'], risk['medium']),
        # Low income + risky debt → high risk (poor capacity to repay)
        ctrl.Rule(income['low'] & debt['risky'], risk['high']),
        # Low income + safe debt → medium risk (limited income but manageable debt)
        ctrl.Rule(income['low'] & debt['safe'], risk['medium']),
        # Medium income + moderate debt → medium risk (average case)
        ctrl.Rule(income['medium'] & debt['moderate'], risk['medium']),
        # Middle-aged + medium income → low risk (stable career phase)
        ctrl.Rule(age['middle'] & income['medium'], risk['low']),
        # Young + low income → high risk (early career, limited earnings)
        ctrl.Rule(age['young'] & income['low'], risk['high']),
        # Senior + high income → low risk (established earner)
        ctrl.Rule(age['senior'] & income['high'], risk['low']),
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system), income, debt, age

def compute_fuzzy(sim, income_val, debt_val, age_val):
    """
    Computes the Preliminary Risk Index (0–100) from crisp inputs.
    Also returns the raw membership values for each variable (used as MLP features).
    """
    # Clamp values to valid ranges
    income_val = np.clip(income_val, 0, 20000000)
    debt_val   = np.clip(debt_val, 0, 100000)
    age_val    = np.clip(age_val, 18, 80)

    sim.input['income'] = income_val
    sim.input['debt']   = debt_val
    sim.input['age']    = age_val

    try:
        sim.compute()
        risk_index = sim.output['risk']
    except Exception:
        # Fallback if no rules fire (edge case)
        risk_index = 50.0

    # --- Compute membership degrees for MLP input ---
    # These represent "how much" a value belongs to each fuzzy set
    inc_lo  = fuzz.interp_membership(np.arange(0, 20000001, 100000), fuzz.trimf(np.arange(0, 20000001, 100000), [0, 0, 8000000]), income_val)
    inc_md  = fuzz.interp_membership(np.arange(0, 20000001, 100000), fuzz.trimf(np.arange(0, 20000001, 100000), [4000000, 10000000, 16000000]), income_val)
    inc_hi  = fuzz.interp_membership(np.arange(0, 20000001, 100000), fuzz.trimf(np.arange(0, 20000001, 100000), [12000000, 20000000, 20000000]), income_val)

    dbt_sf  = fuzz.interp_membership(np.arange(0, 100001, 1000), fuzz.trimf(np.arange(0, 100001, 1000), [0, 0, 40000]), debt_val)
    dbt_md  = fuzz.interp_membership(np.arange(0, 100001, 1000), fuzz.trimf(np.arange(0, 100001, 1000), [20000, 45000, 70000]), debt_val)
    dbt_rk  = fuzz.interp_membership(np.arange(0, 100001, 1000), fuzz.trimf(np.arange(0, 100001, 1000), [50000, 100000, 100000]), debt_val)

    age_yg  = fuzz.interp_membership(np.arange(18, 81, 1), fuzz.trimf(np.arange(18, 81, 1), [18, 18, 35]), age_val)
    age_md  = fuzz.interp_membership(np.arange(18, 81, 1), fuzz.trimf(np.arange(18, 81, 1), [28, 44, 60]), age_val)
    age_sr  = fuzz.interp_membership(np.arange(18, 81, 1), fuzz.trimf(np.arange(18, 81, 1), [50, 80, 80]), age_val)

    memberships = [inc_lo, inc_md, inc_hi, dbt_sf, dbt_md, dbt_rk, age_yg, age_md, age_sr]
    return risk_index, memberships


# =============================================================================
# MLP NEURAL NETWORK MODULE
# =============================================================================
"""
MLP (Multi-Layer Perceptron) — Soft Computing Concept #2
=========================================================
An MLP is a feedforward neural network trained via BACKPROPAGATION.

ARCHITECTURE:
  Input Layer (9 neurons: fuzzy membership values)
      → Hidden Layer 1 (8 neurons, ReLU activation)
      → Hidden Layer 2 (4 neurons, ReLU activation)
      → Output Layer (1 neuron, sigmoid → binary: Approved / Denied)

BACKPROPAGATION (how the network learns):
  1. FORWARD PASS: Input flows through layers, each neuron computes:
       z = Σ(wᵢ·xᵢ) + b       (weighted sum + bias)
       a = σ(z)                 (activation function, e.g. ReLU = max(0,z))

  2. LOSS COMPUTATION: Compare prediction ŷ with true label y:
       L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]   (binary cross-entropy)

  3. BACKWARD PASS: Compute gradient ∂L/∂w for each weight using the
     chain rule, propagating error backwards through layers:
       ∂L/∂wᵢ = ∂L/∂a · ∂a/∂z · ∂z/∂wᵢ

  4. WEIGHT UPDATE: Adjust weights using gradient descent:
       w_new = w_old - η · ∂L/∂w     (η = learning rate)

  This cycle repeats for many epochs until the loss converges.
"""

def build_mlp(fuzzy_sim):
    """
    Generates a synthetic training dataset and trains an MLPClassifier.
    The inputs are the 9 fuzzy membership values; the output is binary
    (1 = Approved, 0 = Denied).
    """
    np.random.seed(42)
    n_samples = 300

    # Generate random crisp inputs
    incomes = np.random.uniform(1000000, 20000000, n_samples)
    debts   = np.random.uniform(0, 100000, n_samples)
    ages    = np.random.uniform(18, 80, n_samples)

    X = []  # Feature matrix (membership values)
    y = []  # Labels (0 or 1)

    for i in range(n_samples):
        risk_idx, memberships = compute_fuzzy(fuzzy_sim, incomes[i], debts[i], ages[i])
        X.append(memberships)
        # Labeling logic: low risk → approved, high risk → denied
        # Add some noise for realism
        threshold = 50 + np.random.normal(0, 8)
        label = 1 if risk_idx < threshold else 0
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Normalize features for better gradient flow during backpropagation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(8, 4),  # Two hidden layers
        activation='relu',          # ReLU: f(x) = max(0, x)
        solver='adam',              # Adam optimizer (adaptive learning rate)
        max_iter=500,               # Maximum training epochs
        random_state=42,
        learning_rate_init=0.001    # Initial learning rate η
    )
    mlp.fit(X_scaled, y)

    return mlp, scaler

def predict_loan(mlp, scaler, memberships):
    """
    Uses the trained MLP to predict loan approval.
    Returns: (decision: 0 or 1, confidence: float 0-1)
    """
    X_input = np.array(memberships).reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    decision = mlp.predict(X_scaled)[0]
    probas = mlp.predict_proba(X_scaled)[0]
    confidence = max(probas)
    return int(decision), confidence


# =============================================================================
# GUI COMPONENTS
# =============================================================================

class Slider:
    """A custom slider widget for Pygame (no external images needed)."""
    def __init__(self, x, y, w, min_val, max_val, initial, label, fmt="{}",
                 color=ACCENT_BLUE):
        self.rect = pygame.Rect(x, y, w, 8)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.fmt = fmt
        self.color = color
        self.knob_r = 14
        self.dragging = False

    @property
    def knob_x(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + int(ratio * self.rect.w)

    def handle_event(self, event):
        kx = self.knob_x
        ky = self.rect.centery
        if event.type == pygame.MOUSEBUTTONDOWN:
            dist = math.hypot(event.pos[0] - kx, event.pos[1] - ky)
            if dist <= self.knob_r + 6:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = max(0, min(event.pos[0] - self.rect.x, self.rect.w))
            ratio = rel_x / self.rect.w
            self.value = self.min_val + ratio * (self.max_val - self.min_val)

    def draw(self, surface):
        # Label
        lbl = FONT_LABEL.render(self.label, True, TEXT_WHITE)
        surface.blit(lbl, (self.rect.x, self.rect.y - 30))

        # Value
        val_str = self.fmt.format(int(self.value))
        val_surf = FONT_VALUE.render(val_str, True, self.color)
        surface.blit(val_surf, (self.rect.right - val_surf.get_width(), self.rect.y - 32))

        # Track background
        pygame.draw.rect(surface, SLIDER_TRACK, self.rect, border_radius=4)

        # Filled portion
        fill_rect = pygame.Rect(self.rect.x, self.rect.y,
                                self.knob_x - self.rect.x, self.rect.h)
        if fill_rect.w > 0:
            pygame.draw.rect(surface, self.color, fill_rect, border_radius=4)

        # Knob
        kx = self.knob_x
        ky = self.rect.centery
        pygame.draw.circle(surface, (255, 255, 255), (kx, ky), self.knob_r)
        pygame.draw.circle(surface, self.color, (kx, ky), self.knob_r - 3)

        # Glow when dragging
        if self.dragging:
            glow = pygame.Surface((self.knob_r * 4, self.knob_r * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*self.color, 40),
                               (self.knob_r * 2, self.knob_r * 2), self.knob_r * 2)
            surface.blit(glow, (kx - self.knob_r * 2, ky - self.knob_r * 2))


class Button:
    """A styled button with hover glow effect."""
    def __init__(self, x, y, w, h, text, color=ACCENT_GOLD, text_color=TEXT_DARK):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and self.hovered:
            return True
        return False

    def draw(self, surface):
        # Glow
        if self.hovered:
            glow = pygame.Surface((self.rect.w + 20, self.rect.h + 20), pygame.SRCALPHA)
            pygame.draw.rect(glow, (*self.color, 50),
                             (0, 0, self.rect.w + 20, self.rect.h + 20),
                             border_radius=18)
            surface.blit(glow, (self.rect.x - 10, self.rect.y - 10))

        # Button body
        c = tuple(min(255, v + 30) for v in self.color) if self.hovered else self.color
        pygame.draw.rect(surface, c, self.rect, border_radius=12)

        # Text
        txt = FONT_BTN.render(self.text, True, self.text_color)
        surface.blit(txt, (self.rect.centerx - txt.get_width() // 2,
                           self.rect.centery - txt.get_height() // 2))


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def draw_character(surface, x, y, scale=1.0, leg_phase=0):
    """
    Draws a simple 2D person using basic shapes.
    x, y = position of the character's feet center.
    leg_phase = animation phase for walking (0 to 2π).
    """
    s = scale
    # Legs (animated swing)
    leg_swing = int(8 * s * math.sin(leg_phase))
    pygame.draw.line(surface, PANTS_COLOR,
                     (x - int(6*s), y - int(30*s)), (x - int(6*s) - leg_swing, y), int(4*s))
    pygame.draw.line(surface, PANTS_COLOR,
                     (x + int(6*s), y - int(30*s)), (x + int(6*s) + leg_swing, y), int(4*s))

    # Body (shirt)
    body_rect = pygame.Rect(x - int(12*s), y - int(55*s), int(24*s), int(28*s))
    pygame.draw.rect(surface, SHIRT_COLOR, body_rect, border_radius=int(5*s))

    # Arms (animated swing, opposite to legs)
    arm_swing = int(6 * s * math.sin(leg_phase + math.pi))
    pygame.draw.line(surface, SKIN_COLOR,
                     (x - int(12*s), y - int(50*s)),
                     (x - int(20*s) - arm_swing, y - int(30*s)), int(3*s))
    pygame.draw.line(surface, SKIN_COLOR,
                     (x + int(12*s), y - int(50*s)),
                     (x + int(20*s) + arm_swing, y - int(30*s)), int(3*s))

    # Head
    pygame.draw.circle(surface, SKIN_COLOR, (x, y - int(65*s)), int(14*s))

    # Hair
    pygame.draw.arc(surface, (50, 40, 30),
                    (x - int(14*s), y - int(80*s), int(28*s), int(20*s)),
                    0, math.pi, int(3*s))

    # Eyes
    pygame.draw.circle(surface, (30, 30, 50), (x - int(5*s), y - int(67*s)), int(2*s))
    pygame.draw.circle(surface, (30, 30, 50), (x + int(5*s), y - int(67*s)), int(2*s))

    # Smile
    pygame.draw.arc(surface, (200, 100, 100),
                    (x - int(5*s), y - int(63*s), int(10*s), int(6*s)),
                    math.pi, 2*math.pi, int(1*s + 1))

    # Briefcase
    bc_x = x + int(20*s) + arm_swing
    bc_y = y - int(32*s)
    pygame.draw.rect(surface, (80, 60, 40),
                     (bc_x - int(6*s), bc_y, int(12*s), int(10*s)), border_radius=2)
    pygame.draw.rect(surface, (100, 80, 50),
                     (bc_x - int(3*s), bc_y - int(3*s), int(6*s), int(3*s)), border_radius=1)


def draw_bank_scene(surface, scroll_x=0):
    """Draws the bank background scene (desk, teller, decorations)."""
    # Floor
    pygame.draw.rect(surface, (25, 30, 55), (0, HEIGHT - 120, WIDTH, 120))
    # Floor line
    pygame.draw.line(surface, (40, 45, 70), (0, HEIGHT - 120), (WIDTH, HEIGHT - 120), 2)

    # Bank building (right side)
    bldg_x = 750
    pygame.draw.rect(surface, (30, 35, 65), (bldg_x, 100, 400, HEIGHT - 220))
    # Pillars
    for px in [bldg_x + 30, bldg_x + 370]:
        pygame.draw.rect(surface, (50, 55, 85), (px, 130, 25, HEIGHT - 250))
    # Sign
    sign_rect = pygame.Rect(bldg_x + 80, 115, 240, 45)
    pygame.draw.rect(surface, ACCENT_GOLD, sign_rect, border_radius=6)
    sign_txt = FONT_HEADING.render("SURAKSHA BANK", True, TEXT_DARK)
    surface.blit(sign_txt, (sign_rect.centerx - sign_txt.get_width()//2,
                            sign_rect.centery - sign_txt.get_height()//2))

    # Counter / Desk
    desk_y = HEIGHT - 220
    pygame.draw.rect(surface, DESK_COLOR, (bldg_x + 100, desk_y, 200, 30), border_radius=4)
    pygame.draw.rect(surface, (80, 55, 35), (bldg_x + 100, desk_y + 30, 200, 70))

    # Computer monitor on desk
    pygame.draw.rect(surface, (20, 20, 40), (bldg_x + 170, desk_y - 40, 50, 35), border_radius=3)
    pygame.draw.rect(surface, ACCENT_TEAL, (bldg_x + 173, desk_y - 37, 44, 29), border_radius=2)
    pygame.draw.rect(surface, (20, 20, 40), (bldg_x + 190, desk_y - 5, 10, 5))

    # Teller (simple figure behind desk)
    tx = bldg_x + 200
    ty = desk_y - 10
    # Body
    pygame.draw.rect(surface, TELLER_SHIRT, (tx - 12, ty - 45, 24, 30), border_radius=5)
    # Head
    pygame.draw.circle(surface, SKIN_COLOR, (tx, ty - 55), 13)
    # Eyes
    pygame.draw.circle(surface, (30, 30, 50), (tx - 4, ty - 57), 2)
    pygame.draw.circle(surface, (30, 30, 50), (tx + 4, ty - 57), 2)

    # Rope barrier
    for ry in [HEIGHT - 180]:
        for rpx in range(bldg_x + 50, bldg_x + 100, 1):
            pass
        # Posts
        pygame.draw.rect(surface, ACCENT_GOLD, (bldg_x + 50, ry - 40, 6, 40))
        pygame.draw.circle(surface, ACCENT_GOLD, (bldg_x + 53, ry - 42), 5)


def draw_neural_net(surface, cx, cy, t, progress=0.0):
    """
    Draws an animated neural network visualization.
    cx, cy = center position
    t = time for animation
    progress = 0.0 to 1.0 for data flow animation
    """
    layers = [9, 8, 4, 1]  # Network architecture
    layer_spacing = 120
    max_nodes = max(layers)
    node_r = 10

    positions = []  # Store node positions for drawing edges

    for li, n_nodes in enumerate(layers):
        layer_x = cx - (len(layers) - 1) * layer_spacing // 2 + li * layer_spacing
        layer_positions = []
        node_spacing = 32
        start_y = cy - (n_nodes - 1) * node_spacing // 2

        for ni in range(n_nodes):
            ny = start_y + ni * node_spacing
            layer_positions.append((layer_x, ny))
        positions.append(layer_positions)

    # Draw edges (connections between neurons)
    for li in range(len(positions) - 1):
        for (x1, y1) in positions[li]:
            for (x2, y2) in positions[li + 1]:
                # Animated brightness based on data flow
                phase = (progress * len(positions) - li) * 2
                brightness = max(0.15, min(1.0, 0.5 + 0.5 * math.sin(t * 3 + phase)))
                alpha = int(brightness * 120)
                color = (60 + int(brightness * 60), 100 + int(brightness * 80),
                         200 + int(brightness * 55))
                pygame.draw.line(surface, color, (x1, y1), (x2, y2), 1)

    # Draw nodes
    for li, layer in enumerate(positions):
        for ni, (nx, ny) in enumerate(layer):
            # Pulsing effect
            pulse = 1.0 + 0.25 * math.sin(t * 4 + li * 1.5 + ni * 0.5)
            r = int(node_r * pulse)

            # Glow
            glow_surf = pygame.Surface((r * 6, r * 6), pygame.SRCALPHA)
            glow_alpha = int(40 + 30 * math.sin(t * 3 + li + ni))
            pygame.draw.circle(glow_surf, (*NODE_COLOR, glow_alpha), (r * 3, r * 3), r * 3)
            surface.blit(glow_surf, (nx - r * 3, ny - r * 3))

            # Node
            pygame.draw.circle(surface, NODE_COLOR, (nx, ny), r)
            pygame.draw.circle(surface, (200, 230, 255), (nx, ny), r - 3)

    # Layer labels
    labels = ["Input\n(Fuzzy μ)", "Hidden 1\n(8 ReLU)", "Hidden 2\n(4 ReLU)", "Output\n(Decision)"]
    for li, lbl_text in enumerate(labels):
        lx = cx - (len(layers) - 1) * layer_spacing // 2 + li * layer_spacing
        for i, line in enumerate(lbl_text.split('\n')):
            lbl = FONT_SMALL.render(line, True, TEXT_DIM)
            surface.blit(lbl, (lx - lbl.get_width() // 2,
                               cy + max_nodes * 16 + 10 + i * 18))


def draw_gauge(surface, cx, cy, radius, value, max_val=100):
    """
    Draws a semi-circular credit score gauge.
    value = current score (0 to max_val)
    """
    # Background arc
    start_angle = math.pi * 0.8
    end_angle = math.pi * 0.2 + math.pi * 2

    # Draw arc segments with color gradient (red → yellow → green)
    n_segments = 60
    for i in range(n_segments):
        ratio = i / n_segments
        angle = start_angle + ratio * (end_angle - start_angle)

        # Color gradient: red → yellow → green
        if ratio < 0.4:
            r = 240
            g = int(60 + ratio * 375)
            b = 60
        elif ratio < 0.7:
            sub = (ratio - 0.4) / 0.3
            r = int(240 - sub * 200)
            g = int(210 + sub * 10)
            b = int(60 + sub * 60)
        else:
            sub = (ratio - 0.7) / 0.3
            r = int(40)
            g = int(220 - sub * 10)
            b = int(120)

        x1 = cx + int(radius * math.cos(angle))
        y1 = cy - int(radius * math.sin(angle))
        x2 = cx + int((radius - 18) * math.cos(angle))
        y2 = cy - int((radius - 18) * math.sin(angle))

        pygame.draw.line(surface, (r, g, b), (x1, y1), (x2, y2), 4)

    # Needle
    val_ratio = np.clip(value / max_val, 0, 1)
    needle_angle = start_angle + val_ratio * (end_angle - start_angle)
    nx = cx + int((radius - 30) * math.cos(needle_angle))
    ny = cy - int((radius - 30) * math.sin(needle_angle))
    pygame.draw.line(surface, TEXT_WHITE, (cx, cy), (nx, ny), 3)
    pygame.draw.circle(surface, TEXT_WHITE, (cx, cy), 8)
    pygame.draw.circle(surface, ACCENT_GOLD, (cx, cy), 5)

    # Score text
    score_txt = FONT_GAUGE.render(str(int(value)), True, TEXT_WHITE)
    surface.blit(score_txt, (cx - score_txt.get_width() // 2, cy + 15))

    # Label
    lbl = FONT_BODY.render("Credit Score", True, TEXT_DIM)
    surface.blit(lbl, (cx - lbl.get_width() // 2, cy + 65))

    # Min/Max labels
    min_lbl = FONT_SMALL.render("0", True, TEXT_DIM)
    max_lbl = FONT_SMALL.render("100", True, TEXT_DIM)
    surface.blit(min_lbl, (cx - radius + 5, cy + 5))
    surface.blit(max_lbl, (cx + radius - 30, cy + 5))


def draw_stamp(surface, cx, cy, text, color, alpha, rotation_deg):
    """Draws a rotated stamp overlay with border."""
    stamp_surf = pygame.Surface((400, 160), pygame.SRCALPHA)

    # Border rectangle
    border_color = (*color, min(255, alpha))
    pygame.draw.rect(stamp_surf, border_color, (10, 10, 380, 140), 6, border_radius=8)

    # Text
    txt = FONT_STAMP.render(text, True, (*color, min(255, alpha)))
    stamp_surf.blit(txt, (200 - txt.get_width() // 2, 80 - txt.get_height() // 2))

    # Rotate
    rotated = pygame.transform.rotate(stamp_surf, rotation_deg)
    rect = rotated.get_rect(center=(cx, cy))
    surface.blit(rotated, rect)


def draw_particles(surface, particles, t):
    """Draw floating particle effects."""
    for p in particles:
        px = p['x'] + math.sin(t * p['speed'] + p['phase']) * 20
        py = p['y'] - t * p['speed'] * 5 % HEIGHT
        if py < 0:
            py += HEIGHT
        alpha = int(40 + 20 * math.sin(t * 2 + p['phase']))
        size = p['size']
        ps = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        pygame.draw.circle(ps, (*ACCENT_TEAL, alpha), (size, size), size)
        surface.blit(ps, (int(px) - size, int(py) - size))


# =============================================================================
# MAIN APPLICATION
# =============================================================================

async def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("SURAKSHA — Credit Risk Assessment Simulator")
    clock = pygame.time.Clock()

    # --- Build Fuzzy System & Train MLP ---
    fuzzy_sim, f_income, f_debt, f_age = build_fuzzy_system()
    mlp, scaler = build_mlp(fuzzy_sim)

    # --- State ---
    state = STATE_INPUT
    t = 0.0  # Global time counter

    # --- Input Sliders ---
    slider_x = 100
    slider_w = 400
    income_slider = Slider(slider_x, 280, slider_w, 1000000, 20000000, 8000000,
                           "💰 Annual Income", "₹{:,}", ACCENT_GOLD)
    age_slider    = Slider(slider_x, 380, slider_w, 18, 80, 35,
                           "🎂 Age (Years)", "{} yrs", ACCENT_TEAL)
    debt_slider   = Slider(slider_x, 480, slider_w, 0, 100000, 25000,
                           "📊 Total Debt", "₹{:,}", RED_REJECT)
    sliders = [income_slider, age_slider, debt_slider]

    apply_btn = Button(slider_x + 80, 560, 240, 55, "APPLY FOR LOAN", ACCENT_GOLD, TEXT_DARK)

    # --- Walking State ---
    char_x = 80.0
    char_target_x = 870.0
    walk_speed = 3.0
    leg_phase = 0.0

    # --- Calculating State ---
    calc_timer = 0.0
    calc_duration = 4.0  # seconds

    # --- Result State ---
    result_decision = 0
    result_confidence = 0.0
    result_risk_index = 50.0
    result_score = 0.0
    score_anim = 0.0
    stamp_alpha = 0.0
    stamp_rotation = -25.0
    result_memberships = []
    reset_btn = Button(WIDTH // 2 - 120, HEIGHT - 85, 240, 50,
                       "APPLY AGAIN", ACCENT_BLUE, TEXT_WHITE)

    # --- Particles ---
    particles = [{'x': random.randint(0, WIDTH), 'y': random.randint(0, HEIGHT),
                   'speed': random.uniform(0.3, 1.0), 'phase': random.uniform(0, 6.28),
                   'size': random.randint(2, 5)} for _ in range(30)]

    # =========================================================================
    # MAIN GAME LOOP
    # =========================================================================
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        t += dt

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if state == STATE_INPUT:
                for s in sliders:
                    s.handle_event(event)
                if apply_btn.handle_event(event):
                    # Transition to walking state
                    state = STATE_WALKING
                    char_x = 80.0
                    leg_phase = 0.0

            elif state == STATE_RESULT:
                if reset_btn.handle_event(event):
                    # Reset to input
                    state = STATE_INPUT
                    score_anim = 0.0
                    stamp_alpha = 0.0

        # =====================================================================
        # STATE UPDATES
        # =====================================================================

        if state == STATE_WALKING:
            char_x += walk_speed
            leg_phase += 0.15
            if char_x >= char_target_x:
                char_x = char_target_x
                state = STATE_CALCULATING
                calc_timer = 0.0

                # --- Run Fuzzy + MLP computation ---
                result_risk_index, result_memberships = compute_fuzzy(
                    fuzzy_sim,
                    income_slider.value,
                    debt_slider.value,
                    age_slider.value
                )
                result_decision, result_confidence = predict_loan(
                    mlp, scaler, result_memberships)

                # Compute a credit score (0-100) blending fuzzy risk and MLP confidence
                # Low risk → high score; approved → boost
                base_score = max(0, min(100, 100 - result_risk_index))
                if result_decision == 1:
                    result_score = min(100, base_score * 0.7 + result_confidence * 40)
                else:
                    result_score = max(0, base_score * 0.5 - (1 - result_confidence) * 20)
                result_score = max(5, min(98, result_score))

        elif state == STATE_CALCULATING:
            calc_timer += dt
            if calc_timer >= calc_duration:
                state = STATE_RESULT
                score_anim = 0.0
                stamp_alpha = 0.0
                stamp_rotation = -25.0

        elif state == STATE_RESULT:
            # Animate score gauge filling up
            if score_anim < result_score:
                score_anim = min(result_score, score_anim + 80 * dt)
            # Animate stamp fade in after gauge fills
            if score_anim >= result_score * 0.9:
                stamp_alpha = min(255, stamp_alpha + 300 * dt)
                # Settle rotation
                if stamp_rotation < -12:
                    stamp_rotation += 40 * dt

        # =====================================================================
        # RENDERING
        # =====================================================================
        screen.fill(BG_DARK)

        # --- Floating particles (all states) ---
        draw_particles(screen, particles, t)

        # --- Header bar ---
        header_rect = pygame.Rect(0, 0, WIDTH, 60)
        pygame.draw.rect(screen, BG_PANEL, header_rect)
        pygame.draw.line(screen, ACCENT_GOLD, (0, 60), (WIDTH, 60), 2)

        title = FONT_HEADING.render("SURAKSHA", True, ACCENT_GOLD)
        screen.blit(title, (20, 14))
        subtitle = FONT_SMALL.render("Credit Risk Assessment Simulator  |  Fuzzy Logic + Neural Network",
                                     True, TEXT_DIM)
        screen.blit(subtitle, (180, 22))

        # --- State-specific rendering ---

        if state == STATE_INPUT:
            # Panel background
            panel = pygame.Rect(60, 100, 500, 550)
            pygame.draw.rect(screen, BG_CARD, panel, border_radius=16)
            pygame.draw.rect(screen, (50, 55, 90), panel, 1, border_radius=16)

            # Panel title
            ptitle = FONT_HEADING.render("Loan Application", True, TEXT_WHITE)
            screen.blit(ptitle, (panel.x + 30, panel.y + 20))
            pdesc = FONT_SMALL.render("Adjust the sliders below and submit your application",
                                      True, TEXT_DIM)
            screen.blit(pdesc, (panel.x + 30, panel.y + 55))

            # Divider
            pygame.draw.line(screen, (50, 55, 90),
                             (panel.x + 20, panel.y + 85), (panel.right - 20, panel.y + 85), 1)

            # Draw sliders
            for s in sliders:
                s.draw(screen)

            # Apply button
            apply_btn.draw(screen)

            # --- Right side: Info panel ---
            info_x = 620
            info_panel = pygame.Rect(info_x, 100, 520, 550)
            pygame.draw.rect(screen, BG_CARD, info_panel, border_radius=16)
            pygame.draw.rect(screen, (50, 55, 90), info_panel, 1, border_radius=16)

            info_title = FONT_HEADING.render("How It Works", True, ACCENT_TEAL)
            screen.blit(info_title, (info_x + 30, 125))

            steps = [
                ("1. FUZZY ANALYSIS", "Your inputs are evaluated using fuzzy",
                 "membership functions (Low/Med/High)."),
                ("2. RISK ASSESSMENT", "A Fuzzy Inference System combines",
                 "rules to compute a Preliminary Risk Index."),
                ("3. NEURAL NETWORK", "An MLP trained on the fuzzy membership",
                 "values makes the final Approve/Deny decision."),
                ("4. CREDIT SCORE", "Both fuzzy risk and MLP confidence are",
                 "blended into a 0-100 Credit Score."),
            ]
            for i, (heading, line1, line2) in enumerate(steps):
                sy = 180 + i * 110
                # Step number accent
                pygame.draw.rect(screen, ACCENT_TEAL,
                                 (info_x + 30, sy, 4, 70), border_radius=2)
                h = FONT_LABEL.render(heading, True, ACCENT_GOLD)
                screen.blit(h, (info_x + 50, sy + 5))
                l1 = FONT_SMALL.render(line1, True, TEXT_DIM)
                screen.blit(l1, (info_x + 50, sy + 30))
                l2 = FONT_SMALL.render(line2, True, TEXT_DIM)
                screen.blit(l2, (info_x + 50, sy + 48))

        elif state == STATE_WALKING:
            draw_bank_scene(screen)
            draw_character(screen, int(char_x), HEIGHT - 130, scale=1.2, leg_phase=leg_phase)

            # Status text
            dots = "." * (int(t * 2) % 4)
            walk_txt = FONT_BODY.render(f"Walking to the bank{dots}", True, ACCENT_GOLD)
            screen.blit(walk_txt, (WIDTH // 2 - walk_txt.get_width() // 2, HEIGHT - 60))

        elif state == STATE_CALCULATING:
            # Semi-transparent overlay for bank scene
            draw_bank_scene(screen)
            draw_character(screen, int(char_x), HEIGHT - 130, scale=1.2, leg_phase=0)

            # Dark overlay
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((12, 15, 28, 200))
            screen.blit(overlay, (0, 0))

            # Title
            calc_title = FONT_HEADING.render("Processing Your Application...", True, ACCENT_GOLD)
            screen.blit(calc_title, (WIDTH // 2 - calc_title.get_width() // 2, 80))

            # Neural net animation
            progress = calc_timer / calc_duration
            draw_neural_net(screen, WIDTH // 2, HEIGHT // 2 - 20, t, progress)

            # Progress bar
            bar_w = 400
            bar_h = 8
            bar_x = WIDTH // 2 - bar_w // 2
            bar_y = HEIGHT - 80
            pygame.draw.rect(screen, SLIDER_TRACK, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
            fill_w = int(bar_w * progress)
            if fill_w > 0:
                pygame.draw.rect(screen, ACCENT_TEAL,
                                 (bar_x, bar_y, fill_w, bar_h), border_radius=4)

            # Percentage
            pct_txt = FONT_BODY.render(f"{int(progress * 100)}%", True, TEXT_WHITE)
            screen.blit(pct_txt, (WIDTH // 2 - pct_txt.get_width() // 2, bar_y - 30))

        elif state == STATE_RESULT:
            # Result panel
            panel = pygame.Rect(30, 75, WIDTH - 60, HEIGHT - 120)
            pygame.draw.rect(screen, BG_CARD, panel, border_radius=20)
            pygame.draw.rect(screen, (50, 55, 90), panel, 1, border_radius=20)

            # ── LEFT COLUMN: Gauge + Application Summary ──
            left_cx = 230  # center x for left column

            # Gauge (smaller, positioned higher)
            gauge_cy = 230
            draw_gauge(screen, left_cx, gauge_cy, 110, score_anim)

            # Risk Index & Confidence (below gauge with clear spacing)
            info_y = gauge_cy + 90
            risk_txt = FONT_SMALL.render(f"Fuzzy Risk Index: {result_risk_index:.1f}",
                                         True, TEXT_DIM)
            screen.blit(risk_txt, (left_cx - risk_txt.get_width() // 2, info_y))

            conf_txt = FONT_SMALL.render(f"MLP Confidence: {result_confidence:.1%}",
                                         True, TEXT_DIM)
            screen.blit(conf_txt, (left_cx - conf_txt.get_width() // 2, info_y + 22))

            # Application summary
            summary_y = info_y + 58
            inp_title = FONT_LABEL.render("Your Application:", True, TEXT_WHITE)
            screen.blit(inp_title, (left_cx - 100, summary_y))
            inp_items = [
                f"Income: Rs.{int(income_slider.value):,}",
                f"Age: {int(age_slider.value)} years",
                f"Debt: Rs.{int(debt_slider.value):,}",
            ]
            for i, item in enumerate(inp_items):
                itxt = FONT_SMALL.render(item, True, TEXT_DIM)
                screen.blit(itxt, (left_cx - 90, summary_y + 26 + i * 20))

            # ── Vertical Divider ──
            div_x = 430
            pygame.draw.line(screen, (50, 55, 90), (div_x, 100), (div_x, HEIGHT - 70), 1)

            # ── RIGHT COLUMN: Decision + Stamp + Explanation ──
            right_cx = (div_x + WIDTH - 30) // 2  # centered in right area

            # Decision heading
            dec_heading = FONT_HEADING.render("Loan Decision", True, TEXT_WHITE)
            screen.blit(dec_heading, (right_cx - dec_heading.get_width() // 2, 95))

            # Stamp
            stamp_cy = 235
            if stamp_alpha > 0:
                if result_decision == 1:
                    draw_stamp(screen, right_cx, stamp_cy,
                               "APPROVED", GREEN_APPROVE,
                               int(stamp_alpha), stamp_rotation)
                else:
                    draw_stamp(screen, right_cx, stamp_cy,
                               "REJECTED", RED_REJECT,
                               int(stamp_alpha), stamp_rotation)

            # Explanation text
            if stamp_alpha > 200:
                exp_y = 340
                if result_decision == 1:
                    exp_lines = [
                        "Congratulations! Your loan has been approved.",
                        f"Your credit profile scored {int(result_score)}/100.",
                        "The fuzzy analysis rated your risk as manageable,",
                        "and the neural network confirmed eligibility."
                    ]
                    exp_color = GREEN_APPROVE
                else:
                    exp_lines = [
                        "We're sorry, your loan application was denied.",
                        f"Your credit profile scored {int(result_score)}/100.",
                        "The risk assessment indicated concerns with your",
                        "debt-to-income ratio or credit profile."
                    ]
                    exp_color = RED_REJECT

                for i, line in enumerate(exp_lines):
                    clr = exp_color if i == 0 else TEXT_DIM
                    ltxt = FONT_SMALL.render(line, True, clr)
                    screen.blit(ltxt, (right_cx - ltxt.get_width() // 2, exp_y + i * 22))

            # ── BOTTOM: Membership Values (full width, below both columns) ──
            mv_y = 480
            # Horizontal divider
            pygame.draw.line(screen, (50, 55, 90), (60, mv_y - 10), (WIDTH - 60, mv_y - 10), 1)

            mv_title = FONT_LABEL.render("Fuzzy Membership Values (MLP Input Features):",
                                         True, TEXT_WHITE)
            screen.blit(mv_title, (WIDTH // 2 - mv_title.get_width() // 2, mv_y))

            mv_labels = ["Income Low", "Income Med", "Income High",
                         "Debt Safe",  "Debt Mod",   "Debt Risky",
                         "Age Young",  "Age Middle", "Age Senior"]

            # 3 columns across the full panel width
            col_w = 340
            col_start_x = 80
            bar_h = 12
            bar_w = 120
            for i, (label, val) in enumerate(zip(mv_labels, result_memberships)):
                col = i % 3
                row = i // 3
                bx = col_start_x + col * col_w
                by = mv_y + 30 + row * 32

                # Label
                lt = FONT_SMALL.render(f"{label}:", True, TEXT_DIM)
                screen.blit(lt, (bx, by))

                # Bar background
                bar_x = bx + 100
                pygame.draw.rect(screen, SLIDER_TRACK,
                                 (bar_x, by + 3, bar_w, bar_h), border_radius=3)
                # Bar fill
                fw = int(bar_w * val)
                if fw > 0:
                    pygame.draw.rect(screen, ACCENT_TEAL,
                                     (bar_x, by + 3, fw, bar_h), border_radius=3)
                # Value text
                vt = FONT_SMALL.render(f"{val:.2f}", True, TEXT_DIM)
                screen.blit(vt, (bar_x + bar_w + 8, by))

            # Reset button (positioned at bottom center)
            reset_btn.rect.y = HEIGHT - 75
            reset_btn.rect.x = WIDTH // 2 - reset_btn.rect.w // 2
            reset_btn.draw(screen)

        # --- Footer ---
        footer = FONT_BANNER.render(
            "Soft Computing Project  |  Fuzzy Logic × MLP Neural Network  |  Built with Pygame",
            True, (60, 65, 90))
        screen.blit(footer, (WIDTH // 2 - footer.get_width() // 2, HEIGHT - 25))

        pygame.display.flip()
        await asyncio.sleep(0)

    pygame.quit()
    sys.exit()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    asyncio.run(main())
