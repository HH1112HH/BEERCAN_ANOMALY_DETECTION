# BeerCan Inspection — Design Direction

Design handoff for the separately deployed React/Vite dashboard. Adapted from the Open Design `redesign-skill` and `taste-skill-v1` workflow: audit the current product, preserve working API behavior, then refine hierarchy, color, states, and motion.

Source: <https://github.com/nexu-io/open-design/tree/main>

## Direction

- Product: desktop-first industrial quality inspection console, usable on a tablet/phone for quick monitoring.
- Mood: bright, precise, technical, lively-but-calm; a real operations tool rather than a marketing landing page.
- Layout: top navigation, asymmetric live-camera + control rail, separated metric strip, recent inspection log. Avoid a permanent left sidebar and equal three-card rows.
- Density: 6/10. Keep key production signals visible without packing every control into the first screen.

## Visual tokens

- Type: Geist Variable for interface text; Geist Mono for counters, timestamps, IDs, and thresholds. No serif or Inter.
- Base: bright neutral canvas (`#F8FAF9`), white surfaces, charcoal text, and a separate dark camera viewport for image contrast.
- Accent: one restrained technology teal (`#117E72`) for primary actions and active states.
- Semantic states: green for GOOD/connected, muted red for NG/error, amber for waiting; never use color alone to communicate state.
- Surfaces: use spacing and hairline dividers first; elevate only the live-feed/control hierarchy. Avoid default glass cards, purple/blue AI gradients, and excess shadows.

## Motion and states

- Motion intensity: 5/10. Use short spring entrance/press feedback, a restrained live indicator, and transform/opacity transitions only.
- Respect `prefers-reduced-motion`; no perpetual movement that could distract an operator.
- Every network-backed area needs loading skeleton, empty state, stale/offline state, and inline error feedback.
- Keep the camera frame as the visual anchor; overlays must not hide the inspected product.
- Use semantic HTML, keyboard-visible focus, accessible names for icon buttons, and responsive collapse below tablet width.

## Product behavior

- Start/Pause controls describe AI inspection processing, not direct motor power.
- Only show live/connected when API status reports it; show an update/stale state when polling fails.
- Threshold changes are sent to the API and display the current value.
- Keep future algorithm selection out of the UI until the backend exposes a real selection API.
