//! Walk Navigation in the window: `Shift+\`` starts it, and while it runs
//! it owns the keyboard, the buttons, the wheel and the pointer. The walking
//! itself is `kreate::walk`'s.

use super::*;
use kreate::walk::{self, Move, Walk};
use winit::window::CursorGrabMode;

impl App {
    pub(super) fn begin_walk(&mut self) {
        if self.walk.is_some() || self.gesture.is_some() || self.render_job.is_some() {
            return;
        }
        self.cancel_all_drags();
        // Out of a scene camera's view: the walk moves Kreate's own view,
        // starting from what the scene camera showed.
        self.camera_view = None;
        self.camera_view_object = None;
        self.motion.interrupt(&self.camera);
        self.walk = Some(Walk::begin(&mut self.camera));
        self.walk_clock = None;
        self.motion.interrupt(&self.camera);
        if let Some(window) = &self.window {
            // Held where it is and hidden, so the mouse turns the head
            // however far it goes. Locked where the platform can, kept
            // inside the window where it cannot.
            let _ = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            window.set_cursor_visible(false);
        }
        self.status(
            "Walk: WASD or arrows move, Q/E down/up, mouse looks, Shift runs, Alt creeps, \
             wheel sets the pace — click or Enter keeps the view, right click or Esc goes back",
        );
        self.redraw.set(RedrawReason::CAMERA_MOVED);
    }

    /// Ends the walk, keeping where it got to or going back.
    pub(super) fn end_walk(&mut self, keep: bool) {
        let Some(walk) = self.walk.take() else {
            return;
        };
        if !keep {
            self.camera = walk.before;
        }
        self.motion.interrupt(&self.camera);
        if let Some(window) = &self.window {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
        self.status(if keep {
            "Walk: view kept"
        } else {
            "Walk cancelled"
        });
        self.redraw.set(RedrawReason::CAMERA_MOVED);
    }

    /// One frame of walking: the keys held, for the time since the last.
    pub(super) fn advance_walk(&mut self) {
        let Some(walk) = &self.walk else { return };
        let now = Instant::now();
        let seconds = self
            .walk_clock
            .map_or(0.0, |t| now.duration_since(t).as_secs_f64().min(0.1));
        self.walk_clock = walk.moving().then_some(now);
        if walk.step(
            &mut self.camera,
            seconds,
            self.input.shift_down,
            self.input.alt_down,
        ) {
            self.motion.interrupt(&self.camera);
            self.redraw.set(RedrawReason::CAMERA_MOVED);
        }
    }

    /// The pointer's raw movement, which turns the head.
    pub(super) fn walk_look(&mut self, dx: f64, dy: f64) {
        if self.walk.is_none() {
            return;
        }
        walk::look(&mut self.camera, dx, dy);
        self.motion.interrupt(&self.camera);
        self.redraw.set(RedrawReason::CAMERA_MOVED);
    }

    /// A window event while walking: true when the walk took it.
    pub(super) fn walk_event(&mut self, event: &WindowEvent) -> bool {
        if self.walk.is_none() {
            return false;
        }
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return true;
                };
                let down = event.state == ElementState::Pressed;
                let way = match code {
                    KeyCode::KeyW | KeyCode::ArrowUp => Some(Move::Forward),
                    KeyCode::KeyS | KeyCode::ArrowDown => Some(Move::Back),
                    KeyCode::KeyA | KeyCode::ArrowLeft => Some(Move::Left),
                    KeyCode::KeyD | KeyCode::ArrowRight => Some(Move::Right),
                    KeyCode::KeyE | KeyCode::PageUp => Some(Move::Up),
                    KeyCode::KeyQ | KeyCode::PageDown => Some(Move::Down),
                    _ => None,
                };
                if let Some(way) = way {
                    if let Some(walk) = &mut self.walk {
                        walk.set(way, down);
                    }
                    self.redraw.set(RedrawReason::CAMERA_MOVED);
                } else if down && !event.repeat {
                    match code {
                        KeyCode::Enter | KeyCode::NumpadEnter | KeyCode::Space => {
                            self.end_walk(true)
                        }
                        KeyCode::Escape => self.end_walk(false),
                        _ => {}
                    }
                }
                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *state == ElementState::Pressed {
                    match button {
                        MouseButton::Left => self.end_walk(true),
                        MouseButton::Right => self.end_walk(false),
                        _ => {}
                    }
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 60.0,
                };
                if let Some(walk) = &mut self.walk {
                    walk.quicken(notches);
                    let pace = walk.pace;
                    self.status(&format!("Walk: {pace:.2} m/s"));
                }
                true
            }
            // Pointer movement is read raw, from the device; the cursor's
            // own position is held and means nothing now.
            WindowEvent::CursorMoved { .. } => true,
            // Focus gone, the keys' releases go elsewhere: kept where it got.
            WindowEvent::Focused(false) => {
                self.end_walk(true);
                false
            }
            _ => false,
        }
    }
}
