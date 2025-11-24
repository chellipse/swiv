use std::{
    ops::*,
    path::PathBuf,
    sync::{Arc, LazyLock},
};

use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::ModifiersState,
    window::{Window, WindowId},
};

use crate::{
    keymap::{KeySpec, Keymap, MappableApp},
    lazy::{ImageLoaderService, LazyImage},
    state::State,
};

static KEYMAPS: LazyLock<Vec<Keymap<App>>> = LazyLock::new(|| {
    // TODO deserialize this from an included toml file
    // HACK no attribute expressions :/
    #[rustfmt::skip]
    let x = vec![
        // universal
        KeySpec::new("q").unwrap().to_bind(App::exit),
        KeySpec::new("ESC").unwrap().to_bind(App::exit),
        KeySpec::new("RET").unwrap().to_bind(App::toggle_mode),
        KeySpec::new("h").unwrap().to_bind(App::left),
        KeySpec::new("left").unwrap().to_bind(App::left),
        KeySpec::new("l").unwrap().to_bind(App::right),
        KeySpec::new("right").unwrap().to_bind(App::right),
        KeySpec::new("j").unwrap().to_bind(App::down),
        KeySpec::new("down").unwrap().to_bind(App::down),
        KeySpec::new("k").unwrap().to_bind(App::up),
        KeySpec::new("up").unwrap().to_bind(App::up),
        KeySpec::new("f").unwrap().to_bind(App::toggle_fullscreen),
        KeySpec::new("g").unwrap().to_bind(App::go_top),
        KeySpec::new("S-g").unwrap().to_bind(App::go_bottom),
        // gallery
        KeySpec::new("-").unwrap().to_bind(App::row_no_increase).with_mode(Mode::Gallery),
        KeySpec::new("S-=").unwrap().to_bind(App::row_no_decrease).with_mode(Mode::Gallery), // '+'
        KeySpec::new("=").unwrap().to_bind(App::row_no_reset).with_mode(Mode::Gallery),
        // single
        KeySpec::new("r").unwrap().to_bind(App::full_reset).with_mode(Mode::SingleImage),
        KeySpec::new("-").unwrap().to_bind(App::zoom_out).with_mode(Mode::SingleImage),
        KeySpec::new("S-=").unwrap().to_bind(App::zoom_in).with_mode(Mode::SingleImage), // '+'
        KeySpec::new("=").unwrap().to_bind(App::zoom_reset).with_mode(Mode::SingleImage),
    ];

    x
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Gallery,
    SingleImage,
}

impl Mode {
    pub fn toggle(&mut self) {
        match self {
            Self::Gallery => *self = Self::SingleImage,
            Self::SingleImage => *self = Self::Gallery,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Convert to normalized device coordinates (-1.0 to 1.0)
    pub fn to_ndc(&self, window_size: &PhysicalSize<u32>) -> Self {
        let ndc_x = (self.x / window_size.width as f64) * 2.0 - 1.0;
        let ndc_y = 1.0 - (self.y / window_size.height as f64) * 2.0;
        Self::new(ndc_x, ndc_y)
    }

    /// Convert from normalized device coordinates (-1.0 to 1.0) back to screen coordinates
    pub fn from_ndc(ndc_x: f64, ndc_y: f64, window_size: &PhysicalSize<u32>) -> Self {
        let x = (ndc_x + 1.0) * 0.5 * window_size.width as f64;
        let y = (1.0 - ndc_y) * 0.5 * window_size.height as f64;
        Self::new(x, y)
    }
}

impl Add for Point2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Point2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl AddAssign for Point2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

#[derive(Debug)]
pub struct App {
    state: Option<State>,
    images: Vec<LazyImage>,
    image_loader_service: ImageLoaderService,
    sys: sysinfo::System,

    /// Store the state from it so it can be accessed by methods triggered
    /// from [`Keymap::apply`] calls
    last_window_event: Option<WindowEvent>,

    // window state
    modifiers: ModifiersState,
    window_size: Option<PhysicalSize<u32>>,
    // last literal screen cursor position
    cursor_pos: Option<Point2>,

    // ui state
    mode: Mode,
    /// - must be bounded within [`Self::images`], which changes in size (during error removal)
    /// - must be dragged by [`Self::row_offset`]
    cursor_idx: usize,

    // gallery mode state
    /// changed via user input
    /// see: [`Self::ROW_NO_MIN`] and [`Self::ROW_NO_MAX`]
    row_no: u32,
    /// computed from window dimensions and [`Self::row_no`]
    col_no: u32,
    /// - must be dragged by [`Self::cursor_idx`]
    row_offset: usize,

    // single image mode state
    drag_start_pos: Option<Point2>,
    drag_total: Point2,
    scale: f32,

    // switches
    exiting: bool,
    exit_msg: Option<String>,
}

impl App {
    const ROW_NO_MIN: u32 = 2;
    const ROW_NO_MAX: u32 = 32;
    const ROW_NO_DEFAULT: u32 = 4;

    pub fn new(paths: Vec<PathBuf>, cursor_idx: Option<usize>) -> Self {
        let image_loader_service = ImageLoaderService::new(0);

        tracing::info!("Path count: {}", paths.len());
        const NO_FIRST_DISPLAYED: usize = 10;
        tracing::debug!(
            "First {NO_FIRST_DISPLAYED} paths: {:?}",
            &paths[..paths.len().saturating_sub(1).min(NO_FIRST_DISPLAYED)]
        );

        let images: Vec<_> = paths
            .into_iter()
            .map(|path| LazyImage::new(path, image_loader_service.clone_sender()))
            .collect();

        let mut exiting = false;
        if images.len() == 0 {
            exiting = true;
        }

        let sys = sysinfo::System::new_all();

        let mode = match cursor_idx {
            Some(_) => Mode::SingleImage,
            None => Mode::Gallery,
        };

        let cursor_idx = cursor_idx.unwrap_or(0);

        Self {
            state: None,
            images,
            image_loader_service,
            last_window_event: None,
            mode,
            row_no: Self::ROW_NO_DEFAULT,
            col_no: 1,
            window_size: None,
            modifiers: ModifiersState::empty(),
            cursor_pos: None,
            drag_start_pos: None,
            drag_total: Point2::new(0.0, 0.0),
            cursor_idx,
            row_offset: 0,
            exiting,
            exit_msg: None,
            sys,
            scale: 1.0,
        }
    }

    pub fn exit(&mut self) {
        self.exiting = true;
    }

    pub fn exit_msg(&mut self, msg: String) {
        self.exit_msg = Some(msg);
        self.exiting = true;
    }

    pub fn toggle_fullscreen(&mut self) {
        self.state.as_ref().unwrap().toggle_fullscreen();
    }

    pub fn toggle_mode(&mut self) {
        self.mode.toggle();
    }

    pub fn row_no_reset(&mut self) {
        self.set_row_no(Self::ROW_NO_DEFAULT);
    }

    pub fn row_no_increase(&mut self) {
        self.set_row_no(self.row_no.add(1).min(Self::ROW_NO_MAX));
    }

    pub fn row_no_decrease(&mut self) {
        self.set_row_no(self.row_no.sub(1).max(Self::ROW_NO_MIN));
    }

    pub fn row_offset_decrease(&mut self) {
        self.set_row_offset(self.row_offset.saturating_sub(1));
    }

    pub fn row_offset_increase(&mut self) {
        let min = (self.images.len() as f64 / self.col_no as f64)
            .sub(self.row_no as f64)
            .ceil()
            .max(0.0) as usize;
        self.set_row_offset(self.row_offset.saturating_add(1).min(min));
    }

    pub fn go_top(&mut self) {
        self.set_cursor_idx(0);
    }

    pub fn go_bottom(&mut self) {
        self.set_cursor_idx(self.images.len() - 1);
    }

    pub fn left(&mut self) {
        self.set_cursor_idx(self.cursor_idx.saturating_sub(1));
    }

    pub fn right(&mut self) {
        self.set_cursor_idx(self.cursor_idx.add(1).min(self.images.len() - 1));
    }

    pub fn up(&mut self) {
        self.set_cursor_idx(self.cursor_idx.saturating_sub(self.col_no as usize));
    }

    pub fn down(&mut self) {
        self.set_cursor_idx(
            self.cursor_idx
                .add(self.col_no as usize)
                .min(self.images.len() - 1),
        );
    }

    pub fn zoom_reset(&mut self) {
        self.scale = 1.0;
    }

    pub fn zoom_in(&mut self) {
        self.scale *= 1.1;
    }

    pub fn zoom_out(&mut self) {
        self.scale *= 0.9;
    }

    pub fn drag_start(&mut self) {
        if let Some(cursor) = &self.cursor_pos
            && let Some(size) = &self.window_size
        {
            self.drag_start_pos = Some(cursor.to_ndc(size));
        }
    }

    pub fn drag_stop(&mut self) {
        if let Some(offset) = &self.get_drag_offset_ndc() {
            self.drag_total += *offset;
        }
        self.drag_start_pos = None;
    }

    pub fn full_reset(&mut self) {
        self.drag_start_pos = None;
        self.drag_total = Point2::default();
        self.zoom_reset();
    }

    pub fn resize(&self) {
        let size = self.window_size.unwrap();

        match self.mode {
            Mode::Gallery => {
                let images = self.visible_grid_images().iter();
                self.state.as_ref().unwrap().resize(
                    images,
                    size.width as f32,
                    size.height as f32,
                    self.row_no as f32,
                    self.col_no as f32,
                    self.get_rel_cursor_idx() as f32,
                );
            }
            Mode::SingleImage => {
                let image = &self.images[self.cursor_idx];
                let mut offset = self.get_drag_offset_ndc().unwrap_or_default();
                offset += self.drag_total;
                self.state.as_ref().unwrap().resize_single_image(
                    image,
                    size.width as f32,
                    size.height as f32,
                    self.scale,
                    offset.x as f32,
                    offset.y as f32,
                );
            }
        }
    }

    pub fn resize_fresh_images(&mut self) {
        let paths = self.image_loader_service.completed();

        if paths.len() > 0 {
            self.sys.refresh_memory();

            let available = self.sys.available_memory();
            let total = self.sys.total_memory();
            let available_pct = (available as f64 / total as f64) * 100.0;

            match available {
                x if x < 2u64 * 10u64.pow(9) => {
                    tracing::error!(
                        "Low memory! Only {} MB available. exiting...",
                        available / 10u64.pow(6)
                    );

                    self.exiting = true;
                }
                x if x < 4u64 * 10u64.pow(9) => {
                    tracing::warn!(
                        "Low memory! Only {} MB available ({available_pct:.1}%)",
                        available / 10u64.pow(6)
                    );
                }
                x if x < 8u64 * 10u64.pow(9) => {
                    tracing::info!(
                        "Low memory! Only {} MB available ({available_pct:.1}%)",
                        available / 10u64.pow(6)
                    );
                }
                _ => {}
            }
        }

        let err_paths: Vec<_> = paths
            .clone()
            .into_iter()
            .filter(|(_, was_err)| *was_err)
            .map(|(path, _)| path)
            .collect();

        let len = self.images.len();
        self.images.retain(|x| {
            if err_paths.contains(x.path()) {
                tracing::trace!("Removing: {:?}", x.path());
                false
            } else {
                true
            }
        });

        let were_images_removed = len - self.images.len() > 0;

        if were_images_removed {
            tracing::debug!("Retain removed {} paths", len - self.images.len());
        }

        if self.images.len() == 0 {
            self.exit_msg("No images left to display. aborting".to_string());
        }

        if self.cursor_idx >= self.images.len() {
            self.set_cursor_idx(self.images.len() - 1);
        }

        let non_err_paths: Vec<_> = paths
            .into_iter()
            .filter(|(_, was_err)| !*was_err)
            .map(|(path, _)| path)
            .collect();

        match self.mode {
            Mode::Gallery => {
                let images = self
                    .visible_grid_images()
                    .iter()
                    .filter(|img| were_images_removed || non_err_paths.contains(img.path()));
                let size = self.window_size.unwrap();
                self.state.as_ref().unwrap().resize(
                    images,
                    size.width as f32,
                    size.height as f32,
                    self.row_no as f32,
                    self.col_no as f32,
                    self.get_rel_cursor_idx() as f32,
                );
            }
            Mode::SingleImage => {
                if non_err_paths.contains(&self.images[self.cursor_idx].path()) {
                    self.resize();
                }
            }
        }
    }

    /// placeholder for developing on keymaps
    pub fn noop(&mut self) {
        todo!()
    }
}

// Setter methods:
impl App {
    pub fn set_cursor_idx(&mut self, value: usize) {
        if self.cursor_idx != value {
            self.cursor_idx = value;
            self.ensure_cursor_visible();
            if let Some(path) = self.images[self.cursor_idx].path().as_os_str().to_str() {
                let title = format!("{}: {}", env!("CARGO_PKG_NAME"), path);
                self.state.as_ref().unwrap().set_title(&title);
            }
        }
    }

    pub fn set_row_no(&mut self, value: u32) {
        if self.row_no != value {
            self.row_no = value;
            self.col_no_recalc();
            self.ensure_cursor_visible();
        }
    }

    pub fn set_row_offset(&mut self, value: usize) {
        if self.row_offset != value {
            self.row_offset = value;
            self.bound_cursor_to_grid();
        }
    }

    pub fn set_cursor_pos(&mut self, pos: PhysicalPosition<f64>) {
        self.cursor_pos = Some(Point2::new(pos.x, pos.y));
        if self.mode == Mode::SingleImage && self.drag_start_pos.is_some() {
            self.resize()
        }
    }

    pub fn set_window_size(&mut self, size: PhysicalSize<u32>) {
        self.window_size = Some(size);
        self.col_no_recalc();
        self.ensure_cursor_visible();
    }
}

// Recalculation methods:
// cannot use setter methods due to recursion, these should only be called from
// within setters
impl App {
    fn col_no_recalc(&mut self) {
        if let Some(size) = self.window_size {
            let grid_height = size.height / self.row_no;
            self.col_no = size.width / grid_height;
        }
    }

    /// Bounds cursor_idx to be within the current visible grid
    /// based on row_no, col_no, and row_offset.
    /// This ensures the cursor doesn't exceed the grid boundaries.
    fn bound_cursor_to_grid(&mut self) {
        let grid_size = (self.row_no * self.col_no) as usize;
        let grid_start = self.row_offset * self.col_no as usize;
        let grid_end = (grid_start + grid_size).min(self.images.len());

        if grid_start > grid_end.saturating_sub(1) {
            self.cursor_idx = self.images.len().saturating_sub(1);
        } else {
            self.cursor_idx = self
                .cursor_idx
                .clamp(grid_start, grid_end.saturating_sub(1));
        }
    }

    /// Updates row_offset to ensure the current cursor_idx is visible
    /// in the grid. If cursor_idx is already visible, row_offset is unchanged.
    fn ensure_cursor_visible(&mut self) {
        if self.col_no == 0 {
            return;
        }

        let cursor_row = self.cursor_idx / self.col_no as usize;
        let visible_start_row = self.row_offset;
        let visible_end_row = self.row_offset + self.row_no as usize;

        if cursor_row < visible_start_row {
            // Cursor is above visible area, scroll up
            self.row_offset = cursor_row;
        } else if cursor_row >= visible_end_row {
            // Cursor is below visible area, scroll down
            self.row_offset = cursor_row.saturating_sub(self.row_no as usize - 1);
        }
    }
}

// Helper methods:
impl App {
    fn get_rel_cursor_idx(&self) -> usize {
        self.cursor_idx - (self.row_offset * self.col_no as usize)
    }

    fn get_drag_offset_ndc(&self) -> Option<Point2> {
        match self {
            Self {
                window_size: Some(window_size),
                drag_start_pos: Some(start),
                cursor_pos: Some(current),
                ..
            } => {
                let current_ndc = current.to_ndc(window_size);
                Some(current_ndc - *start)
            }
            _ => None,
        }
    }

    pub fn visible_grid_images(&self) -> &[LazyImage] {
        if self.col_no == 0 || self.row_no == 0 {
            return &mut [];
        }

        let grid_size = (self.row_no * self.col_no) as usize;
        let start_idx = self.row_offset * self.col_no as usize;
        let end_idx = (start_idx + grid_size).min(self.images.len());

        if start_idx >= self.images.len() {
            return &mut [];
        }

        self.images[start_idx..end_idx]
            .iter()
            .enumerate()
            .for_each(|(val, img)| img.set_pos(val));

        &self.images[start_idx..end_idx]
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        tracing::debug!("Creating window...");
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title(env!("CARGO_PKG_NAME")))
                .unwrap(),
        );

        let state = pollster::block_on(State::new(window.clone()));
        self.state = Some(state);

        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                tracing::debug!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if self.exiting {
                    if let Some(msg) = &self.exit_msg {
                        println!("{}: {msg}", env!("CARGO_PKG_NAME"));
                    }
                    std::process::exit(0);
                }

                self.resize_fresh_images();

                match self.mode {
                    Mode::Gallery => {
                        let images = self.visible_grid_images().iter();
                        self.state.as_ref().unwrap().render(images, true);
                    }
                    Mode::SingleImage => {
                        let image = &self.images[self.cursor_idx];
                        self.state
                            .as_ref()
                            .unwrap()
                            .render(std::iter::once(image), false);
                    }
                }

                self.state.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(size) => {
                tracing::debug!("Resize: {}x{}", size.width, size.height);
                self.state.as_mut().unwrap().resize_surface(size);
                self.set_window_size(size);
                self.resize();
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => match state {
                    ElementState::Pressed => self.drag_start(),
                    ElementState::Released => self.drag_stop(),
                },
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } => {
                // tracing::debug!("Delta: {delta:?}");
                let mut need_resize = true;
                match (self.mode, delta) {
                    (Mode::Gallery, MouseScrollDelta::LineDelta(_, y)) if y.is_sign_positive() => {
                        self.row_offset_decrease();
                    }
                    (Mode::Gallery, MouseScrollDelta::LineDelta(_, y)) if y.is_sign_negative() => {
                        self.row_offset_increase();
                    }
                    (Mode::SingleImage, MouseScrollDelta::LineDelta(_, y))
                        if y.is_sign_positive() =>
                    {
                        self.zoom_in();
                    }
                    (Mode::SingleImage, MouseScrollDelta::LineDelta(_, y))
                        if y.is_sign_negative() =>
                    {
                        self.zoom_out();
                    }
                    (Mode::SingleImage, MouseScrollDelta::PixelDelta(pos)) => {
                        let delta = pos.y as f32 / self.window_size.unwrap().height as f32;
                        self.scale += delta;
                    }
                    _ => need_resize = false,
                }

                if need_resize {
                    self.resize();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.set_cursor_pos(position);
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }
            _ => {
                self.last_window_event = Some(event);

                let mut mutated = false;

                for keymap in KEYMAPS.iter() {
                    if keymap.apply(self) {
                        mutated = true;
                    };
                }

                if mutated {
                    self.resize();
                }
            }
        }
    }
}

impl MappableApp for App {
    type Mode = Mode;

    fn event(&self) -> &WindowEvent {
        self.last_window_event.as_ref().unwrap()
    }

    fn modifiers(&self) -> &ModifiersState {
        &self.modifiers
    }

    fn mode(&self) -> &Mode {
        &self.mode
    }
}
