use std::{
    ops::*,
    path::PathBuf,
    sync::{Arc, LazyLock},
};

use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::{MouseScrollDelta, WindowEvent},
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
        // gallery
        KeySpec::new("-").unwrap().to_bind(App::row_no_increase).with_mode(Mode::Gallery),
        KeySpec::new("S-=").unwrap().to_bind(App::noop).with_mode(Mode::Gallery), // '+'
        KeySpec::new("=").unwrap().to_bind(App::row_no_decrease).with_mode(Mode::Gallery),
        // single
        KeySpec::new("-").unwrap().to_bind(App::noop).with_mode(Mode::SingleImage),
        KeySpec::new("S-=").unwrap().to_bind(App::noop).with_mode(Mode::SingleImage), // '+'
        KeySpec::new("=").unwrap().to_bind(App::noop).with_mode(Mode::SingleImage),
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

#[derive(Debug, Clone, Copy)]
pub struct CursorPosition {
    pub x: f64,
    pub y: f64,
}

impl CursorPosition {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Convert to normalized device coordinates (-1.0 to 1.0)
    pub fn to_ndc(&self, window_size: &PhysicalSize<u32>) -> (f64, f64) {
        let ndc_x = (self.x / window_size.width as f64) * 2.0 - 1.0;
        let ndc_y = 1.0 - (self.y / window_size.height as f64) * 2.0;
        (ndc_x, ndc_y)
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
    cursor_pos: Option<CursorPosition>,

    // ui state
    mode: Mode,
    /// changed via user input
    /// see: [`Self::ROW_NO_MIN`] and [`Self::ROW_NO_MAX`]
    row_no: u32,
    /// computed from window dimensions and [`Self::row_no`]
    col_no: u32,
    /// - must be bounded within [`Self::images`], which changes in size (during error removal)
    /// - must be dragged by [`Self::row_offset`]
    cursor_idx: usize,
    /// - must be dragged by [`Self::selection_idx`]
    row_offset: usize,
    drag_start_pos: Option<CursorPosition>,

    // switches
    exiting: bool,
}

impl App {
    const ROW_NO_MIN: u32 = 2;
    const ROW_NO_MAX: u32 = 32;

    pub fn new(path_func: fn() -> Vec<PathBuf>) -> Self {
        let image_loader_service = ImageLoaderService::new(0);

        let paths = path_func();
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

        Self {
            state: None,
            images,
            image_loader_service,
            last_window_event: None,
            mode: Mode::Gallery,
            row_no: 3,
            col_no: 1,
            window_size: None,
            modifiers: ModifiersState::empty(),
            cursor_pos: None,
            drag_start_pos: None,
            cursor_idx: 0,
            row_offset: 0,
            exiting,
            sys,
        }
    }

    pub fn exit(&mut self) {
        self.exiting = true;
    }

    pub fn update_cursor_pos(&mut self, pos: PhysicalPosition<f64>) {
        self.cursor_pos = Some(CursorPosition::new(pos.x, pos.y));
    }

    pub fn update_window_size(&mut self, size: PhysicalSize<u32>) {
        self.window_size = Some(size);
        self.col_no_recalc();
        self.bound_cursor_to_grid();
    }

    pub fn toggle_mode(&mut self) {
        todo!();
        // self.mode.toggle();
    }

    pub fn row_no_increase(&mut self) {
        self.row_no = self.row_no.add(1).min(Self::ROW_NO_MAX);
        self.col_no_recalc();
        self.bound_cursor_to_grid();
    }

    pub fn row_no_decrease(&mut self) {
        self.row_no = self.row_no.sub(1).max(Self::ROW_NO_MIN);
        self.col_no_recalc();
        self.bound_cursor_to_grid();
    }

    pub fn row_offset_decrease(&mut self) {
        self.row_offset = self.row_offset.saturating_sub(1);
        self.bound_cursor_to_grid();
    }

    pub fn row_offset_increase(&mut self) {
        let min = (self.images.len() as f64 / self.col_no as f64)
            .sub(self.row_no as f64)
            .ceil()
            .max(0.0) as usize;
        self.row_offset = self.row_offset.saturating_add(1).min(min);
        self.bound_cursor_to_grid();
    }

    pub fn left(&mut self) {
        self.cursor_idx = self.cursor_idx.saturating_sub(1);
        self.ensure_cursor_visible();
    }

    pub fn right(&mut self) {
        self.cursor_idx = self.cursor_idx.add(1).min(self.images.len() - 1);
        self.ensure_cursor_visible();
    }

    pub fn up(&mut self) {
        self.cursor_idx = self.cursor_idx.saturating_sub(self.col_no as usize);
        self.ensure_cursor_visible();
    }

    pub fn down(&mut self) {
        self.cursor_idx = self
            .cursor_idx
            .add(self.col_no as usize)
            .min(self.images.len() - 1);
        self.ensure_cursor_visible();
    }

    /// TODO what happens if window size changes while drag is in progress?
    /// FIX store it as NDC
    pub fn start_drag(&mut self) {
        self.drag_start_pos = self.cursor_pos;
    }

    fn get_drag_offset_ndc(&mut self) -> Option<(f64, f64)> {
        match self {
            Self {
                window_size: Some(window_size),
                drag_start_pos: Some(start),
                cursor_pos: Some(current),
                ..
            } => {
                let (sx, sy) = start.to_ndc(window_size);
                let (cx, cy) = current.to_ndc(window_size);
                Some((cx.sub(sx), cy.sub(sy)))
            }
            _ => None,
        }
    }

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

    fn resize(&self) {
        let images = self.visible_grid_images().iter();
        let size = self.window_size.unwrap();
        self.state.as_ref().unwrap().resize(
            images,
            size.width as f32,
            size.height as f32,
            self.row_no as f32,
            self.col_no as f32,
        );
    }

    fn resize_fresh_images(&mut self) {
        let paths = self.image_loader_service.completed();

        if paths.len() > 0 {
            self.sys.refresh_memory();

            let available = self.sys.available_memory();
            let total = self.sys.total_memory();
            let available_pct = (available as f64 / total as f64) * 100.0;
            tracing::info!(
                "Mem Avail: {} MB ({:.1}%)",
                available / 10u64.pow(6),
                available_pct
            );

            match available {
                x if x < 2u64 * 10u64.pow(9) => {
                    tracing::error!(
                        "Low memory! Only {} MB available. exiting...",
                        available / 10u64.pow(6)
                    );

                    self.exiting = true;
                }
                x if x < 4u64 * 10u64.pow(9) => {
                    tracing::warn!("Low memory! Only {} MB available", available / 10u64.pow(6));
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
        self.images.retain(|x| !err_paths.contains(x.path()));
        if len - self.images.len() > 0 {
            tracing::debug!("Retain removed {} paths", len - self.images.len());
        }

        if self.images.len() == 0 {
            self.exiting = true;
        }

        let non_err_paths: Vec<_> = paths
            .into_iter()
            .filter(|(_, was_err)| !*was_err)
            .map(|(path, _)| path)
            .collect();

        let images = self
            .visible_grid_images()
            .iter()
            .filter(|img| non_err_paths.contains(img.path()));
        let size = self.window_size.unwrap();
        self.state.as_ref().unwrap().resize(
            images,
            size.width as f32,
            size.height as f32,
            self.row_no as f32,
            self.col_no as f32,
        );
    }

    /// placeholder for developing on keymaps
    fn noop(&mut self) {
        todo!()
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        tracing::debug!("Creating window...");
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
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
                self.resize_fresh_images();
                let images = self.visible_grid_images().iter();
                self.state.as_ref().unwrap().render(images);
                self.state.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(size) => {
                tracing::debug!("Resize: {}x{}", size.width, size.height);
                self.state.as_mut().unwrap().resize_surface(size);
                self.update_window_size(size);
                self.resize();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // tracing::error!("Delta: {delta:?}");
                match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        if y > 0.0 {
                            self.row_offset_decrease();
                            self.resize();
                        } else {
                            self.row_offset_increase();
                            self.resize();
                        }
                    }
                    MouseScrollDelta::PixelDelta(_) => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.update_cursor_pos(position);
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
                    self.resize()
                }

                if self.exiting {
                    event_loop.exit();
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
