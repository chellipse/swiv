use std::str::{from_utf8, Utf8Error};

use winit::{
    event::{ElementState, KeyEvent, WindowEvent},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
};

pub trait MappableApp {
    type Mode: Eq + Copy;
    fn event(&self) -> &WindowEvent;
    fn modifiers(&self) -> &ModifiersState;
    fn mode(&self) -> &Self::Mode;
}

#[derive(Debug, Clone)]
pub struct Keymap<A: MappableApp> {
    mode: Option<A::Mode>,
    kc: Option<KeyCode>,
    modifiers: Option<ModifiersState>,
    state: ElementState,
    func: fn(&mut A),
}

impl<A: MappableApp> Keymap<A> {
    pub fn new(func: fn(&mut A)) -> Self {
        Self {
            mode: None,
            kc: None,
            modifiers: None,
            state: ElementState::Pressed,
            func,
        }
    }

    pub fn with_kc(mut self, kc: KeyCode) -> Self {
        self.kc = Some(kc);
        self
    }

    pub fn with_mode(mut self, mode: A::Mode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Overrides the modifier state, so xor things *before* using this
    pub fn with_mods(mut self, modifiers: ModifiersState) -> Self {
        self.modifiers = Some(modifiers);
        self
    }

    #[inline(always)]
    pub fn apply(&self, app: &mut A) -> bool {
        let event = app.event();

        if let Some(mode) = self.mode.as_ref()
            && mode != app.mode()
        {
            return false;
        }

        if let Some(mmods) = &self.modifiers
            && mmods != app.modifiers()
        {
            return false;
        }

        if let Some(mkc) = self.kc.as_ref()
            && !matches!(
                event,
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(kc),
                        ..
                    },
                    ..
                } if kc == mkc && state == &self.state
            )
        {
            return false;
        }

        (self.func)(app);
        true
    }
}

#[derive(Debug, Clone)]
pub enum Error {
    NoNoneWhitespaceChars,
    Decode(Utf8Error),
    UnrecognizedKey(String),
}

#[derive(Debug, Clone)]
pub struct KeySpec {
    kc: KeyCode,
    modifiers: ModifiersState,
}

impl KeySpec {
    pub fn new(s: &str) -> Result<Self, Error> {
        // TODO handle creating multi-key binds, ie: C-x s
        let first = s
            .split_ascii_whitespace()
            .next()
            .ok_or(Error::NoNoneWhitespaceChars)?;

        let bytes = first.as_bytes();

        let mut idx = 0;
        let mut modifiers = ModifiersState::empty();
        while bytes.len() - idx > 1 && bytes[idx + 1] == b'-' {
            match bytes[idx] {
                b'M' => modifiers |= ModifiersState::SUPER,
                b'S' => modifiers |= ModifiersState::SHIFT,
                b'C' => modifiers |= ModifiersState::CONTROL,
                b'A' => modifiers |= ModifiersState::ALT,
                _ => {}
            };
            idx += 2;
        }

        let kc = match from_utf8(&bytes[idx..]).map_err(|e| Error::Decode(e))? {
            "a" => KeyCode::KeyA,
            "b" => KeyCode::KeyB,
            "c" => KeyCode::KeyC,
            "d" => KeyCode::KeyD,
            "e" => KeyCode::KeyE,
            "f" => KeyCode::KeyF,
            "g" => KeyCode::KeyG,
            "h" => KeyCode::KeyH,
            "i" => KeyCode::KeyI,
            "j" => KeyCode::KeyJ,
            "k" => KeyCode::KeyK,
            "l" => KeyCode::KeyL,
            "m" => KeyCode::KeyM,
            "n" => KeyCode::KeyN,
            "o" => KeyCode::KeyO,
            "p" => KeyCode::KeyP,
            "q" => KeyCode::KeyQ,
            "r" => KeyCode::KeyR,
            "s" => KeyCode::KeyS,
            "t" => KeyCode::KeyT,
            "u" => KeyCode::KeyU,
            "v" => KeyCode::KeyV,
            "w" => KeyCode::KeyW,
            "x" => KeyCode::KeyX,
            "y" => KeyCode::KeyY,
            "z" => KeyCode::KeyZ,

            "0" => KeyCode::Digit0,
            "1" => KeyCode::Digit1,
            "2" => KeyCode::Digit2,
            "3" => KeyCode::Digit3,
            "4" => KeyCode::Digit4,
            "5" => KeyCode::Digit5,
            "6" => KeyCode::Digit6,
            "7" => KeyCode::Digit7,
            "8" => KeyCode::Digit8,
            "9" => KeyCode::Digit9,

            "`" => KeyCode::Backquote,
            "-" => KeyCode::Minus,
            "=" => KeyCode::Equal,
            "[" => KeyCode::BracketLeft,
            "]" => KeyCode::BracketRight,
            "\\" => KeyCode::Backslash,
            ";" => KeyCode::Semicolon,
            "'" => KeyCode::Quote,
            "," => KeyCode::Comma,
            "." => KeyCode::Period,
            "/" => KeyCode::Slash,

            "RET" | "return" => KeyCode::Enter,
            "SPC" | "space" => KeyCode::Space,
            "ESC" | "escape" => KeyCode::Escape,
            "TAB" | "tab" => KeyCode::Tab,
            "DEL" | "delete" => KeyCode::Delete,
            "BSPC" | "backspace" => KeyCode::Backspace,

            "up" => KeyCode::ArrowUp,
            "down" => KeyCode::ArrowDown,
            "left" => KeyCode::ArrowLeft,
            "right" => KeyCode::ArrowRight,

            "pageup" => KeyCode::PageUp,
            "pagedown" => KeyCode::PageDown,
            "home" => KeyCode::Home,
            "end" => KeyCode::End,

            "f1" => KeyCode::F1,
            "f2" => KeyCode::F2,
            "f3" => KeyCode::F3,
            "f4" => KeyCode::F4,
            "f5" => KeyCode::F5,
            "f6" => KeyCode::F6,
            "f7" => KeyCode::F7,
            "f8" => KeyCode::F8,
            "f9" => KeyCode::F9,
            "f10" => KeyCode::F10,
            "f11" => KeyCode::F11,
            "f12" => KeyCode::F12,

            "insert" => KeyCode::Insert,
            "capslock" => KeyCode::CapsLock,

            unknown => return Err(Error::UnrecognizedKey(unknown.to_string())),
        };

        Ok(Self { kc, modifiers })
    }

    pub fn to_bind<A: MappableApp>(self, func: fn(&mut A)) -> Keymap<A> {
        Keymap::new(func).with_kc(self.kc).with_mods(self.modifiers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::{KeyCode, ModifiersState};

    #[test]
    fn test_single_modifiers() {
        // Control
        let spec = KeySpec::new("C-a").unwrap();
        assert_eq!(spec.kc, KeyCode::KeyA);
        assert_eq!(spec.modifiers, ModifiersState::CONTROL);

        // Shift
        let spec = KeySpec::new("S-b").unwrap();
        assert_eq!(spec.kc, KeyCode::KeyB);
        assert_eq!(spec.modifiers, ModifiersState::SHIFT);

        // Alt
        let spec = KeySpec::new("A-c").unwrap();
        assert_eq!(spec.kc, KeyCode::KeyC);
        assert_eq!(spec.modifiers, ModifiersState::ALT);

        // Super/Meta
        let spec = KeySpec::new("M-d").unwrap();
        assert_eq!(spec.kc, KeyCode::KeyD);
        assert_eq!(spec.modifiers, ModifiersState::SUPER);
    }

    #[test]
    fn test_multiple_modifiers() {
        // Control + Shift
        let spec = KeySpec::new("C-S-a").unwrap();
        assert_eq!(spec.kc, KeyCode::KeyA);
        assert_eq!(
            spec.modifiers,
            ModifiersState::CONTROL | ModifiersState::SHIFT
        );

        // Control + Alt
        let spec = KeySpec::new("C-A-b").unwrap();
        assert_eq!(spec.kc, KeyCode::KeyB);
        assert_eq!(
            spec.modifiers,
            ModifiersState::CONTROL | ModifiersState::ALT
        );

        // Super + Shift
        let spec = KeySpec::new("M-S-x").unwrap();
        assert_eq!(spec.kc, KeyCode::KeyX);
        assert_eq!(
            spec.modifiers,
            ModifiersState::SUPER | ModifiersState::SHIFT
        );

        // All modifiers
        let spec = KeySpec::new("M-C-S-A-z").unwrap();
        assert_eq!(spec.kc, KeyCode::KeyZ);
        assert_eq!(
            spec.modifiers,
            ModifiersState::SUPER
                | ModifiersState::CONTROL
                | ModifiersState::SHIFT
                | ModifiersState::ALT
        );
    }

    #[test]
    fn test_no_modifiers() {
        let spec = KeySpec::new("a").unwrap();
        assert_eq!(spec.kc, KeyCode::KeyA);
        assert_eq!(spec.modifiers, ModifiersState::empty());

        let spec = KeySpec::new("5").unwrap();
        assert_eq!(spec.kc, KeyCode::Digit5);
        assert_eq!(spec.modifiers, ModifiersState::empty());
    }

    #[test]
    fn test_special_keys_with_modifiers() {
        // Control + Return
        let spec = KeySpec::new("C-RET").unwrap();
        assert_eq!(spec.kc, KeyCode::Enter);
        assert_eq!(spec.modifiers, ModifiersState::CONTROL);

        // Shift + Space
        let spec = KeySpec::new("S-SPC").unwrap();
        assert_eq!(spec.kc, KeyCode::Space);
        assert_eq!(spec.modifiers, ModifiersState::SHIFT);

        // Alt + Escape
        let spec = KeySpec::new("A-ESC").unwrap();
        assert_eq!(spec.kc, KeyCode::Escape);
        assert_eq!(spec.modifiers, ModifiersState::ALT);

        // Control + Shift + Tab
        let spec = KeySpec::new("C-S-TAB").unwrap();
        assert_eq!(spec.kc, KeyCode::Tab);
        assert_eq!(
            spec.modifiers,
            ModifiersState::CONTROL | ModifiersState::SHIFT
        );
    }

    #[test]
    fn test_function_keys() {
        let spec = KeySpec::new("f11").unwrap();
        assert_eq!(spec.kc, KeyCode::F11);
        assert_eq!(spec.modifiers, ModifiersState::empty());

        let spec = KeySpec::new("C-f5").unwrap();
        assert_eq!(spec.kc, KeyCode::F5);
        assert_eq!(spec.modifiers, ModifiersState::CONTROL);

        let spec = KeySpec::new("M-A-f12").unwrap();
        assert_eq!(spec.kc, KeyCode::F12);
        assert_eq!(spec.modifiers, ModifiersState::SUPER | ModifiersState::ALT);
    }

    #[test]
    fn test_digits_with_modifiers() {
        let spec = KeySpec::new("C-1").unwrap();
        assert_eq!(spec.kc, KeyCode::Digit1);
        assert_eq!(spec.modifiers, ModifiersState::CONTROL);

        let spec = KeySpec::new("M-S-9").unwrap();
        assert_eq!(spec.kc, KeyCode::Digit9);
        assert_eq!(
            spec.modifiers,
            ModifiersState::SUPER | ModifiersState::SHIFT
        );
    }

    #[test]
    fn test_symbols_with_modifiers() {
        let spec = KeySpec::new("C-;").unwrap();
        assert_eq!(spec.kc, KeyCode::Semicolon);
        assert_eq!(spec.modifiers, ModifiersState::CONTROL);

        let spec = KeySpec::new("S-[").unwrap();
        assert_eq!(spec.kc, KeyCode::BracketLeft);
        assert_eq!(spec.modifiers, ModifiersState::SHIFT);

        let spec = KeySpec::new("C-/").unwrap();
        assert_eq!(spec.kc, KeyCode::Slash);
        assert_eq!(spec.modifiers, ModifiersState::CONTROL);
    }

    #[test]
    fn test_arrow_keys_with_modifiers() {
        let spec = KeySpec::new("C-up").unwrap();
        assert_eq!(spec.kc, KeyCode::ArrowUp);
        assert_eq!(spec.modifiers, ModifiersState::CONTROL);

        let spec = KeySpec::new("S-A-left").unwrap();
        assert_eq!(spec.kc, KeyCode::ArrowLeft);
        assert_eq!(spec.modifiers, ModifiersState::SHIFT | ModifiersState::ALT);
    }

    #[test]
    fn test_error_cases() {
        // Empty string
        assert!(matches!(
            KeySpec::new(""),
            Err(Error::NoNoneWhitespaceChars)
        ));

        // Only whitespace
        assert!(matches!(
            KeySpec::new("   "),
            Err(Error::NoNoneWhitespaceChars)
        ));

        // Unrecognized key
        assert!(matches!(
            KeySpec::new("C-unknown"),
            Err(Error::UnrecognizedKey(_))
        ));

        assert!(matches!(
            KeySpec::new("notakey"),
            Err(Error::UnrecognizedKey(_))
        ));
    }
}
