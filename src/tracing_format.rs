use chrono::Utc;
use nu_ansi_term::{Color, Style};
use tracing::Event;
use tracing_log::NormalizeEvent;
use tracing_subscriber::{
    fmt::{
        format::{FormatEvent, FormatFields, Writer},
        FmtContext,
    },
    registry::LookupSpan,
};

pub struct CustomFormat {
    dev_mode: bool,
}

impl CustomFormat {
    pub fn new() -> Self {
        Self {
            dev_mode: std::env::var("DEV").is_ok(),
        }
    }
}

impl<S, N> FormatEvent<S, N> for CustomFormat
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        let dimmed = Style::new().dimmed();

        let normalized_meta = event.normalized_metadata();
        let meta = normalized_meta.as_ref().unwrap_or_else(|| event.metadata());

        let timestamp = Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        write!(writer, "{} ", dimmed.paint(timestamp))?;

        let level_str = match *meta.level() {
            tracing::Level::ERROR => Color::Red.paint("ERROR"),
            tracing::Level::WARN => Color::Yellow.paint("WARN "),
            tracing::Level::INFO => Color::Green.paint("INFO "),
            tracing::Level::DEBUG => Color::Blue.paint("DEBUG"),
            tracing::Level::TRACE => Color::Purple.paint("TRACE"),
        };
        write!(writer, "{} ", level_str)?;

        write!(writer, "{} ", dimmed.paint(meta.target()))?;

        if self.dev_mode
            && let Some(file) = meta.file()
        {
            write!(
                writer,
                "{}:{} ",
                dimmed.paint(file),
                dimmed.paint(meta.line().unwrap_or(0).to_string())
            )?;
        }

        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}
