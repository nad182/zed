use acp_thread::{ToolCall, ToolCallContent, ToolCallStatus};
use gpui::App;

// A canceled tool call that produced visible output is still worth showing, but one
// canceled before producing anything would just be a useless "Canceled" card.
pub(super) fn tool_call_renders_nothing(tool_call: &ToolCall, cx: &App) -> bool {
    matches!(tool_call.status(), ToolCallStatus::Canceled)
        && !tool_call.content().iter().any(|content| match content {
            ToolCallContent::ContentBlock { block, .. } => block.visible_content(cx),
            ToolCallContent::Diff(_)
            | ToolCallContent::LegacyDiff { .. }
            | ToolCallContent::Terminal { .. } => true,
            ToolCallContent::DiffPatch { render, .. } => {
                !render.files.is_empty()
                    || render
                        .fallback
                        .as_ref()
                        .is_some_and(|markdown| !markdown.read(cx).source().is_empty())
            }
            ToolCallContent::Other { markdown, .. } => !markdown.read(cx).source().is_empty(),
        })
}
