use std::ops::Range;

use acp_thread::{
    AcpThread, AgentThreadEntry, AssistantMessageChunk, ElicitationStatus, ToolCall,
    ToolCallContent, ToolCallStatus,
};
use agent_client_protocol::schema::v1 as acp_v1;
use collections::HashSet;
use gpui::App;

use super::elicitation::should_render_elicitation;

const MIN_TOOL_CALLS_PER_GROUP: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EntryKind {
    UserMessage,
    Answer,
    Thinking,
    ToolCall { groupable: bool, renders: bool },
    NeedsAction,
    Aside,
    Invisible,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CollapseKey {
    ToolCalls(acp_v1::ToolCallId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Header {
    ToolCallGroup {
        key: CollapseKey,
        count: usize,
        is_expanded: bool,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Content {
    #[default]
    Full,
    Hidden,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EntryPresentation {
    pub header: Option<Header>,
    pub content: Content,
}

impl EntryPresentation {
    pub(crate) fn is_full(&self) -> bool {
        self.header.is_none() && self.content == Content::Full
    }
}

pub(crate) struct LayoutInput<'a> {
    pub kinds: &'a [EntryKind],
    pub tool_call_ids: &'a [Option<acp_v1::ToolCallId>],
    pub tail_is_growing: bool,
    pub expanded: &'a HashSet<CollapseKey>,
}

pub(crate) fn layout(input: LayoutInput) -> Vec<EntryPresentation> {
    let mut presentation = vec![EntryPresentation::default(); input.kinds.len()];
    for run in tool_call_runs(input.kinds, input.tail_is_growing) {
        let Some(Some(first_tool_call_id)) = input.tool_call_ids.get(run.entries.start) else {
            continue;
        };
        let key = CollapseKey::ToolCalls(first_tool_call_id.clone());
        let is_expanded = input.expanded.contains(&key);
        let content = if is_expanded {
            Content::Full
        } else {
            Content::Hidden
        };
        for entry in &mut presentation[run.entries.clone()] {
            entry.content = content;
        }
        presentation[run.entries.start].header = Some(Header::ToolCallGroup {
            key,
            count: run.tool_call_count,
            is_expanded,
        });
    }
    presentation
}

struct ToolCallRun {
    entries: Range<usize>,
    tool_call_count: usize,
}

// While the thread is generating, the latest tool call of the trailing run stays out of
// the group so its result remains visible until the next entry arrives.
fn tool_call_runs(kinds: &[EntryKind], tail_is_growing: bool) -> Vec<ToolCallRun> {
    let mut runs = Vec::new();
    let mut members = Vec::new();
    for (entry_ix, kind) in kinds.iter().enumerate() {
        match kind {
            EntryKind::Invisible | EntryKind::ToolCall { renders: false, .. } => {}
            EntryKind::ToolCall {
                groupable: true, ..
            } => members.push(entry_ix),
            _ => {
                runs.extend(run_from_members(&members));
                members.clear();
            }
        }
    }
    if tail_is_growing {
        members.pop();
    }
    runs.extend(run_from_members(&members));
    runs
}

fn run_from_members(members: &[usize]) -> Option<ToolCallRun> {
    if members.len() < MIN_TOOL_CALLS_PER_GROUP {
        return None;
    }
    let first = *members.first()?;
    let last = *members.last()?;
    Some(ToolCallRun {
        entries: first..last + 1,
        tool_call_count: members.len(),
    })
}

pub(crate) fn entry_kind(entry: &AgentThreadEntry, thread: &AcpThread, cx: &App) -> EntryKind {
    match entry {
        AgentThreadEntry::UserMessage(_) => EntryKind::UserMessage,
        AgentThreadEntry::AssistantMessage(message) => {
            let mut has_text = false;
            let mut has_thoughts = false;
            for chunk in &message.chunks {
                match chunk {
                    AssistantMessageChunk::Message { block, .. } => {
                        has_text |= block.visible_content(cx);
                    }
                    AssistantMessageChunk::Thought { block, .. } => {
                        has_thoughts |= block.visible_content(cx);
                    }
                }
            }
            if has_text {
                EntryKind::Answer
            } else if has_thoughts {
                EntryKind::Thinking
            } else {
                EntryKind::Invisible
            }
        }
        AgentThreadEntry::ToolCall(tool_call) if tool_call.authorization().is_some() => {
            EntryKind::NeedsAction
        }
        AgentThreadEntry::ToolCall(tool_call) => EntryKind::ToolCall {
            groupable: tool_call_is_groupable(tool_call),
            renders: !tool_call_renders_nothing(tool_call, cx),
        },
        AgentThreadEntry::Elicitation(elicitation_id) => match thread.elicitation(elicitation_id) {
            Some((_, elicitation))
                if matches!(elicitation.status, ElicitationStatus::Pending { .. }) =>
            {
                EntryKind::NeedsAction
            }
            Some((_, elicitation)) if should_render_elicitation(elicitation) => EntryKind::Aside,
            _ => EntryKind::Invisible,
        },
        AgentThreadEntry::ContextCompaction(_) => EntryKind::Aside,
    }
}

fn tool_call_is_groupable(tool_call: &ToolCall) -> bool {
    let is_finished = matches!(
        tool_call.status(),
        ToolCallStatus::Completed
            | ToolCallStatus::Failed
            | ToolCallStatus::Rejected
            | ToolCallStatus::Canceled
    );
    is_finished && !tool_call.is_subagent() && !tool_call.shows_diff()
}

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

#[cfg(test)]
mod tests {
    use super::EntryKind::{Answer, Aside, Invisible, NeedsAction, Thinking, UserMessage};
    use super::*;

    const TOOL: EntryKind = EntryKind::ToolCall {
        groupable: true,
        renders: true,
    };
    const DIFF: EntryKind = EntryKind::ToolCall {
        groupable: false,
        renders: true,
    };
    const EMPTY_TOOL: EntryKind = EntryKind::ToolCall {
        groupable: true,
        renders: false,
    };

    fn tool_call_id(entry_ix: usize) -> acp_v1::ToolCallId {
        acp_v1::ToolCallId::new(format!("tool-{entry_ix}"))
    }

    fn layout_of(
        kinds: &[EntryKind],
        tail_is_growing: bool,
        expanded: &HashSet<CollapseKey>,
    ) -> Vec<EntryPresentation> {
        let tool_call_ids = kinds
            .iter()
            .enumerate()
            .map(|(entry_ix, kind)| {
                matches!(kind, EntryKind::ToolCall { .. }).then(|| tool_call_id(entry_ix))
            })
            .collect::<Vec<_>>();
        layout(LayoutInput {
            kinds,
            tool_call_ids: &tool_call_ids,
            tail_is_growing,
            expanded,
        })
    }

    fn collapsed(kinds: &[EntryKind], tail_is_growing: bool) -> Vec<EntryPresentation> {
        layout_of(kinds, tail_is_growing, &HashSet::default())
    }

    fn full() -> EntryPresentation {
        EntryPresentation::default()
    }

    fn hidden() -> EntryPresentation {
        EntryPresentation {
            header: None,
            content: Content::Hidden,
        }
    }

    fn group(first_entry_ix: usize, count: usize, is_expanded: bool) -> EntryPresentation {
        EntryPresentation {
            header: Some(Header::ToolCallGroup {
                key: CollapseKey::ToolCalls(tool_call_id(first_entry_ix)),
                count,
                is_expanded,
            }),
            content: if is_expanded {
                Content::Full
            } else {
                Content::Hidden
            },
        }
    }

    #[test]
    fn groups_consecutive_finished_tool_calls_between_barriers() {
        let kinds = [Answer, TOOL, TOOL, TOOL, Answer, TOOL, Answer];
        assert_eq!(
            collapsed(&kinds, false),
            vec![
                full(),
                group(1, 3, false),
                hidden(),
                hidden(),
                full(),
                full(),
                full()
            ]
        );
    }

    #[test]
    fn ignores_runs_below_minimum_size() {
        let kinds = [Answer, TOOL, Answer, TOOL];
        assert_eq!(collapsed(&kinds, false), vec![full(); 4]);
    }

    #[test]
    fn invisible_entries_do_not_split_runs() {
        let kinds = [Answer, TOOL, Invisible, TOOL, Invisible, Answer];
        assert_eq!(
            collapsed(&kinds, false),
            vec![
                full(),
                group(1, 2, false),
                hidden(),
                hidden(),
                full(),
                full()
            ]
        );
    }

    #[test]
    fn tool_calls_that_render_nothing_neither_join_nor_end_runs() {
        let kinds = [Answer, TOOL, EMPTY_TOOL, TOOL, Answer, EMPTY_TOOL, TOOL];
        assert_eq!(
            collapsed(&kinds, false),
            vec![
                full(),
                group(1, 2, false),
                hidden(),
                hidden(),
                full(),
                full(),
                full()
            ]
        );
    }

    #[test]
    fn diffs_and_other_visible_entries_end_runs() {
        for barrier in [UserMessage, Answer, Thinking, DIFF, NeedsAction, Aside] {
            let kinds = [TOOL, barrier, TOOL];
            assert_eq!(collapsed(&kinds, false), vec![full(); 3], "{barrier:?}");
        }
    }

    #[test]
    fn trailing_run_is_grouped_once_generation_stops() {
        let kinds = [UserMessage, TOOL, TOOL, TOOL];
        assert_eq!(
            collapsed(&kinds, false),
            vec![full(), group(1, 3, false), hidden(), hidden()]
        );
    }

    #[test]
    fn trailing_run_keeps_latest_tool_call_visible_while_generating() {
        let kinds = [UserMessage, Answer, TOOL, TOOL, TOOL, Invisible];
        assert_eq!(
            collapsed(&kinds, true),
            vec![full(), full(), group(2, 2, false), hidden(), full(), full()]
        );

        let kinds = [UserMessage, TOOL, TOOL];
        assert_eq!(collapsed(&kinds, true), vec![full(); 3]);
    }

    #[test]
    fn earlier_runs_are_grouped_while_generating() {
        let kinds = [TOOL, TOOL, Answer, TOOL];
        assert_eq!(
            collapsed(&kinds, true),
            vec![group(0, 2, false), hidden(), full(), full()]
        );
    }

    #[test]
    fn expanded_groups_keep_their_header_and_show_every_entry() {
        let kinds = [Answer, TOOL, TOOL, Answer, TOOL, TOOL];
        let expanded = HashSet::from_iter([CollapseKey::ToolCalls(tool_call_id(4))]);
        assert_eq!(
            layout_of(&kinds, false, &expanded),
            vec![
                full(),
                group(1, 2, false),
                hidden(),
                full(),
                group(4, 2, true),
                full()
            ]
        );
    }
}
