use std::{ops::Range, time::Duration};

use acp_thread::{
    AcpThread, AgentThreadEntry, AssistantMessageChunk, ElicitationStatus, ToolCall,
    ToolCallContent, ToolCallStatus,
};
use agent_client_protocol::schema::v1 as acp_v1;
use collections::{HashMap, HashSet};
use gpui::App;
use util::time::duration_alt_display;

use super::STOPWATCH_THRESHOLD;
use super::elicitation::should_render_elicitation;
use super::turn_lifecycle::{TurnOutcome, TurnRecord};
use crate::completion_provider::pluralize;

const MIN_TOOL_CALLS_PER_GROUP: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EntryKind {
    UserMessage,
    Answer { thinking_blocks: usize },
    Thinking { blocks: usize },
    ToolCall { groupable: bool, renders: bool },
    NeedsAction,
    Aside,
    Invisible,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CollapseKey {
    Turn(usize),
    ToolCalls(acp_v1::ToolCallId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Header {
    TurnSummary {
        key: CollapseKey,
        steps: usize,
        duration: Option<Duration>,
        is_expanded: bool,
    },
    ToolCallGroup {
        key: CollapseKey,
        count: usize,
        is_expanded: bool,
    },
}

impl Header {
    pub(crate) fn label(&self) -> String {
        match self {
            Header::TurnSummary {
                steps, duration, ..
            } => turn_summary_label(*steps, *duration),
            Header::ToolCallGroup { count, .. } => {
                format!("{count} {}", pluralize("tool call", *count))
            }
        }
    }
}

fn turn_summary_label(steps: usize, duration: Option<Duration>) -> String {
    match duration {
        Some(duration) if duration > STOPWATCH_THRESHOLD => {
            format!("Worked for {}", duration_alt_display(duration))
        }
        _ if steps > 0 => format!("Worked · {steps} {}", pluralize("step", steps)),
        _ => "Worked".to_string(),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Content {
    #[default]
    Full,
    Hidden,
    AnswerWithoutThoughts,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EntryPresentation {
    pub headers: Vec<Header>,
    pub content: Content,
}

impl EntryPresentation {
    pub(crate) fn is_full(&self) -> bool {
        self.headers.is_empty() && self.content == Content::Full
    }
}

pub(crate) struct LayoutInput<'a> {
    pub kinds: &'a [EntryKind],
    pub tool_call_ids: &'a [Option<acp_v1::ToolCallId>],
    pub tail_is_growing: bool,
    pub last_turn_is_live: bool,
    pub turn_records: &'a HashMap<usize, TurnRecord>,
    pub expanded: &'a HashSet<CollapseKey>,
}

pub(crate) fn layout(input: LayoutInput) -> Vec<EntryPresentation> {
    let mut presentation = vec![EntryPresentation::default(); input.kinds.len()];
    for turn in collapsible_turns(input.kinds, input.last_turn_is_live) {
        let record = input.turn_records.get(&turn.user_message_ix);
        if record.is_some_and(|record| record.outcome == TurnOutcome::Interrupted) {
            continue;
        }
        let key = CollapseKey::Turn(turn.user_message_ix);
        let is_expanded = input.expanded.contains(&key);
        presentation[turn.user_message_ix + 1]
            .headers
            .push(Header::TurnSummary {
                key,
                steps: turn.steps,
                duration: record.and_then(|record| record.duration),
                is_expanded,
            });
        if !is_expanded {
            for entry in &mut presentation[turn.work.clone()] {
                entry.content = Content::Hidden;
            }
            presentation[turn.final_answer_ix].content = Content::AnswerWithoutThoughts;
        }
    }

    for run in tool_call_runs(input.kinds, input.tail_is_growing) {
        if presentation[run.entries.start].content == Content::Hidden {
            continue;
        }
        let Some(Some(first_tool_call_id)) = input.tool_call_ids.get(run.entries.start) else {
            continue;
        };
        let key = CollapseKey::ToolCalls(first_tool_call_id.clone());
        let is_expanded = input.expanded.contains(&key);
        if !is_expanded {
            for entry in &mut presentation[run.entries.clone()] {
                entry.content = Content::Hidden;
            }
        }
        presentation[run.entries.start]
            .headers
            .push(Header::ToolCallGroup {
                key,
                count: run.tool_call_count,
                is_expanded,
            });
    }
    presentation
}

struct CollapsibleTurn {
    user_message_ix: usize,
    work: Range<usize>,
    final_answer_ix: usize,
    steps: usize,
}

fn collapsible_turns(kinds: &[EntryKind], last_turn_is_live: bool) -> Vec<CollapsibleTurn> {
    let user_message_indices = kinds
        .iter()
        .enumerate()
        .filter(|(_, kind)| **kind == EntryKind::UserMessage)
        .map(|(entry_ix, _)| entry_ix)
        .collect::<Vec<_>>();
    user_message_indices
        .iter()
        .enumerate()
        .filter_map(|(turn_ix, &user_message_ix)| {
            let next_user_message_ix = user_message_indices.get(turn_ix + 1).copied();
            if next_user_message_ix.is_none() && last_turn_is_live {
                return None;
            }
            let body = user_message_ix + 1..next_user_message_ix.unwrap_or(kinds.len());
            collapsible_turn(kinds, user_message_ix, body)
        })
        .collect()
}

// Turns that end on work (a tool call, or thinking after the last answer) stay open, so
// how they ended stays visible.
fn collapsible_turn(
    kinds: &[EntryKind],
    user_message_ix: usize,
    body: Range<usize>,
) -> Option<CollapsibleTurn> {
    let body_kinds = &kinds[body.clone()];
    if body_kinds.contains(&EntryKind::NeedsAction) {
        return None;
    }
    let final_answer_offset = body_kinds
        .iter()
        .rposition(|kind| matches!(kind, EntryKind::Answer { .. }))?;
    let ends_on_answer = body_kinds[final_answer_offset + 1..]
        .iter()
        .all(|kind| matches!(kind, EntryKind::Aside | EntryKind::Invisible));
    if !ends_on_answer {
        return None;
    }
    let final_answer_ix = body.start + final_answer_offset;
    let EntryKind::Answer { thinking_blocks } = kinds[final_answer_ix] else {
        return None;
    };
    let work = body.start..final_answer_ix;
    let has_visible_work = kinds[work.clone()].iter().any(|kind| {
        !matches!(
            kind,
            EntryKind::Invisible | EntryKind::ToolCall { renders: false, .. }
        )
    });
    let final_answer_has_thoughts = thinking_blocks > 0;
    (has_visible_work || final_answer_has_thoughts).then(|| CollapsibleTurn {
        user_message_ix,
        steps: steps(&kinds[work.clone()]) + usize::from(final_answer_has_thoughts),
        work,
        final_answer_ix,
    })
}

fn steps(kinds: &[EntryKind]) -> usize {
    kinds
        .iter()
        .map(|kind| match kind {
            EntryKind::ToolCall { renders: true, .. } => 1,
            EntryKind::Thinking { blocks } => *blocks,
            EntryKind::Answer { thinking_blocks } => *thinking_blocks,
            EntryKind::ToolCall { renders: false, .. }
            | EntryKind::UserMessage
            | EntryKind::NeedsAction
            | EntryKind::Aside
            | EntryKind::Invisible => 0,
        })
        .sum()
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

pub(crate) fn has_pending_request_elicitation(thread: &AcpThread, cx: &App) -> bool {
    thread
        .connection()
        .request_elicitations()
        .is_some_and(|store| {
            store
                .read(cx)
                .elicitations()
                .iter()
                .any(|elicitation| matches!(elicitation.status, ElicitationStatus::Pending { .. }))
        })
}

pub(crate) fn entry_kind(entry: &AgentThreadEntry, thread: &AcpThread, cx: &App) -> EntryKind {
    match entry {
        AgentThreadEntry::UserMessage(_) => EntryKind::UserMessage,
        AgentThreadEntry::AssistantMessage(message) => {
            let mut has_text = false;
            let mut thinking_blocks = 0;
            for chunk in &message.chunks {
                match chunk {
                    AssistantMessageChunk::Message { block, .. } => {
                        has_text |= block.visible_content(cx);
                    }
                    AssistantMessageChunk::Thought { block, .. } => {
                        thinking_blocks += usize::from(block.visible_content(cx));
                    }
                }
            }
            if has_text {
                EntryKind::Answer { thinking_blocks }
            } else if thinking_blocks > 0 {
                EntryKind::Thinking {
                    blocks: thinking_blocks,
                }
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
    use super::EntryKind::{Aside, Invisible, NeedsAction, UserMessage};
    use super::*;

    const TEXT: EntryKind = EntryKind::Answer { thinking_blocks: 0 };
    const TEXT_WITH_THOUGHTS: EntryKind = EntryKind::Answer { thinking_blocks: 1 };
    const THOUGHT: EntryKind = EntryKind::Thinking { blocks: 1 };
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

    struct Scenario<'a> {
        kinds: &'a [EntryKind],
        last_turn_is_live: bool,
        turn_records: HashMap<usize, TurnRecord>,
        expanded: HashSet<CollapseKey>,
    }

    impl<'a> Scenario<'a> {
        fn new(kinds: &'a [EntryKind]) -> Self {
            Self {
                kinds,
                last_turn_is_live: false,
                turn_records: HashMap::default(),
                expanded: HashSet::default(),
            }
        }

        fn live(mut self) -> Self {
            self.last_turn_is_live = true;
            self
        }

        fn record(mut self, user_message_ix: usize, outcome: TurnOutcome) -> Self {
            self.turn_records.insert(
                user_message_ix,
                TurnRecord {
                    duration: Some(Duration::from_secs(54)),
                    outcome,
                },
            );
            self
        }

        fn expand(mut self, key: CollapseKey) -> Self {
            self.expanded.insert(key);
            self
        }

        fn layout(&self) -> Vec<EntryPresentation> {
            let tool_call_ids = self
                .kinds
                .iter()
                .enumerate()
                .map(|(entry_ix, kind)| {
                    matches!(kind, EntryKind::ToolCall { .. }).then(|| tool_call_id(entry_ix))
                })
                .collect::<Vec<_>>();
            layout(LayoutInput {
                kinds: self.kinds,
                tool_call_ids: &tool_call_ids,
                tail_is_growing: self.last_turn_is_live,
                last_turn_is_live: self.last_turn_is_live,
                turn_records: &self.turn_records,
                expanded: &self.expanded,
            })
        }
    }

    fn layout_of(
        kinds: &[EntryKind],
        tail_is_growing: bool,
        expanded: &HashSet<CollapseKey>,
    ) -> Vec<EntryPresentation> {
        let mut scenario = Scenario::new(kinds);
        scenario.last_turn_is_live = tail_is_growing;
        scenario.expanded = expanded.clone();
        scenario.layout()
    }

    fn collapsed(kinds: &[EntryKind], tail_is_growing: bool) -> Vec<EntryPresentation> {
        layout_of(kinds, tail_is_growing, &HashSet::default())
    }

    fn full() -> EntryPresentation {
        EntryPresentation::default()
    }

    fn hidden() -> EntryPresentation {
        EntryPresentation {
            headers: Vec::new(),
            content: Content::Hidden,
        }
    }

    fn group_header(first_entry_ix: usize, count: usize, is_expanded: bool) -> Header {
        Header::ToolCallGroup {
            key: CollapseKey::ToolCalls(tool_call_id(first_entry_ix)),
            count,
            is_expanded,
        }
    }

    fn group(first_entry_ix: usize, count: usize, is_expanded: bool) -> EntryPresentation {
        EntryPresentation {
            headers: vec![group_header(first_entry_ix, count, is_expanded)],
            content: if is_expanded {
                Content::Full
            } else {
                Content::Hidden
            },
        }
    }

    fn turn_header(user_message_ix: usize, steps: usize, is_expanded: bool) -> Header {
        Header::TurnSummary {
            key: CollapseKey::Turn(user_message_ix),
            steps,
            duration: None,
            is_expanded,
        }
    }

    fn with_headers(content: Content, headers: Vec<Header>) -> EntryPresentation {
        EntryPresentation { headers, content }
    }

    fn answer_without_thoughts() -> EntryPresentation {
        with_headers(Content::AnswerWithoutThoughts, Vec::new())
    }

    fn turn_summaries(presentation: &[EntryPresentation]) -> Vec<(usize, usize)> {
        presentation
            .iter()
            .enumerate()
            .flat_map(|(entry_ix, entry)| {
                entry.headers.iter().filter_map(move |header| match header {
                    Header::TurnSummary { steps, .. } => Some((entry_ix, *steps)),
                    Header::ToolCallGroup { .. } => None,
                })
            })
            .collect()
    }

    #[test]
    fn groups_consecutive_finished_tool_calls_between_barriers() {
        let kinds = [TEXT, TOOL, TOOL, TOOL, TEXT, TOOL, TEXT];
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
        let kinds = [TEXT, TOOL, TEXT, TOOL];
        assert_eq!(collapsed(&kinds, false), vec![full(); 4]);
    }

    #[test]
    fn invisible_entries_do_not_split_runs() {
        let kinds = [TEXT, TOOL, Invisible, TOOL, Invisible, TEXT];
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
        let kinds = [TEXT, TOOL, EMPTY_TOOL, TOOL, TEXT, EMPTY_TOOL, TOOL];
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
        for barrier in [UserMessage, TEXT, THOUGHT, DIFF, NeedsAction, Aside] {
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
        let kinds = [UserMessage, TEXT, TOOL, TOOL, TOOL, Invisible];
        assert_eq!(
            collapsed(&kinds, true),
            vec![full(), full(), group(2, 2, false), hidden(), full(), full()]
        );

        let kinds = [UserMessage, TOOL, TOOL];
        assert_eq!(collapsed(&kinds, true), vec![full(); 3]);
    }

    #[test]
    fn earlier_runs_are_grouped_while_generating() {
        let kinds = [TOOL, TOOL, TEXT, TOOL];
        assert_eq!(
            collapsed(&kinds, true),
            vec![group(0, 2, false), hidden(), full(), full()]
        );
    }

    #[test]
    fn expanded_groups_keep_their_header_and_show_every_entry() {
        let kinds = [TEXT, TOOL, TOOL, TEXT, TOOL, TOOL];
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

    #[test]
    fn collapses_finished_turns_up_to_the_final_answer() {
        let kinds = [UserMessage, TEXT, TOOL, THOUGHT, TOOL, TOOL, TEXT, Aside];
        assert_eq!(
            Scenario::new(&kinds).layout(),
            vec![
                full(),
                with_headers(Content::Hidden, vec![turn_header(0, 4, false)]),
                hidden(),
                hidden(),
                hidden(),
                hidden(),
                answer_without_thoughts(),
                full(),
            ]
        );
    }

    #[test]
    fn expanded_turns_keep_their_tool_call_groups() {
        let kinds = [UserMessage, TOOL, TOOL, TOOL, TEXT];
        let scenario = Scenario::new(&kinds).expand(CollapseKey::Turn(0));
        assert_eq!(
            scenario.layout(),
            vec![
                full(),
                with_headers(
                    Content::Hidden,
                    vec![turn_header(0, 3, true), group_header(1, 3, false)]
                ),
                hidden(),
                hidden(),
                full(),
            ]
        );

        let scenario = scenario.expand(CollapseKey::ToolCalls(tool_call_id(1)));
        assert_eq!(
            scenario.layout(),
            vec![
                full(),
                with_headers(
                    Content::Full,
                    vec![turn_header(0, 3, true), group_header(1, 3, true)]
                ),
                full(),
                full(),
                full(),
            ]
        );
    }

    #[test]
    fn runs_next_to_a_collapsed_turn_keep_their_groups() {
        let kinds = [
            UserMessage,
            TOOL,
            TOOL,
            UserMessage,
            TOOL,
            TOOL,
            TEXT,
            UserMessage,
            TOOL,
            TOOL,
        ];
        assert_eq!(
            Scenario::new(&kinds).layout(),
            vec![
                full(),
                group(1, 2, false),
                hidden(),
                full(),
                with_headers(Content::Hidden, vec![turn_header(3, 2, false)]),
                hidden(),
                answer_without_thoughts(),
                full(),
                group(8, 2, false),
                hidden(),
            ]
        );
    }

    #[test]
    fn interrupted_turns_never_collapse() {
        let kinds = [UserMessage, THOUGHT, TOOL, TEXT];
        let scenario = Scenario::new(&kinds).record(0, TurnOutcome::Interrupted);
        assert_eq!(scenario.layout(), vec![full(); 4]);
    }

    #[test]
    fn finished_and_unknown_turns_follow_the_shape_rule() {
        let kinds = [
            UserMessage,
            TOOL,
            TEXT,
            UserMessage,
            TOOL,
            TEXT,
            UserMessage,
            TOOL,
        ];
        for outcome in [TurnOutcome::Finished, TurnOutcome::Unknown] {
            let presentation = Scenario::new(&kinds)
                .record(0, outcome)
                .record(3, outcome)
                .record(6, outcome)
                .layout();
            assert_eq!(turn_summaries(&presentation), vec![(1, 1), (4, 1)]);
            let Some(Header::TurnSummary { duration, .. }) = presentation[1].headers.first() else {
                panic!("expected a turn summary");
            };
            assert_eq!(*duration, Some(Duration::from_secs(54)));
        }
        let restored = Scenario::new(&kinds).layout();
        assert_eq!(turn_summaries(&restored), vec![(1, 1), (4, 1)]);
    }

    #[test]
    fn turns_that_end_on_work_stay_open() {
        for kinds in [
            [UserMessage, TEXT, TOOL],
            [UserMessage, TEXT, THOUGHT],
            [UserMessage, TOOL, TOOL],
            [UserMessage, TEXT, NeedsAction],
        ] {
            assert!(
                turn_summaries(&Scenario::new(&kinds).layout()).is_empty(),
                "{kinds:?}"
            );
        }
        let kinds = [UserMessage, TOOL, NeedsAction, TEXT];
        assert!(turn_summaries(&Scenario::new(&kinds).layout()).is_empty());
    }

    #[test]
    fn only_finished_turns_collapse_while_the_last_turn_is_live() {
        let kinds = [UserMessage, TOOL, TEXT, UserMessage, TOOL, TEXT];
        assert_eq!(
            turn_summaries(&Scenario::new(&kinds).live().layout()),
            vec![(1, 1)]
        );
        assert_eq!(
            turn_summaries(&Scenario::new(&kinds).layout()),
            vec![(1, 1), (4, 1)]
        );
    }

    #[test]
    fn leaves_turns_without_work_alone() {
        let kinds = [
            TOOL,
            TEXT,
            UserMessage,
            TEXT,
            UserMessage,
            Invisible,
            EMPTY_TOOL,
            TEXT,
        ];
        assert_eq!(Scenario::new(&kinds).layout(), vec![full(); 8]);
    }

    #[test]
    fn steps_count_tool_calls_and_thinking_blocks() {
        let steps = |kinds: &[EntryKind]| turn_summaries(&Scenario::new(kinds).layout());

        assert_eq!(steps(&[UserMessage, Aside, TEXT]), vec![(1, 0)]);
        assert_eq!(steps(&[UserMessage, TEXT_WITH_THOUGHTS]), vec![(1, 1)]);
        assert_eq!(
            steps(&[
                UserMessage,
                EntryKind::Thinking { blocks: 2 },
                TEXT_WITH_THOUGHTS,
                TOOL,
                EMPTY_TOOL,
                DIFF,
                Invisible,
                TEXT,
                EntryKind::Answer { thinking_blocks: 3 },
            ]),
            vec![(1, 6)]
        );
        assert_eq!(
            Scenario::new(&[UserMessage, TEXT_WITH_THOUGHTS]).layout()[1],
            with_headers(
                Content::AnswerWithoutThoughts,
                vec![turn_header(0, 1, false)]
            )
        );
    }

    #[test]
    fn turn_summary_label_prefers_long_durations_over_step_counts() {
        assert_eq!(
            turn_summary_label(3, Some(Duration::from_secs(54))),
            "Worked for 54s"
        );
        assert_eq!(
            turn_summary_label(3, Some(STOPWATCH_THRESHOLD)),
            "Worked · 3 steps"
        );
        let just_over_threshold = STOPWATCH_THRESHOLD + Duration::from_secs(1);
        assert_eq!(
            turn_summary_label(3, Some(just_over_threshold)),
            format!("Worked for {}", duration_alt_display(just_over_threshold))
        );
        assert_eq!(
            turn_summary_label(3, Some(Duration::from_secs(4))),
            "Worked · 3 steps"
        );
        assert_eq!(turn_summary_label(1, None), "Worked · 1 step");
        assert_eq!(turn_summary_label(0, None), "Worked");
    }
}
