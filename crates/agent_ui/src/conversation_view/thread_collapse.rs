use std::ops::Range;

use acp_thread::{ToolCall, ToolCallContent, ToolCallStatus};
use agent_client_protocol::schema::v2 as acp_v2;

/// How a thread entry participates in grouping finished tool calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EntryGrouping {
    FinishedToolCall,
    /// Renders nothing, so it neither joins nor ends a run.
    Invisible,
    Barrier,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ToolCallRun {
    pub entries: Range<usize>,
    pub tool_call_count: usize,
}

const MIN_TOOL_CALLS_PER_RUN: usize = 2;

pub(super) fn tool_call_is_groupable(tool_call: &ToolCall) -> bool {
    let is_finished = matches!(
        tool_call.status(),
        ToolCallStatus::Completed
            | ToolCallStatus::Failed
            | ToolCallStatus::Rejected
            | ToolCallStatus::Canceled
    );
    is_finished
        && tool_call.authorization().is_none()
        && !tool_call.is_subagent()
        && !tool_call_shows_diff(tool_call)
}

fn tool_call_shows_diff(tool_call: &ToolCall) -> bool {
    matches!(tool_call.kind(), acp_v2::ToolKind::Edit)
        || tool_call.content().iter().any(|content| {
            matches!(
                content,
                ToolCallContent::Diff(_)
                    | ToolCallContent::LegacyDiff { .. }
                    | ToolCallContent::DiffPatch { .. }
            )
        })
}

/// Finds maximal runs of consecutive finished tool calls worth collapsing.
///
/// While the thread is still generating, the most recent tool call of a run at
/// the end of the thread stays out of the run, so the latest result remains
/// visible and the run only grows once the next entry arrives.
pub(super) fn finished_tool_call_runs(
    entries: &[EntryGrouping],
    tail_is_growing: bool,
) -> Vec<ToolCallRun> {
    let mut runs = Vec::new();
    let mut tool_call_indices = Vec::new();
    for (entry_ix, grouping) in entries.iter().enumerate() {
        match grouping {
            EntryGrouping::FinishedToolCall => tool_call_indices.push(entry_ix),
            EntryGrouping::Invisible => {}
            EntryGrouping::Barrier => {
                runs.extend(run_from_indices(&tool_call_indices));
                tool_call_indices.clear();
            }
        }
    }
    if tail_is_growing {
        tool_call_indices.pop();
    }
    runs.extend(run_from_indices(&tool_call_indices));
    runs
}

fn run_from_indices(tool_call_indices: &[usize]) -> Option<ToolCallRun> {
    if tool_call_indices.len() < MIN_TOOL_CALLS_PER_RUN {
        return None;
    }
    let first = *tool_call_indices.first()?;
    let last = *tool_call_indices.last()?;
    Some(ToolCallRun {
        entries: first..last + 1,
        tool_call_count: tool_call_indices.len(),
    })
}

/// How a thread entry participates in collapsing a finished turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TurnEntry {
    UserMessage,
    /// An assistant message with visible text. The last one in a turn is its
    /// final answer, provided no further work follows it.
    Answer { thinking_blocks: usize },
    /// Tool calls and thought-only messages, with the number of visible steps
    /// they render (zero for a tool call that renders nothing).
    Work { steps: usize },
    /// Collapsed with the work when it precedes the final answer, but doesn't
    /// stop a turn from ending on its answer (e.g. context compactions).
    Aside,
    Invisible,
    NeedsAction,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CollapsibleTurn {
    pub user_message_ix: usize,
    /// Everything between the user message and the final answer.
    pub work: Range<usize>,
    pub final_answer_ix: usize,
    pub final_answer_has_thoughts: bool,
    /// Visible entries hidden while collapsed: tool calls, thinking blocks and
    /// intermediate messages, with the final answer's thoughts counting as one.
    pub steps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TurnEntryContent {
    Full,
    Hidden,
    AnswerWithoutThoughts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TurnEntryLayout {
    pub shows_turn_summary: bool,
    pub content: TurnEntryContent,
}

impl CollapsibleTurn {
    pub fn summary_row_ix(&self) -> usize {
        self.user_message_ix + 1
    }

    /// Entries whose rendering depends on whether the turn is collapsed.
    pub fn affected_entries(&self) -> Range<usize> {
        self.summary_row_ix()..self.final_answer_ix + 1
    }

    pub fn hides_entry(&self, entry_ix: usize) -> bool {
        self.work.contains(&entry_ix)
    }

    pub fn entry_layout(&self, entry_ix: usize, is_expanded: bool) -> Option<TurnEntryLayout> {
        if !self.affected_entries().contains(&entry_ix) {
            return None;
        }
        let content = if is_expanded {
            TurnEntryContent::Full
        } else if entry_ix == self.final_answer_ix {
            TurnEntryContent::AnswerWithoutThoughts
        } else {
            TurnEntryContent::Hidden
        };
        Some(TurnEntryLayout {
            shows_turn_summary: entry_ix == self.summary_row_ix(),
            content,
        })
    }
}

/// Finds finished turns that end on an answer and have work worth hiding
/// behind a single summary row.
///
/// Turns that end on work (a tool call, a canceled or failed response) are
/// left alone, so how they ended stays visible.
pub(super) fn collapsible_turns(
    entries: &[TurnEntry],
    last_turn_is_live: bool,
) -> Vec<CollapsibleTurn> {
    let user_message_indices = entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| matches!(entry, TurnEntry::UserMessage))
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
            let body = user_message_ix + 1..next_user_message_ix.unwrap_or(entries.len());
            collapsible_turn(entries, user_message_ix, body)
        })
        .collect()
}

fn collapsible_turn(
    entries: &[TurnEntry],
    user_message_ix: usize,
    body: Range<usize>,
) -> Option<CollapsibleTurn> {
    let body_entries = &entries[body.clone()];
    if body_entries.contains(&TurnEntry::NeedsAction) {
        return None;
    }
    let final_answer_offset = body_entries
        .iter()
        .rposition(|entry| matches!(entry, TurnEntry::Answer { .. }))?;
    let ends_on_work = body_entries[final_answer_offset + 1..]
        .iter()
        .any(|entry| matches!(entry, TurnEntry::Work { .. }));
    if ends_on_work {
        return None;
    }
    let TurnEntry::Answer { thinking_blocks } = body_entries[final_answer_offset] else {
        return None;
    };
    let has_thoughts = thinking_blocks > 0;
    let final_answer_ix = body.start + final_answer_offset;
    let work = body.start..final_answer_ix;
    let has_visible_work = entries[work.clone()]
        .iter()
        .any(|entry| *entry != TurnEntry::Invisible);
    (has_visible_work || has_thoughts).then(|| CollapsibleTurn {
        user_message_ix,
        work: work.clone(),
        final_answer_ix,
        final_answer_has_thoughts: has_thoughts,
        steps: steps(&entries[work]) + usize::from(has_thoughts),
    })
}

fn steps(entries: &[TurnEntry]) -> usize {
    entries
        .iter()
        .map(|entry| match entry {
            TurnEntry::Answer { thinking_blocks } => 1 + thinking_blocks,
            TurnEntry::Work { steps } => *steps,
            TurnEntry::UserMessage
            | TurnEntry::Aside
            | TurnEntry::Invisible
            | TurnEntry::NeedsAction => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::EntryGrouping::{Barrier, FinishedToolCall as Tool, Invisible};
    use super::*;

    fn run(entries: Range<usize>, tool_call_count: usize) -> ToolCallRun {
        ToolCallRun {
            entries,
            tool_call_count,
        }
    }

    #[test]
    fn groups_consecutive_finished_tool_calls_between_barriers() {
        let entries = [Barrier, Tool, Tool, Tool, Barrier, Tool, Barrier];
        assert_eq!(finished_tool_call_runs(&entries, false), vec![run(1..4, 3)]);
    }

    #[test]
    fn ignores_runs_below_minimum_size() {
        let entries = [Barrier, Tool, Barrier, Tool];
        assert_eq!(finished_tool_call_runs(&entries, false), vec![]);
    }

    #[test]
    fn invisible_entries_do_not_split_runs() {
        let entries = [Barrier, Tool, Invisible, Tool, Invisible, Barrier];
        assert_eq!(finished_tool_call_runs(&entries, false), vec![run(1..4, 2)]);
    }

    #[test]
    fn trailing_run_is_grouped_once_generation_stops() {
        let entries = [Barrier, Tool, Tool, Tool];
        assert_eq!(finished_tool_call_runs(&entries, false), vec![run(1..4, 3)]);
    }

    #[test]
    fn trailing_run_keeps_latest_tool_call_visible_while_generating() {
        let entries = [Barrier, Tool, Tool, Tool, Invisible];
        assert_eq!(finished_tool_call_runs(&entries, true), vec![run(1..3, 2)]);

        let entries = [Barrier, Tool, Tool];
        assert_eq!(finished_tool_call_runs(&entries, true), vec![]);
    }

    #[test]
    fn earlier_runs_are_grouped_while_generating() {
        let entries = [Tool, Tool, Barrier, Tool];
        assert_eq!(finished_tool_call_runs(&entries, true), vec![run(0..2, 2)]);
    }

    mod turns {
        use super::super::TurnEntry::{Answer, Aside, Invisible, NeedsAction, UserMessage, Work};
        use super::super::*;

        const TEXT: TurnEntry = Answer { thinking_blocks: 0 };
        const TEXT_WITH_THOUGHTS: TurnEntry = Answer { thinking_blocks: 1 };
        const TOOL: TurnEntry = Work { steps: 1 };
        const EMPTY_TOOL: TurnEntry = Work { steps: 0 };

        fn turn(
            user_message_ix: usize,
            final_answer_ix: usize,
            final_answer_has_thoughts: bool,
            steps: usize,
        ) -> CollapsibleTurn {
            CollapsibleTurn {
                user_message_ix,
                work: user_message_ix + 1..final_answer_ix,
                final_answer_ix,
                final_answer_has_thoughts,
                steps,
            }
        }

        fn layouts(turn: &CollapsibleTurn, is_expanded: bool) -> Vec<Option<TurnEntryLayout>> {
            (0..=turn.final_answer_ix + 1)
                .map(|entry_ix| turn.entry_layout(entry_ix, is_expanded))
                .collect()
        }

        fn layout(shows_turn_summary: bool, content: TurnEntryContent) -> Option<TurnEntryLayout> {
            Some(TurnEntryLayout {
                shows_turn_summary,
                content,
            })
        }

        #[test]
        fn collapses_interleaved_messages_and_tool_calls_up_to_the_final_answer() {
            let entries = [UserMessage, TEXT, TOOL, TEXT, TOOL, TOOL, TEXT];
            let turns = collapsible_turns(&entries, false);
            assert_eq!(turns, vec![turn(0, 6, false, 5)]);

            use TurnEntryContent::{AnswerWithoutThoughts, Hidden};
            assert_eq!(
                layouts(&turns[0], false),
                vec![
                    None,
                    layout(true, Hidden),
                    layout(false, Hidden),
                    layout(false, Hidden),
                    layout(false, Hidden),
                    layout(false, Hidden),
                    layout(false, AnswerWithoutThoughts),
                    None,
                ]
            );
        }

        #[test]
        fn collapses_a_single_tool_call() {
            let entries = [UserMessage, TOOL, TEXT];
            assert_eq!(collapsible_turns(&entries, false), vec![turn(0, 2, false, 1)]);
        }

        #[test]
        fn leaves_turns_without_work_alone() {
            let entries = [UserMessage, TEXT, UserMessage, Invisible, TEXT];
            assert_eq!(collapsible_turns(&entries, false), vec![]);
        }

        #[test]
        fn collapses_thoughts_of_a_final_answer_without_other_work() {
            let entries = [UserMessage, TEXT_WITH_THOUGHTS];
            let turns = collapsible_turns(&entries, false);
            assert_eq!(turns, vec![turn(0, 1, true, 1)]);
            assert_eq!(
                turns[0].entry_layout(1, false),
                layout(true, TurnEntryContent::AnswerWithoutThoughts)
            );
            assert_eq!(
                turns[0].entry_layout(1, true),
                layout(true, TurnEntryContent::Full)
            );
        }

        #[test]
        fn leaves_turns_that_end_on_work_alone() {
            let entries = [UserMessage, TEXT, TOOL];
            assert_eq!(collapsible_turns(&entries, false), vec![]);

            let entries = [UserMessage, TOOL, TOOL];
            assert_eq!(collapsible_turns(&entries, false), vec![]);
        }

        #[test]
        fn asides_after_the_final_answer_stay_visible() {
            let entries = [UserMessage, Aside, TOOL, TEXT, Aside, Invisible];
            let turns = collapsible_turns(&entries, false);
            assert_eq!(turns, vec![turn(0, 3, false, 1)]);
            assert_eq!(turns[0].entry_layout(4, false), None);
            assert!(turns[0].hides_entry(1));
        }

        #[test]
        fn never_collapses_turns_that_need_action() {
            let entries = [UserMessage, TOOL, NeedsAction, TEXT];
            assert_eq!(collapsible_turns(&entries, false), vec![]);
        }

        #[test]
        fn only_finished_turns_collapse_while_the_last_turn_is_live() {
            let entries = [UserMessage, TOOL, TEXT, UserMessage, TOOL, TEXT];
            assert_eq!(collapsible_turns(&entries, true), vec![turn(0, 2, false, 1)]);
            assert_eq!(
                collapsible_turns(&entries, false),
                vec![turn(0, 2, false, 1), turn(3, 5, false, 1)]
            );
        }

        #[test]
        fn expanded_turns_show_everything_and_keep_their_tool_call_runs() {
            use super::super::EntryGrouping::{Barrier, FinishedToolCall};

            let entries = [UserMessage, TEXT, TOOL, TOOL, TOOL, TEXT];
            let groupings = [
                Barrier,
                Barrier,
                FinishedToolCall,
                FinishedToolCall,
                FinishedToolCall,
                Barrier,
            ];
            let turns = collapsible_turns(&entries, false);
            assert_eq!(turns, vec![turn(0, 5, false, 4)]);
            assert_eq!(
                layouts(&turns[0], true)[1..=5],
                [
                    layout(true, TurnEntryContent::Full),
                    layout(false, TurnEntryContent::Full),
                    layout(false, TurnEntryContent::Full),
                    layout(false, TurnEntryContent::Full),
                    layout(false, TurnEntryContent::Full),
                ]
            );
            assert_eq!(
                finished_tool_call_runs(&groupings, false),
                vec![ToolCallRun {
                    entries: 2..5,
                    tool_call_count: 3,
                }]
            );
        }

        #[test]
        fn counts_steps_hidden_by_the_collapse() {
            let steps = |entries: &[TurnEntry]| {
                collapsible_turns(entries, false)
                    .iter()
                    .map(|turn| turn.steps)
                    .collect::<Vec<_>>()
            };

            assert_eq!(steps(&[UserMessage, Aside, TEXT]), vec![0]);
            assert_eq!(steps(&[UserMessage, TOOL, TEXT]), vec![1]);

            let thinking = Work { steps: 2 };
            let message_with_thoughts = Answer { thinking_blocks: 1 };
            let entries = [
                UserMessage,
                thinking,
                message_with_thoughts,
                TOOL,
                EMPTY_TOOL,
                Invisible,
                Aside,
                TOOL,
                TEXT,
            ];
            assert_eq!(steps(&entries), vec![6]);
        }

        #[test]
        fn final_answer_thoughts_count_as_one_step() {
            let answer = Answer { thinking_blocks: 3 };
            assert_eq!(
                collapsible_turns(&[UserMessage, answer], false),
                vec![turn(0, 1, true, 1)]
            );
            assert_eq!(
                collapsible_turns(&[UserMessage, TOOL, answer], false),
                vec![turn(0, 2, true, 2)]
            );
        }

        #[test]
        fn ignores_entries_before_the_first_user_message() {
            let entries = [TOOL, TEXT, UserMessage, TOOL, TEXT];
            assert_eq!(collapsible_turns(&entries, false), vec![turn(2, 4, false, 1)]);
        }
    }
}
