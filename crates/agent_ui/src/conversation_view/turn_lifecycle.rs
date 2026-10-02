use std::{ops::Range, time::Duration};

use agent_client_protocol::schema::v2 as acp_v2;

use crate::entry_view_state::reindex_after_removal;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TurnOutcome {
    Finished,
    Interrupted,
    Unknown,
}

impl TurnOutcome {
    pub(crate) fn from_stop_reason(stop_reason: Option<&acp_v2::StopReason>) -> Self {
        match stop_reason {
            Some(acp_v2::StopReason::EndTurn) => Self::Finished,
            Some(
                acp_v2::StopReason::Cancelled
                | acp_v2::StopReason::MaxTokens
                | acp_v2::StopReason::MaxTurnRequests
                | acp_v2::StopReason::Refusal,
            ) => Self::Interrupted,
            _ => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TurnRecord {
    pub duration: Option<Duration>,
    pub outcome: TurnOutcome,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum TurnOwner {
    #[default]
    None,
    NextUserMessage,
    UserMessage(usize),
}

// Prompts that already belong to a turn, including ones restored with the thread, are never
// claimed again, so activity without a prompt of its own (such as `/compact` on an agent
// that reports its activity) cannot overwrite the record of an earlier prompt.
#[derive(Debug, Default)]
pub(crate) struct TurnLifecycle {
    owner: TurnOwner,
    newest_owned_user_message_ix: Option<usize>,
    in_flight: bool,
    outcome: Option<TurnOutcome>,
    reported_duration: Option<Duration>,
    last_recorded_owner: TurnOwner,
}

impl TurnLifecycle {
    pub(crate) fn new(last_user_message_ix: Option<usize>) -> Self {
        Self {
            newest_owned_user_message_ix: last_user_message_ix,
            ..Self::default()
        }
    }

    pub(crate) fn start(
        &mut self,
        owner: TurnOwner,
        previous_elapsed: Option<Duration>,
    ) -> Option<(usize, TurnRecord)> {
        let previous_turn = if self.is_live() {
            self.outcome.get_or_insert(TurnOutcome::Interrupted);
            self.record(previous_elapsed)
        } else {
            None
        };
        self.in_flight = true;
        self.outcome = None;
        self.reported_duration = None;
        self.claim(owner);
        previous_turn
    }

    pub(crate) fn start_reported(
        &mut self,
        last_user_message_ix: Option<usize>,
        previous_elapsed: Option<Duration>,
    ) -> Option<(usize, TurnRecord)> {
        let unowned_user_message_ix = last_user_message_ix.filter(|&user_message_ix| {
            self.newest_owned_user_message_ix
                .is_none_or(|newest| user_message_ix > newest)
        });
        let owner = match unowned_user_message_ix {
            Some(user_message_ix) => TurnOwner::UserMessage(user_message_ix),
            None => TurnOwner::NextUserMessage,
        };
        self.start(owner, previous_elapsed)
    }

    pub(crate) fn user_message_pushed(&mut self, entry_ix: usize) {
        if self.owner == TurnOwner::NextUserMessage {
            self.claim(TurnOwner::UserMessage(entry_ix));
        }
    }

    pub(crate) fn finish(
        &mut self,
        stop_reason: Option<&acp_v2::StopReason>,
        reported_duration: Option<Duration>,
    ) {
        if !self.in_flight {
            return;
        }
        self.set_outcome(TurnOutcome::from_stop_reason(stop_reason));
        self.reported_duration = reported_duration;
    }

    pub(crate) fn interrupt(&mut self) {
        self.set_outcome(TurnOutcome::Interrupted);
    }

    pub(crate) fn record(&mut self, elapsed: Option<Duration>) -> Option<(usize, TurnRecord)> {
        self.in_flight = false;
        let outcome = self.outcome.take().unwrap_or(TurnOutcome::Unknown);
        let reported_duration = self.reported_duration.take();
        let user_message_ix = match std::mem::take(&mut self.owner) {
            TurnOwner::UserMessage(user_message_ix) => Some(user_message_ix),
            TurnOwner::None | TurnOwner::NextUserMessage => None,
        };
        self.last_recorded_owner = user_message_ix.map_or(TurnOwner::None, TurnOwner::UserMessage);
        user_message_ix.map(|user_message_ix| {
            (
                user_message_ix,
                TurnRecord {
                    duration: reported_duration.or(elapsed),
                    outcome,
                },
            )
        })
    }

    pub(crate) fn retry_owner(&self) -> TurnOwner {
        self.last_recorded_owner
    }

    pub(crate) fn is_live(&self) -> bool {
        matches!(self.owner, TurnOwner::UserMessage(_))
    }

    pub(crate) fn remove(&mut self, removed: &Range<usize>) {
        for owner in [&mut self.owner, &mut self.last_recorded_owner] {
            if let TurnOwner::UserMessage(user_message_ix) = *owner {
                *owner = reindex_after_removal(user_message_ix, removed)
                    .map_or(TurnOwner::None, TurnOwner::UserMessage);
            }
        }
        self.newest_owned_user_message_ix = self.newest_owned_user_message_ix.and_then(|newest| {
            reindex_after_removal(newest, removed).or_else(|| removed.start.checked_sub(1))
        });
    }

    #[cfg(test)]
    pub(crate) fn outcome(&self) -> Option<TurnOutcome> {
        self.outcome
    }

    fn claim(&mut self, owner: TurnOwner) {
        self.owner = owner;
        if let TurnOwner::UserMessage(user_message_ix) = owner {
            self.newest_owned_user_message_ix = Some(user_message_ix);
        }
    }

    fn set_outcome(&mut self, outcome: TurnOutcome) {
        if self.in_flight && self.outcome != Some(TurnOutcome::Interrupted) {
            self.outcome = Some(outcome);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acp_v2::StopReason;

    fn owner_ix(record: Option<(usize, TurnRecord)>) -> Option<usize> {
        record.map(|(user_message_ix, _)| user_message_ix)
    }

    fn outcome(record: Option<(usize, TurnRecord)>) -> Option<TurnOutcome> {
        record.map(|(_, record)| record.outcome)
    }

    #[test]
    fn maps_stop_reasons_to_turn_outcomes() {
        assert_eq!(
            TurnOutcome::from_stop_reason(Some(&StopReason::EndTurn)),
            TurnOutcome::Finished
        );
        for stop_reason in [
            StopReason::Cancelled,
            StopReason::MaxTokens,
            StopReason::MaxTurnRequests,
            StopReason::Refusal,
        ] {
            assert_eq!(
                TurnOutcome::from_stop_reason(Some(&stop_reason)),
                TurnOutcome::Interrupted
            );
        }
        assert_eq!(TurnOutcome::from_stop_reason(None), TurnOutcome::Unknown);
    }

    #[test]
    fn native_commands_and_restored_prompts_are_never_claimed() {
        let mut lifecycle = TurnLifecycle::new(Some(0));
        lifecycle.start_reported(Some(0), None);
        assert_eq!(owner_ix(lifecycle.record(None)), None);

        lifecycle.start(TurnOwner::NextUserMessage, None);
        lifecycle.user_message_pushed(3);
        assert_eq!(owner_ix(lifecycle.record(None)), Some(3));

        lifecycle.start(TurnOwner::None, None);
        lifecycle.user_message_pushed(6);
        assert_eq!(owner_ix(lifecycle.record(None)), None);

        lifecycle.start_reported(Some(3), None);
        assert!(!lifecycle.is_live());
        lifecycle.user_message_pushed(8);
        assert_eq!(owner_ix(lifecycle.record(None)), Some(8));

        lifecycle.start_reported(Some(10), None);
        assert_eq!(owner_ix(lifecycle.record(None)), Some(10));
    }

    #[test]
    fn removed_owners_are_dropped_and_later_ones_shift() {
        let mut lifecycle = TurnLifecycle::default();
        lifecycle.start(TurnOwner::UserMessage(4), None);
        lifecycle.remove(&(4..6));
        assert_eq!(owner_ix(lifecycle.record(None)), None);
        lifecycle.start_reported(Some(3), None);
        assert_eq!(owner_ix(lifecycle.record(None)), None);

        lifecycle.start(TurnOwner::UserMessage(6), None);
        lifecycle.remove(&(1..3));
        assert_eq!(owner_ix(lifecycle.record(None)), Some(4));
    }

    #[test]
    fn starting_a_turn_interrupts_the_live_one() {
        let mut lifecycle = TurnLifecycle::default();
        let elapsed = Duration::from_secs(3);
        assert_eq!(lifecycle.start(TurnOwner::UserMessage(0), None), None);
        assert_eq!(
            lifecycle.start(TurnOwner::NextUserMessage, Some(elapsed)),
            Some((
                0,
                TurnRecord {
                    duration: Some(elapsed),
                    outcome: TurnOutcome::Interrupted,
                }
            ))
        );

        lifecycle.user_message_pushed(2);
        lifecycle.finish(Some(&StopReason::EndTurn), None);
        assert_eq!(
            outcome(lifecycle.start(TurnOwner::None, None)),
            Some(TurnOutcome::Finished)
        );
    }

    #[test]
    fn stops_outside_a_turn_are_ignored() {
        let mut lifecycle = TurnLifecycle::default();
        lifecycle.interrupt();
        lifecycle.finish(Some(&StopReason::Cancelled), Some(Duration::from_secs(9)));
        assert_eq!(lifecycle.outcome(), None);

        lifecycle.start(TurnOwner::UserMessage(0), None);
        assert_eq!(
            lifecycle.record(None),
            Some((
                0,
                TurnRecord {
                    duration: None,
                    outcome: TurnOutcome::Unknown,
                }
            ))
        );

        lifecycle.finish(Some(&StopReason::Cancelled), None);
        lifecycle.interrupt();
        assert_eq!(lifecycle.outcome(), None);
    }

    #[test]
    fn interrupted_outcomes_are_sticky() {
        let mut lifecycle = TurnLifecycle::default();
        lifecycle.start(TurnOwner::UserMessage(0), None);
        lifecycle.interrupt();
        lifecycle.finish(Some(&StopReason::EndTurn), None);
        assert_eq!(
            outcome(lifecycle.record(None)),
            Some(TurnOutcome::Interrupted)
        );

        lifecycle.start(TurnOwner::UserMessage(2), None);
        lifecycle.finish(Some(&StopReason::EndTurn), None);
        lifecycle.finish(Some(&StopReason::Cancelled), None);
        lifecycle.finish(Some(&StopReason::EndTurn), None);
        assert_eq!(
            outcome(lifecycle.record(None)),
            Some(TurnOutcome::Interrupted)
        );
    }

    #[test]
    fn retries_resume_the_owner_of_the_last_recorded_turn() {
        let mut lifecycle = TurnLifecycle::default();
        lifecycle.start(TurnOwner::UserMessage(3), None);
        lifecycle.record(None);
        lifecycle.start(TurnOwner::None, None);
        lifecycle.record(None);
        assert_eq!(lifecycle.retry_owner(), TurnOwner::None);

        lifecycle.start(TurnOwner::NextUserMessage, None);
        lifecycle.record(None);
        assert_eq!(lifecycle.retry_owner(), TurnOwner::None);

        lifecycle.start(TurnOwner::UserMessage(5), None);
        lifecycle.record(None);
        assert_eq!(lifecycle.retry_owner(), TurnOwner::UserMessage(5));

        lifecycle.remove(&(1..3));
        assert_eq!(lifecycle.retry_owner(), TurnOwner::UserMessage(3));
        lifecycle.remove(&(3..4));
        assert_eq!(lifecycle.retry_owner(), TurnOwner::None);
    }

    #[test]
    fn reported_durations_win_over_elapsed_time() {
        let mut lifecycle = TurnLifecycle::default();
        let reported = Duration::from_secs(5);
        let elapsed = Duration::from_secs(7);
        lifecycle.start(TurnOwner::UserMessage(0), None);
        lifecycle.finish(Some(&StopReason::EndTurn), Some(reported));
        assert_eq!(
            lifecycle
                .record(Some(elapsed))
                .map(|(_, record)| record.duration),
            Some(Some(reported))
        );

        lifecycle.start(TurnOwner::UserMessage(2), None);
        lifecycle.finish(Some(&StopReason::EndTurn), None);
        assert_eq!(
            lifecycle
                .record(Some(elapsed))
                .map(|(_, record)| record.duration),
            Some(Some(elapsed))
        );
    }
}
