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
pub(crate) struct TurnOwnership {
    owner: TurnOwner,
    newest_owned_user_message_ix: Option<usize>,
}

impl TurnOwnership {
    pub(crate) fn new(last_user_message_ix: Option<usize>) -> Self {
        Self {
            owner: TurnOwner::None,
            newest_owned_user_message_ix: last_user_message_ix,
        }
    }

    pub(crate) fn start(&mut self, owner: TurnOwner) {
        self.owner = owner;
        if let TurnOwner::UserMessage(user_message_ix) = owner {
            self.newest_owned_user_message_ix = Some(user_message_ix);
        }
    }

    pub(crate) fn start_reported(&mut self, last_user_message_ix: Option<usize>) {
        let unowned_user_message_ix = last_user_message_ix.filter(|&user_message_ix| {
            self.newest_owned_user_message_ix
                .is_none_or(|newest| user_message_ix > newest)
        });
        self.start(match unowned_user_message_ix {
            Some(user_message_ix) => TurnOwner::UserMessage(user_message_ix),
            None => TurnOwner::NextUserMessage,
        });
    }

    pub(crate) fn user_message_pushed(&mut self, entry_ix: usize) {
        if self.owner == TurnOwner::NextUserMessage {
            self.start(TurnOwner::UserMessage(entry_ix));
        }
    }

    pub(crate) fn has_owner(&self) -> bool {
        matches!(self.owner, TurnOwner::UserMessage(_))
    }

    pub(crate) fn take(&mut self) -> Option<usize> {
        match std::mem::take(&mut self.owner) {
            TurnOwner::UserMessage(user_message_ix) => Some(user_message_ix),
            TurnOwner::None | TurnOwner::NextUserMessage => None,
        }
    }

    pub(crate) fn remove(&mut self, removed: &Range<usize>) {
        if let TurnOwner::UserMessage(user_message_ix) = self.owner {
            self.owner = reindex_after_removal(user_message_ix, removed)
                .map_or(TurnOwner::None, TurnOwner::UserMessage);
        }
        self.newest_owned_user_message_ix = self.newest_owned_user_message_ix.and_then(|newest| {
            reindex_after_removal(newest, removed).or_else(|| removed.start.checked_sub(1))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_stop_reasons_to_turn_outcomes() {
        use acp_v2::StopReason;
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
        let mut ownership = TurnOwnership::new(Some(0));
        ownership.start_reported(Some(0));
        assert_eq!(ownership.take(), None);

        ownership.start(TurnOwner::NextUserMessage);
        ownership.user_message_pushed(3);
        assert_eq!(ownership.take(), Some(3));

        ownership.start(TurnOwner::None);
        ownership.user_message_pushed(6);
        assert_eq!(ownership.take(), None);

        ownership.start_reported(Some(3));
        assert!(!ownership.has_owner());
        ownership.user_message_pushed(8);
        assert_eq!(ownership.take(), Some(8));

        ownership.start_reported(Some(10));
        assert_eq!(ownership.take(), Some(10));
    }

    #[test]
    fn removed_owners_are_dropped_and_later_ones_shift() {
        let mut ownership = TurnOwnership::default();
        ownership.start(TurnOwner::UserMessage(4));
        ownership.remove(&(4..6));
        assert_eq!(ownership.take(), None);
        ownership.start_reported(Some(3));
        assert_eq!(ownership.take(), None);

        ownership.start(TurnOwner::UserMessage(6));
        ownership.remove(&(1..3));
        assert_eq!(ownership.take(), Some(4));
    }
}
