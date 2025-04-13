use futures_channel::mpsc::UnboundedReceiver;
use futures_util::{Stream, StreamExt};

/// A stream of text from a tokio channel.
pub struct ChannelTextStream<S: AsRef<str> = String> {
    receiver: UnboundedReceiver<S>,
}

impl<S: AsRef<str>> std::fmt::Debug for ChannelTextStream<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelTextStream").finish()
    }
}

impl<S: AsRef<str>> From<UnboundedReceiver<S>> for ChannelTextStream<S> {
    fn from(receiver: UnboundedReceiver<S>) -> Self {
        Self { receiver }
    }
}

impl<S: AsRef<str>> Stream for ChannelTextStream<S> {
    type Item = S;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> core::task::Poll<Option<Self::Item>> {
        self.receiver.poll_next_unpin(cx)
    }
}
