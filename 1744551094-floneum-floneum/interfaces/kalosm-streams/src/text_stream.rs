//! Streams for text data.

use pin_project_lite::pin_project;
use std::{
    collections::VecDeque,
    pin::Pin,
    task::{Context, Poll},
};

pub use crate::sender::*;
use futures_util::{Stream, StreamExt};

/// A stream of text. This is automatically implemented for all streams of something that acts like a string (String, &str).
pub trait TextStream<I: AsRef<str> = String>: Stream<Item = I> {
    /// Split the stream into words.
    fn words(self) -> WordStream<Self, I>
    where
        Self: Sized,
    {
        WordStream::new(self)
    }

    /// Split the stream into sentences.
    fn sentences(self) -> SentenceStream<Self, I>
    where
        Self: Sized,
    {
        SentenceStream::new(self)
    }

    /// Split the stream into paragraphs.
    fn paragraphs(self) -> ParagraphStream<Self, I>
    where
        Self: Sized,
    {
        ParagraphStream::new(self)
    }

    /// Write the stream to a writer.
    fn write_to<'a, W: std::io::Write + Send + 'a>(
        &'a mut self,
        mut writer: W,
    ) -> impl std::future::Future<Output = std::io::Result<()>> + Send + 'a
    where
        Self: Sized + Unpin + Send,
    {
        async move {
            while let Some(text) = self.next().await {
                writer.write_all(text.as_ref().as_bytes())?;
                writer.flush()?;
            }
            Ok(())
        }
    }

    /// Get all the text from the stream.
    fn all_text(&mut self) -> impl std::future::Future<Output = String> + Send + '_
    where
        Self: Sized + Unpin + Send,
    {
        async move {
            let mut all_text = String::new();
            while let Some(text) = self.next().await {
                all_text.push_str(text.as_ref());
            }
            all_text
        }
    }

    /// Write the stream to standard output.
    fn to_std_out(&mut self) -> impl std::future::Future<Output = std::io::Result<()>> + Send + '_
    where
        Self: Sized + Unpin + Send,
    {
        self.write_to(std::io::stdout())
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>> TextStream<I> for S {}

/// A pattern that matches a character.
pub trait Pattern {
    /// Check if a character matches the pattern.
    fn matches(&self, char: char) -> bool;
}

pin_project! {
    /// A stream that output segments of text at a time.
    pub struct SegmentedStream<S: Stream<Item = I>, I: AsRef<str>, P: Pattern> {
        #[pin]
        backing: S,
        queue: VecDeque<String>,
        incomplete: String,
        pattern: P,
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>, P: Pattern> SegmentedStream<S, I, P> {
    /// Create a new segmented stream from a stream of text and a pattern that separates segments
    fn new(backing: S, pattern: P) -> Self {
        Self {
            backing,
            queue: Default::default(),
            incomplete: Default::default(),
            pattern,
        }
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>, P: Pattern> Stream for SegmentedStream<S, I, P> {
    type Item = String;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let projected = self.project();
        let mut backing = projected.backing;
        let incomplete = projected.incomplete;
        let queue = projected.queue;
        if let Some(next) = queue.pop_front() {
            return Poll::Ready(Some(next));
        }
        loop {
            let poll = backing.as_mut().poll_next(cx);
            match poll {
                Poll::Ready(Some(item)) => {
                    let item = item.as_ref();
                    let mut completed = None;
                    for char in item.chars() {
                        if projected.pattern.matches(char) {
                            incomplete.push(char);
                            let full_sentence = std::mem::take(incomplete);
                            if completed.is_some() {
                                queue.push_back(full_sentence);
                            } else {
                                completed = Some(full_sentence);
                            }
                        } else {
                            incomplete.push(char);
                        }
                    }
                    if let Some(completed) = completed {
                        return Poll::Ready(Some(completed));
                    }
                }
                Poll::Ready(None) => {
                    if !incomplete.is_empty() {
                        return Poll::Ready(Some(std::mem::take(incomplete)));
                    } else {
                        return Poll::Ready(None);
                    }
                }
                _ => {
                    return Poll::Pending;
                }
            }
        }
    }
}

struct SentencePattern;

impl Pattern for SentencePattern {
    fn matches(&self, char: char) -> bool {
        char == '.' || char == '?' || char == '!'
    }
}

pin_project! {
    /// A stream that output sentences of text at a time.
    pub struct SentenceStream<S: Stream<Item = I>, I: AsRef<str>> {
        #[pin]
        segmented: SegmentedStream<S, I, SentencePattern>,
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>> SentenceStream<S, I> {
    /// Create a new sentence stream from a stream of text
    fn new(backing: S) -> Self {
        Self {
            segmented: SegmentedStream::new(backing, SentencePattern),
        }
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>> Stream for SentenceStream<S, I> {
    type Item = String;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().segmented.poll_next(cx)
    }
}

pin_project! {
    /// A stream that output words of text at a time.
    pub struct WordStream<S: Stream<Item = I>, I: AsRef<str>> {
        #[pin]
        segmented: SegmentedStream<S, I, WordPattern>,
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>> WordStream<S, I> {
    /// Create a new word stream from a stream of text
    fn new(backing: S) -> Self {
        Self {
            segmented: SegmentedStream::new(backing, WordPattern),
        }
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>> Stream for WordStream<S, I> {
    type Item = String;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().segmented.poll_next(cx)
    }
}

struct WordPattern;

impl Pattern for WordPattern {
    fn matches(&self, char: char) -> bool {
        char.is_whitespace()
    }
}

pin_project! {
    /// A stream that output paragraphs of text at a time.
    pub struct ParagraphStream<S: Stream<Item = I>, I: AsRef<str>> {
        #[pin]
        segmented: SegmentedStream<S, I, ParagraphPattern>,
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>> ParagraphStream<S, I> {
    /// Create a new paragraph stream from a stream of text
    pub fn new(backing: S) -> Self {
        Self {
            segmented: SegmentedStream::new(backing, ParagraphPattern),
        }
    }
}

impl<S: Stream<Item = I>, I: AsRef<str>> Stream for ParagraphStream<S, I> {
    type Item = String;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().segmented.poll_next(cx)
    }
}

struct ParagraphPattern;

impl Pattern for ParagraphPattern {
    fn matches(&self, char: char) -> bool {
        char == '\n'
    }
}
