use kalosm_language_model::Embedder;
use std::ops::Range;

use super::{Chunker, SentenceChunker};
use crate::{prelude::Document, search::Chunk};

/// A strategy for chunking a document into smaller pieces.
///
/// This is used to split a document into smaller pieces to generate embeddings for each piece.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkStrategy {
    /// Split the document into paragraphs.
    Paragraph {
        /// The number of paragraphs to include in each chunk.
        paragraph_count: usize,
        /// The number of paragraphs to overlap between chunks.
        overlap: usize,
    },
    /// Split the document into sentences.
    Sentence {
        /// The number of sentences to include in each chunk.
        sentence_count: usize,
        /// The number of sentences to overlap between chunks.
        overlap: usize,
    },
    /// Split the document into words.
    Words {
        /// The number of words to include in each chunk.
        word_count: usize,
        /// The number of words to overlap between chunks.
        overlap: usize,
    },
}

impl ChunkStrategy {
    /// Chunk a string into smaller ranges.
    pub fn chunk_str(&self, string: &str) -> Vec<Range<usize>> {
        match self {
            Self::Paragraph {
                paragraph_count,
                overlap,
            } => {
                let mut chunks = Vec::new();
                let mut start = 0;
                let mut newline_indexes = Vec::new();
                for (i, c) in string.char_indices() {
                    if c == '\n' {
                        newline_indexes.push(i + 1);
                        if newline_indexes.len() >= *paragraph_count {
                            if !string[start..i].trim().is_empty() {
                                chunks.push(start..i);
                            }
                            for _ in 0..(newline_indexes.len() - *overlap) {
                                start = newline_indexes.remove(0);
                            }
                        }
                    }
                }

                if !string[start..].trim().is_empty() {
                    chunks.push(start..string.len());
                }

                chunks
            }
            Self::Sentence {
                sentence_count,
                overlap,
            } => {
                let mut chunks = Vec::new();

                let splits = SentenceChunker::default().split_sentences(string);

                for window in splits
                    .windows(*sentence_count)
                    .step_by(*sentence_count - *overlap)
                {
                    if window.len() < *sentence_count {
                        break;
                    }
                    let start = window.first().unwrap().start;
                    let end = window.last().unwrap().end;
                    chunks.push(start..end);
                }

                chunks
            }
            Self::Words {
                word_count,
                overlap,
            } => {
                let mut chunks = Vec::new();
                let mut start = 0;
                let mut word_start_indexes = Vec::new();
                for (i, c) in string.char_indices() {
                    if c == ' ' {
                        word_start_indexes.push(i + 1);
                        if word_start_indexes.len() >= *word_count {
                            if !string[start..i].trim().is_empty() {
                                chunks.push(start..i);
                            }
                            for _ in 0..(word_start_indexes.len() - *overlap) {
                                start = word_start_indexes.remove(0);
                            }
                        }
                    }
                }

                if !string[start..].trim().is_empty() {
                    chunks.push(start..string.len());
                }

                chunks
            }
        }
    }
}

impl Default for ChunkStrategy {
    fn default() -> Self {
        Self::Paragraph {
            paragraph_count: 3,
            overlap: 1,
        }
    }
}

#[test]
fn test_chunking() {
    let string = "The quick brown fox jumps over the lazy dog.";
    let chunks = ChunkStrategy::Words {
        word_count: 3,
        overlap: 1,
    };
    let chunks = chunks.chunk_str(string);
    assert_eq!(chunks.len(), 4);
    assert_eq!(string[chunks[0].clone()].trim(), "The quick brown");
    assert_eq!(string[chunks[1].clone()].trim(), "brown fox jumps");
    assert_eq!(string[chunks[2].clone()].trim(), "jumps over the");
    assert_eq!(string[chunks[3].clone()].trim(), "the lazy dog.");

    let chunks = ChunkStrategy::Words {
        word_count: 3,
        overlap: 2,
    };
    let chunks = chunks.chunk_str(string);
    assert_eq!(chunks.len(), 7);
    assert_eq!(string[chunks[0].clone()].trim(), "The quick brown");
    assert_eq!(string[chunks[1].clone()].trim(), "quick brown fox");
    assert_eq!(string[chunks[2].clone()].trim(), "brown fox jumps");
    assert_eq!(string[chunks[3].clone()].trim(), "fox jumps over");
    assert_eq!(string[chunks[4].clone()].trim(), "jumps over the");
    assert_eq!(string[chunks[5].clone()].trim(), "over the lazy");
    assert_eq!(string[chunks[6].clone()].trim(), "the lazy dog.");

    let chunks = ChunkStrategy::Sentence {
        sentence_count: 2,
        overlap: 1,
    };

    let string = "first sentence. second sentence. third sentence. fourth sentence.";

    let chunks = chunks.chunk_str(string);
    assert_eq!(chunks.len(), 3);
    assert_eq!(
        string[chunks[0].clone()].trim(),
        "first sentence. second sentence."
    );
    assert_eq!(
        string[chunks[1].clone()].trim(),
        "second sentence. third sentence."
    );
    assert_eq!(
        string[chunks[2].clone()].trim(),
        "third sentence. fourth sentence."
    );

    let chunks = ChunkStrategy::Paragraph {
        paragraph_count: 3,
        overlap: 1,
    };

    let string = "first paragraph\n\nsecond paragraph\n\nthird paragraph\n\nfourth paragraph";

    let chunks = chunks.chunk_str(string);
    assert_eq!(chunks.len(), 3);
    assert_eq!(
        string[chunks[0].clone()].trim(),
        "first paragraph\n\nsecond paragraph"
    );
    assert_eq!(
        string[chunks[1].clone()].trim(),
        "second paragraph\n\nthird paragraph"
    );
    assert_eq!(
        string[chunks[2].clone()].trim(),
        "third paragraph\n\nfourth paragraph"
    );
}

impl Chunker for ChunkStrategy {
    type Error<E: Send + Sync + 'static> = E;

    async fn chunk<E: Embedder + Send>(
        &self,
        document: &Document,
        embedder: &E,
    ) -> Result<Vec<Chunk>, E::Error> {
        let mut chunks = Vec::new();
        let body = document.body();
        let mut documents = Vec::new();
        let chunk_ranges = self.chunk_str(body);
        for byte_range in &chunk_ranges {
            documents.push(document.body()[byte_range.clone()].to_string());
        }
        let embeddings = embedder.embed_vec(documents).await?;
        for (byte_range, embedding) in chunk_ranges.into_iter().zip(embeddings) {
            chunks.push(Chunk {
                byte_range,
                embeddings: vec![embedding],
            });
        }
        Ok(chunks)
    }

    async fn chunk_batch<'a, I, E: Embedder + Send>(
        &self,
        documents: I,
        embedder: &E,
    ) -> Result<Vec<Vec<Chunk>>, E::Error>
    where
        I: IntoIterator<Item = &'a Document> + Send,
        I::IntoIter: Send,
    {
        let mut chunks = Vec::new();
        let mut chunk_strings = Vec::new();
        for document in documents {
            let body = document.body();
            let chunk = self.chunk_str(body);
            for byte_range in &chunk {
                chunk_strings.push(body[byte_range.clone()].to_string());
            }
            chunks.push(chunk);
        }

        let mut embeddings = embedder.embed_vec(chunk_strings).await?;
        let mut embeddings = embeddings.drain(..);
        let mut embedded_chunks = Vec::new();

        for chunk in chunks {
            let mut document_chunks = Vec::new();
            for byte_range in chunk {
                let embedding = embeddings.next().unwrap();
                document_chunks.push(Chunk {
                    byte_range,
                    embeddings: vec![embedding],
                });
            }
            embedded_chunks.push(document_chunks);
        }

        Ok(embedded_chunks)
    }
}
