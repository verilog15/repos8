use floneum_rust::*;

#[export_plugin]
/// Adds a embedding to a database. The model used to generate the embedding and the model type used to create the database must be the same.
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![EmbeddingDb::new(&[], &[]).into_input_value(), Embedding { vector: vec![0.0, 0.0, 0.0] }.into_input_value(), String::from("Text to embed").into_input_value()],
///         outputs: vec![],
///     },
/// ]
fn add_embedding(
    /// the database to add the embedding to
    database: EmbeddingDb,
    /// the embedding to add
    embedding: Embedding,
    /// the value to associate with the embedding
    value: String,
) {
    database.add_embedding(&embedding, &value);
}
