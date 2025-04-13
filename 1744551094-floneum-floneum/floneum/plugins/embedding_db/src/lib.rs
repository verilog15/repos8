use floneum_rust::*;

#[export_plugin]
/// Creates a database of embeddings. (A database is just a different way to store information, in this cases this stores documents in a way that makes it easy to find other documents with similar meanings)
///
/// When using this embedding database, you must use the same model to generate the embeddings you insert into this database.
///
/// ### Examples
/// vec![
///     Example {
///         name: "example".into(),
///         inputs: vec![ModelType::LlamaSevenChat.into_input_value(), vec![String::from("Text to embed"), String::from("Another text to embed")].into_input_value()],
///         outputs: vec![EmbeddingDb::new(&[], &[]).into_return_value()],
///     },
/// ]
fn embedding_db(
    /// the model to use
    model: EmbeddingModelType,
    /// the documents to index
    documents: Vec<String>,
) -> EmbeddingDb {
    let instance = EmbeddingModel::new(model);

    let embeddings = documents
        .iter()
        .map(|s| instance.get_embedding(s))
        .collect::<Vec<_>>();

    EmbeddingDb::new(&embeddings, &documents)
}
