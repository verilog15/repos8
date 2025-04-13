use crate::plugins::main;
use crate::resource::ResourceStorage;
use crate::Both;
use main::imports::{self};
use main::types::{EmbeddingDbResource, EmbeddingModelResource, TextGenerationModelResource};
use std::ops::Deref;

use kalosm::language::DynamicNodeId;
use once_cell::sync::Lazy;

use reqwest::header::{HeaderName, HeaderValue};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use wasmtime::component::__internal::async_trait;
use wasmtime::component::{Linker, ResourceTable};
use wasmtime::Config;
use wasmtime::Engine;
use wasmtime_wasi::bindings::Command;
use wasmtime_wasi::WasiCtxBuilder;
use wasmtime_wasi::{self, DirPerms, FilePerms, WasiCtx, WasiView};

pub(crate) static LINKER: Lazy<Linker<State>> = Lazy::new(|| {
    let mut linker = Linker::new(&ENGINE);
    let l = &mut linker;
    Both::add_to_linker(l, |x| x).unwrap();
    Command::add_to_linker(l, |x| x).unwrap();

    linker
});
pub(crate) static ENGINE: Lazy<Engine> = Lazy::new(|| {
    let mut config = Config::new();
    config.wasm_component_model(true).async_support(true);
    Engine::new(&config).unwrap()
});

#[derive(Clone, Copy)]
pub(crate) struct AnyNodeRef {
    pub(crate) node_id: DynamicNodeId,
    pub(crate) page_id: usize,
}

#[derive(Clone)]
pub struct SharedPluginState {
    pub(crate) logs: Arc<RwLock<Vec<String>>>,
    pub(crate) resources: ResourceStorage,
}

impl SharedPluginState {
    /// Create shared plugins state from the resources
    pub fn new(resources: ResourceStorage) -> Self {
        Self {
            resources,
            logs: Default::default(),
        }
    }
}

pub struct State {
    pub(crate) shared: SharedPluginState,
    pub(crate) plugin_state: HashMap<Vec<u8>, Vec<u8>>,
    pub(crate) table: ResourceTable,
    pub(crate) ctx: WasiCtx,
}

impl Deref for State {
    type Target = SharedPluginState;

    fn deref(&self) -> &Self::Target {
        &self.shared
    }
}

impl State {
    pub fn new(shared: SharedPluginState) -> Self {
        let sandbox = Path::new("./sandbox");
        std::fs::create_dir_all(sandbox).unwrap();
        let mut ctx = WasiCtxBuilder::new();
        let ctx_builder = ctx
            .inherit_stderr()
            .inherit_stdin()
            .inherit_stdio()
            .inherit_stdout()
            .preopened_dir(sandbox, "./", DirPerms::all(), FilePerms::all())
            .unwrap();
        let table = ResourceTable::new();
        let ctx = ctx_builder.build();
        State {
            plugin_state: Default::default(),
            shared,
            table,
            ctx,
        }
    }
}

impl WasiView for State {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }

    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.ctx
    }
}

// This implementation defines the interface for the plugin to use.
// Most functions call out to another method for the implementation so that this file doesn't get too long.
#[async_trait]
impl main::types::Host for State {
    async fn get_request(
        &mut self,
        url: String,
        headers: Vec<main::types::Header>,
    ) -> std::result::Result<String, wasmtime::Error> {
        let mut headers = headers
            .into_iter()
            .map(|header| {
                Ok((
                    HeaderName::try_from(header.key)?,
                    HeaderValue::from_str(&header.value)?,
                ))
            })
            .collect::<wasmtime::Result<Vec<_>>>()?;
        headers.push((
            HeaderName::from_static("user-agent"),
            HeaderValue::from_static("floneum"),
        ));
        let res = reqwest::Client::new()
            .get(&url)
            .headers(reqwest::header::HeaderMap::from_iter(headers))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        Ok(res)
    }

    async fn create_page(
        &mut self,
        mode: main::types::BrowserMode,
        url: String,
    ) -> wasmtime::Result<main::types::PageResource> {
        self.resources.impl_create_page(mode, url)
    }

    async fn find_in_current_page(
        &mut self,
        self_: main::types::PageResource,
        query: String,
    ) -> wasmtime::Result<main::types::NodeResource> {
        self.resources.impl_find_in_current_page(self_, query).await
    }

    async fn screenshot_browser(
        &mut self,
        self_: main::types::PageResource,
    ) -> wasmtime::Result<Vec<u8>> {
        self.resources.impl_screenshot_browser(self_).await
    }

    async fn page_html(&mut self, self_: main::types::PageResource) -> wasmtime::Result<String> {
        self.resources.impl_page_html(self_).await
    }

    async fn drop_node(&mut self, self_: main::types::NodeResource) -> wasmtime::Result<()> {
        self.resources.impl_drop_node(self_)
    }

    async fn drop_page(&mut self, self_: main::types::PageResource) -> wasmtime::Result<()> {
        self.resources.impl_drop_page(self_)
    }

    async fn get_element_text(
        &mut self,
        self_: main::types::NodeResource,
    ) -> wasmtime::Result<String> {
        self.resources.impl_get_element_text(self_).await
    }

    async fn click_element(&mut self, self_: main::types::NodeResource) -> wasmtime::Result<()> {
        self.resources.impl_click_element(self_).await
    }

    async fn type_into_element(
        &mut self,
        self_: main::types::NodeResource,
        keys: String,
    ) -> wasmtime::Result<()> {
        self.resources.impl_type_into_element(self_, keys).await
    }

    async fn get_element_outer_html(
        &mut self,
        self_: main::types::NodeResource,
    ) -> wasmtime::Result<String> {
        self.resources.impl_get_element_outer_html(self_).await
    }

    async fn screenshot_element(
        &mut self,
        self_: main::types::NodeResource,
    ) -> wasmtime::Result<Vec<u8>> {
        self.resources.impl_screenshot_element(self_).await
    }

    async fn find_child_of_element(
        &mut self,
        self_: main::types::NodeResource,
        query: String,
    ) -> wasmtime::Result<main::types::NodeResource> {
        self.resources
            .impl_find_child_of_element(self_, query)
            .await
    }

    async fn create_embedding_db(
        &mut self,
        embeddings: Vec<main::types::Embedding>,
        documents: Vec<String>,
    ) -> wasmtime::Result<EmbeddingDbResource> {
        Ok(self
            .resources
            .impl_create_embedding_db(embeddings, documents)?)
    }

    async fn drop_embedding_db(&mut self, rep: EmbeddingDbResource) -> wasmtime::Result<()> {
        self.resources.impl_drop_embedding_db(rep)
    }

    async fn add_embedding(
        &mut self,
        self_: EmbeddingDbResource,
        embedding: main::types::Embedding,
        document: String,
    ) -> wasmtime::Result<()> {
        self.resources
            .impl_add_embedding(self_, embedding, document)
            .await
    }

    async fn find_closest_documents(
        &mut self,
        self_: EmbeddingDbResource,
        search: main::types::Embedding,
        count: u32,
    ) -> wasmtime::Result<Vec<String>> {
        self.resources
            .impl_find_closest_documents(self_, search, count)
            .await
    }

    async fn create_model(
        &mut self,
        ty: main::types::ModelType,
    ) -> wasmtime::Result<TextGenerationModelResource> {
        Ok(self.resources.impl_create_text_generation_model(ty))
    }

    async fn drop_model(
        &mut self,
        model: main::types::TextGenerationModelResource,
    ) -> wasmtime::Result<()> {
        self.resources.impl_drop_text_generation_model(model)
    }

    async fn text_generation_model_downloaded(
        &mut self,
        ty: main::types::ModelType,
    ) -> wasmtime::Result<bool> {
        self.resources
            .impl_text_generation_model_downloaded(ty)
            .await
    }

    async fn infer(
        &mut self,
        self_: TextGenerationModelResource,
        input: String,
        max_tokens: Option<u32>,
        stop_on: Option<String>,
    ) -> wasmtime::Result<String> {
        self.resources
            .impl_infer(self_, input, max_tokens, stop_on)
            .await
    }

    async fn infer_structured(
        &mut self,
        self_: TextGenerationModelResource,
        input: String,
        regex: String,
    ) -> wasmtime::Result<String> {
        self.resources
            .impl_infer_structured(self_, input, regex)
            .await
    }

    async fn create_embedding_model(
        &mut self,
        ty: main::types::EmbeddingModelType,
    ) -> wasmtime::Result<EmbeddingModelResource> {
        self.resources.impl_create_embedding_model(ty)
    }

    async fn drop_embedding_model(
        &mut self,
        model: EmbeddingModelResource,
    ) -> wasmtime::Result<()> {
        self.resources.impl_drop_embedding_model(model)
    }

    async fn embedding_model_downloaded(
        &mut self,
        ty: main::types::EmbeddingModelType,
    ) -> wasmtime::Result<bool> {
        self.resources.impl_embedding_model_downloaded(ty).await
    }

    async fn get_embedding(
        &mut self,
        self_: EmbeddingModelResource,
        document: String,
    ) -> wasmtime::Result<main::types::Embedding> {
        self.resources.impl_get_embedding(self_, document).await
    }
}

#[async_trait]
impl imports::Host for State {
    async fn log_to_user(&mut self, message: String) -> std::result::Result<(), wasmtime::Error> {
        let mut logs = self
            .logs
            .write()
            .map_err(|e| wasmtime::Error::msg(format!("Failed to lock logs: {}", e)))?;
        if logs.len() >= 100 {
            logs.remove(0);
        }
        logs.push(message);
        Ok(())
    }

    async fn store(
        &mut self,
        key: Vec<u8>,
        value: Vec<u8>,
    ) -> std::result::Result<(), wasmtime::Error> {
        self.plugin_state.insert(key, value);

        Ok(())
    }

    async fn load(&mut self, key: Vec<u8>) -> std::result::Result<Vec<u8>, wasmtime::Error> {
        Ok(self.plugin_state.get(&key).cloned().unwrap_or_default())
    }

    async fn unload(&mut self, key: Vec<u8>) -> std::result::Result<(), wasmtime::Error> {
        self.plugin_state.remove(&key);
        Ok(())
    }
}
