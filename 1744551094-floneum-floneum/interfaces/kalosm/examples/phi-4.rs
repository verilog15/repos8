use kalosm::language::*;

#[tokio::main]
async fn main() {
    let model = Llama::builder()
        .with_source(LlamaSource::phi_4())
        .build()
        .await
        .unwrap();
    let mut chat = model
        .chat()
        .with_system_prompt("You will act like a pirate");

    loop {
        chat(&prompt_input("\n> ").unwrap())
            .to_std_out()
            .await
            .unwrap();
    }
}
