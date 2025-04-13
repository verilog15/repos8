# Floneum
# [Download the latest release](https://github.com/floneum/floneum/releases/tag/v0.2.0)

Floneum is a graph editor for AI workflows with a focus on community-made plugins, local AI and safety.

<img width="1512" alt="Screenshot 2023-06-18 at 4 26 11 PM" src="https://floneum.com/assets/question_answer_example.png">

## Features

- Visual interface: You can use Floneum without any knowledge of programming. The visual graph editor makes it easy to combine community-made plugins with local AI models
- Quickly run local large language models: Floneum does not require any external dependencies or even a GPU to run. It uses [Candle](https://github.com/huggingface/candle) to run quantized versions of large language models locally. Because of this, you can run models in Floneum without worrying about privacy
- Plugins: By combining large language models with plugins, you can improve their performance and make models work better for your specific use case. All plugins run in an isolated environment so you don't need to trust any plugins you load. Plugins can only interact with their environment in a safe way
- Multi-language plugins: Plugins can be used in any language that supports web assembly. In addition to the API that can be accessed in any language, Floneum has a rust wrapper with ergonomic macros that make it simple to create plugins
- Controlled text generation: Plugins can control the output of the large language models with a process similar to JSONformer or guidance. This allows plugins to force models to output valid JSON, or any other structure they define. This can be useful when communicating between a language model and a typed API

## Documentation

- If you are looking to use Floneum, you can read the [User Documentation](https://floneum.com/docs/user/).

- If you are looking to develop plugins for Floneum, you can read the [Developer Documentation](https://floneum.com/docs/developer/)

## Community

If you are interested in Floneum, you can join the [discord](https://discord.gg/dQdmhuB8q5) to discuss the project and get help.

## Contributing

- Report issues on our [issue tracker](https://github.com/floneum/floneum/issues).
- Help other users in the Floneum discord
- If you are interested in contributing, reach out on discord

## Building default plugins

```sh
floneum build --release --packages floneum_add_embedding,floneum_embedding,floneum_embedding_db,floneum_format,floneum_generate_text,floneum_generate_structured_text,floneum_search,floneum_search_engine,floneum_if,floneum_contains,floneum_write_to_file,floneum_read_from_file,floneum_python,floneum_find_node,floneum_find_child_node,floneum_click_node,floneum_node_text,floneum_type_in_node,floneum_navigate_to,floneum_get_article,floneum_read_rss,floneum_split,floneum_slice,floneum_join,floneum_add_to_list,floneum_new_list,floneum_length,floneum_more_than,floneum_less_than,floneum_equals,floneum_and,floneum_or,floneum_calculate,floneum_not,floneum_add,floneum_subtract,floneum_multiply,floneum_divide,floneum_power,floneum_number,floneum_string
```

## Building the UI

```
npx tailwindcss -i ./input.css -o ./public/tailwind.css --watch
cargo run --release --target aarch64-apple-darwin # Or whatever the target triple for your current device is
```
