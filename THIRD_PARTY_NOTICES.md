# Third-party code and model provenance

The project includes the following upstream source files for architecture compatibility and small implementation tests. Their original notices are preserved. Model weights and tokenizer downloads are not distributed in this repository.

## Ouro

- Upstream: [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)
- Fixed revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`
- Included: `ouro_depth/vendor/configuration_ouro.py`, `ouro_depth/vendor/modeling_ouro.py`.
- The pinned upstream model card declares Apache-2.0. The configuration source also retains the Qwen/Alibaba/Hugging Face copyright and Apache-2.0 header.

The applicable Apache-2.0 license text is included at [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). This notice covers the identified third-party material; it does not grant a new license over the project's original research code. The cache implementations, evaluation utilities and research notes outside the vendor directory are project-specific work.
