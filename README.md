# Agent-Learning
Agent学习：[Agent-Learning-Hub](https://github.com/datawhalechina/Agent-Learning-Hub)


### Stage 1: Build A Minimal Agent Loop

- [ ] 会用一个 LLM API 完成普通对话。
- [ ] 会让模型输出结构化 JSON。
- [ ] 会定义一个工具函数，例如 search、calculator、read_file。
- [ ] 会解析模型的 tool call / function call。
- [ ] 会执行工具，并把工具结果喂回模型。
- [ ] 会给 agent loop 加最大步数、超时和错误处理。

推荐阅读：

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Gemini API Function Calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Claude Tool Use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)

### Stage 2: Learn Tool Use, RAG, And Memory

- [ ] 会做检索增强生成：chunk、embed、retrieve、answer with citations。
- [ ] 会把搜索、数据库、文件、浏览器、代码执行接成工具。
- [ ] 会区分短期上下文、会话记忆、长期记忆。
- [ ] 会处理工具失败、空结果、重复调用、幻觉引用。
- [ ] 会让 agent 在回答里给出来源或证据。

推荐阅读：

- [LlamaIndex Agents](https://docs.llamaindex.ai/en/stable/use_cases/agents/)
- [LangChain Docs](https://docs.langchain.com/)
- [Gemini API Code Execution](https://ai.google.dev/gemini-api/docs/code-execution)
- [Model Context Protocol](https://modelcontextprotocol.io/)
