Financial Document Intelligence Pipeline RAG
Description
Build a RAG pipeline that answers questions across financial documents - SEC filings, earnings call transcripts, insurance claims, or loan documents. Implement two chunking strategies (fixed-size vs. semantic chunking) and compare retrieval quality on the same set of queries. Add a reranking step and measure the improvement. This project is particularly relevant for anyone working in financial services, insurance, or regulated industries, but the techniques apply to any domain with dense, structured documents. This project now has code and n8n in the Week 2 project list.
Best For
Data Scientists/Analysts (financial analysis), Finance roles (Wells Fargo, JPMC, Citi, Morgan Stanley, NYLife), Architects (pipeline design), Consultants (client financial analysis).
Deliverable
A working financial RAG pipeline with a chunking strategy comparison report and reranking impact analysis.
Submission
Demo recording + GitHub link or zip file + comparison report document.
Difficulty
Intermediate | Code and n8n options | PMs can define test questions and evaluate business relevance of retrieved answers
Add-on: Chatbot UI (Vibe Coded)
Description
Vibe code a chatbot and connect it as the front-end interface for your Week 2 RAG system. Your chatbot now has a real knowledge base behind it — turning a simple conversational UI into a grounded, document-backed assistant. This is how the weekly projects start compounding.

The Framework
Fill out every field in 1 to 2 sentences. The framework forces a decision at every layer of the RAG stack so nothing gets hand-waved.
Field
Fill in (1 to 2 sentences max)
Use case
(one line)
What question are you answering, who asks it, and where does it show up (Slack, widget, API, internal agent)?
Corpus
Sources, rough doc count, formats, language, and who owns the source of truth.
Ingestion + cleaning
How docs get in, and what cleaning happens before chunking (strip markup, drop boilerplate, decode entities)
Ingestion + freshness
How docs get in, refresh cadence, freshness SLA.
Chunking + embedding
Chunk strategy and size, semantic vs fixed, embedding model, and why.
Retrieve
Store, dense / sparse / hybrid, and top-k.



💡 Tips before you fill it in
Pick chunk size and embed the model together. A 512-token chunk on a 384-dim embedding is wasted; a 2000-token chunk on a small model loses signal. Match capacity.
Hybrid retrieval is usually right. Pure dense misses exact matches (error codes, names, ticker symbols). Pure BM25 misses semantic intent. Combine them and re-rank.
Your "I don't know" path matters more than your happy path. A RAG app that hallucinates when retrieval fails is worse than one that says "I could not find this in our docs." Design the refusal first.


Part 2: Pick Your Build Track


The framework is identical across tracks. Your track determines how you implement it. Pick by Tuesday so you have four full days to build.
How to decide
Default to Track 2 (LangChain + LangGraph) if you write code regularly. You will learn more about what is actually happening under the hood, and it generalizes to production systems.
Pick Track 1 (n8n) if you are non-technical, time-constrained, or your use case fits an available no-code path. 
Quick note: You’ll have access to Nebius credits, which you can use for model calls, including both embeddings and generation.


Track 1: No-code with n8n 
What it is
A no-code path where you build RAG without writing Python. In n8n, you wire AI Agent, Embeddings, Vector Store, and Document Loader nodes together. 
Best for
Rapid prototyping, integrations-heavy workflows (CRM, helpdesk, Slack), demos for non-technical stakeholders, and builders who want a guided no-code path.
Key building blocks
n8n: AI Agent node, Embeddings nodes (OpenAI, Cohere, HuggingFace), Vector Store nodes (Pinecone, Qdrant, Supabase), Document Loaders, prompt templates, Webhook triggers. .
Tradeoffs
Less control over chunking, reranking, and custom evals than code. You will trade depth for speed and guided setup.
Track 2: Code-heavy with LangChain + LangGraph
What it is
Python framework with low-level RAG primitives (loaders, splitters, embeddings, retrievers, chains) plus LangGraph for stateful, multi-step graph flows.
Best for
Production-grade systems, custom retrieval logic (parent-doc, self-query, multi-query), complex multi-hop or agentic RAG, anyone writing evals as code.
Key building blocks
DocumentLoaders, TextSplitters, Embeddings, VectorStores, Retrievers, LCEL chains, LangGraph state machines, LangSmith for tracing and evals.
Tradeoffs
Steeper ramp, more code to maintain, more decisions to make explicit. The upside is full control and a portfolio piece you can extend.

