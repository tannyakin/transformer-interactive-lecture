// ============================================================
// Lecture 2 glossary: every abbreviation from Part 12 of the
// notes. Schema: { term, full, meaning }. `term` is the exact
// text used in <abbr data-term="..."> on the page.
// js/core/glossary.js reads Lecture.glossary for the tooltips
// and the glossary list at the end of the lecture.
// ============================================================
(function (root) {
  "use strict";

  const glossary = [
    { term: "ALiBi", full: "Attention with Linear Biases", meaning: "A fixed penalty, growing with distance, added to the attention scores so far-away words count for less." },
    { term: "ALBERT", full: "A Lite BERT", meaning: "A slimmer BERT that shares weights across its layers, so it needs far fewer parameters." },
    { term: "BART", full: "Bidirectional and Auto-Regressive Transformer", meaning: "An encoder-decoder model trained to repair text that has been deliberately corrupted." },
    { term: "BERT", full: "Bidirectional Encoder Representations from Transformers", meaning: "An encoder-only model pretrained by guessing hidden words from both sides of the sentence." },
    { term: "BPE", full: "Byte-Pair Encoding", meaning: "A tokenizer that keeps merging the most frequent pairs of characters into bigger tokens." },
    { term: "ByT5", full: "Byte-level T5", meaning: "A version of T5 that reads raw bytes instead of tokens." },
    { term: "CNN", full: "Convolutional Neural Network", meaning: "An image model built from small sliding filters; it is where the idea of a \"receptive field\" comes from." },
    { term: "C4", full: "Colossal Clean Crawled Corpus", meaning: "The cleaned-up web text that T5 was trained on." },
    { term: "[CLS]", full: "classification token", meaning: "The first token in BERT's input; its final vector acts as a summary of the whole input." },
    { term: "DeBERTa", full: "Decoding-enhanced BERT with disentangled attention", meaning: "A BERT variant that keeps what a word is and where it sits as two separate signals." },
    { term: "ELECTRA", full: "Efficiently Learning an Encoder that Classifies Token Replacements Accurately", meaning: "An encoder that learns by spotting which tokens in a sentence have been swapped for fakes." },
    { term: "FFN", full: "Feedforward Network", meaning: "A small two-layer network applied to each token on its own, after attention." },
    { term: "FLOPs", full: "Floating Point Operations", meaning: "A count of arithmetic operations, used to measure how much compute something needs." },
    { term: "GELU", full: "Gaussian Error Linear Unit", meaning: "A smooth activation function, used in BERT and GPT-2." },
    { term: "GeGLU", full: "GELU Gated Linear Unit", meaning: "A gated feedforward network that uses GELU for its gate." },
    { term: "GLU", full: "Gated Linear Unit", meaning: "A feedforward design with a gate that turns each feature up or down." },
    { term: "GLUE", full: "General Language Understanding Evaluation", meaning: "A benchmark made of several language understanding tasks." },
    { term: "GPT", full: "Generative Pre-trained Transformer", meaning: "OpenAI's family of decoder-only models." },
    { term: "GQA", full: "Grouped-Query Attention", meaning: "Query heads are split into groups, and each group shares one key head and one value head." },
    { term: "HBM", full: "High Bandwidth Memory", meaning: "The GPU's large main memory: roomy, but slower to reach than on-chip memory." },
    { term: "KL", full: "Kullback-Leibler divergence", meaning: "Measures how different two probability distributions are; it is the loss used in distillation." },
    { term: "KV cache", full: "Key-Value cache", meaning: "The keys and values of earlier tokens, stored during generation so they are not recomputed." },
    { term: "L, H, A", full: "Layers, Hidden size, Attention heads", meaning: "The three numbers BERT uses to describe how big a model is." },
    { term: "LLM", full: "Large Language Model", meaning: "A very large language model, usually decoder-only." },
    { term: "LN", full: "Layer Normalization", meaning: "Rescales each token's vector to a standard scale so training stays steady." },
    { term: "LSTM", full: "Long Short-Term Memory", meaning: "An improved RNN with gates; the leading sequence model before transformers." },
    { term: "MHA", full: "Multi-Head Attention", meaning: "Several attention heads side by side, each with its own Q, K and V." },
    { term: "MLA", full: "Multi-head Latent Attention", meaning: "Compresses K and V into a smaller form before caching them (used by DeepSeek)." },
    { term: "MLM", full: "Masked Language Modeling", meaning: "Hide some words and predict them using the context on both sides." },
    { term: "MoE", full: "Mixture of Experts", meaning: "Many feedforward networks per layer, with a router that picks a few of them for each token." },
    { term: "MQA", full: "Multi-Query Attention", meaning: "All query heads share a single key head and a single value head." },
    { term: "mT5", full: "multilingual T5", meaning: "T5 trained on text in over 100 languages." },
    { term: "[MASK]", full: "mask token", meaning: "Stands in for the hidden tokens in BERT's masked word task." },
    { term: "[PAD]", full: "padding token", meaning: "Fills unused positions so every input in a batch has the same length." },
    { term: "NER", full: "Named Entity Recognition", meaning: "Finding the names of people, places and organizations in text." },
    { term: "NSP", full: "Next Sentence Prediction", meaning: "BERT's second task: does sentence B really follow sentence A?" },
    { term: "RAG", full: "Retrieval-Augmented Generation", meaning: "Fetch the relevant documents first, then let an LLM answer using them." },
    { term: "ReLU", full: "Rectified Linear Unit", meaning: "An activation where negatives become 0 and positives pass through unchanged." },
    { term: "RLHF", full: "Reinforcement Learning from Human Feedback", meaning: "Training a model toward the responses people prefer." },
    { term: "RMSNorm", full: "Root Mean Square Normalization", meaning: "Layer norm without subtracting the mean: cheaper, and works just as well." },
    { term: "RNN", full: "Recurrent Neural Network", meaning: "A model that reads a sequence one step at a time, carrying a memory forward." },
    { term: "RoBERTa", full: "Robustly optimized BERT approach", meaning: "BERT trained with a much better recipe: more data, longer training, no NSP." },
    { term: "RoPE", full: "Rotary Position Embedding", meaning: "Rotates q and k by an angle that depends on position, so attention sees the distance between words." },
    { term: "[SEP]", full: "separator token", meaning: "Marks the end of a sentence in BERT's input." },
    { term: "SFT", full: "Supervised Fine-Tuning", meaning: "Training on example inputs paired with ideal outputs." },
    { term: "SOP", full: "Sentence Order Prediction", meaning: "ALBERT's task: are these two sentences in the right order?" },
    { term: "SQuAD", full: "Stanford Question Answering Dataset", meaning: "A benchmark where the model finds the answer inside a given passage." },
    { term: "SRAM", full: "Static Random Access Memory", meaning: "The GPU's small but very fast on-chip memory." },
    { term: "SwiGLU", full: "Swish Gated Linear Unit", meaning: "The gated feedforward network used in most modern LLMs." },
    { term: "SWA", full: "Sliding Window Attention", meaning: "Each token attends only to a window of nearby tokens." },
    { term: "T5", full: "Text-to-Text Transfer Transformer", meaning: "An encoder-decoder that treats every task as text in, text out." },
    { term: "YaRN", full: "Yet another RoPE extensioN", meaning: "A method for stretching RoPE so a model can handle longer contexts." },
  ];

  root.Lecture = root.Lecture || {};
  root.Lecture.glossary = glossary;
  if (typeof module !== "undefined" && module.exports) module.exports = glossary;
})(typeof window !== "undefined" ? window : globalThis);
