# Lumina: Automated Critique System for Research Papers

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NLP](https://img.shields.io/badge/NLP-Peer%20Review%20Automation-green)

A specialized system for generating expert-level critiques of academic research papers using Natural Language Processing (NLP) and Large Language Models (LLMs).

## Project Overview

Lumina presents a focused system that automates the critique of academic research papers. By leveraging state-of-the-art NLP techniques and LLMs, the system performs targeted analysis of research papers, including:

- PDF document parsing and section extraction
- Paragraph-level scholarly critique generation
- Quality assessment of logical flow and argumentation
- Targeted feedback on academic writing style and clarity

The system aims to streamline the peer review process and support authors in producing higher-quality research papers by providing detailed, actionable feedback.

## Key Features

- **Rhetorical Role Classification**: Identifies the purpose and structure of different sections in a research paper
- **Context-Aware Critique**: Provides specific, actionable suggestions for improving academic tone, clarity, grammar, and logical consistency
- **Multi-dimensional Feedback**: Addresses various aspects of academic writing including evidence strength, logical coherence, and clarity of arguments

## Dataset Creation and Preparation

A significant contribution of this project is the creation of a specialized dataset for academic paper critique:

1. **Diverse Source Collection**: We systematically gathered papers from arXiv across 10 distinct academic domains:
   - Artificial Intelligence (cs.AI)
   - Computational Linguistics (cs.CL)
   - Computer Vision (cs.CV)
   - Machine Learning (cs.LG, stat.ML)
   - Software Engineering (cs.SE)
   - Computational Physics (physics.comp-ph)
   - Quantitative Biology (q-bio.QM)
   - Statistical Finance (q-fin.ST)
   - Human-Computer Interaction (cs.HC)

2. **Intelligent Paragraph Extraction**: 
   - Implemented PyMuPDF-based extraction with heuristic filtering
   - Applied length constraints (50-300 words) to focus on substantive content
   - Excluded non-content elements like captions, references, and headers
   - Detected and skipped front matter by identifying introduction sections

3. **High-Quality Critique Generation**: 
   - Leveraged Meta-Llama-3-8B-Instruct to generate expert-level critiques
   - Designed detailed prompts focused on specific issue types to create diverse feedback:
     ```
     You are an expert academic reviewer with years of experience reviewing research papers.
     Analyze the following paragraph from a real research paper and provide a detailed critique.
     Focus on identifying issues related to: [specific_issue_type].
     If you genuinely find no issues, explain why the paragraph is well-written instead.
     Your critique should be specific, actionable, and professional.
     ```
   - Balanced critique types across 11 categories: missing evidence, logical contradictions, unclear arguments, poor citations, grammar/spelling, undefined terminology, statistical errors, methodology issues, unsubstantiated claims, structural problems, and well-written examples

4. **Optimized Format for Fine-tuning**:
   - Structured data in specialized chat format for Mistral
   - Created consistent user/assistant message pairs
   - Applied train-validation split (90/10) with stratification across issue types
   - Performed quality checks on generated critiques for coherence and helpfulness

5. **Dataset Analysis and Validation**:
   - Tracked distribution of issue types to ensure balanced representation
   - Calculated and monitored metrics including average paragraph length (150 words) and average critique length (180 words)
   - Manually reviewed samples to verify critique quality and relevance

This custom dataset creation approach ensures that the fine-tuned model can provide varied, high-quality critiques targeting specific aspects of academic writing that commonly need improvement.

## Model Fine-tuning

The critique generation component was developed through a parameter-efficient fine-tuning process:

1. **Base Model**: Mistral-7B-Instruct-v0.1 was used as the foundation model due to its strong instruction-following capabilities and performance-to-size ratio.

2. **Quantization**: 4-bit quantization via BitsAndBytes (NF4 format) was applied to enable training on consumer-grade GPU hardware with limited VRAM.

3. **LoRA Configuration**: 
   - Low-Rank Adaptation with r=16 and alpha=16
   - Target modules: query, key, value, and output projection matrices (`q_proj`, `v_proj`)
   - LoRA dropout of 0.05 for regularization

4. **Training Parameters**:
   - Learning rate: 2e-4
   - Batch size: 1 with gradient accumulation steps of 8
   - Training epochs: 3
   - Warmup ratio: 0.03
   - Max gradient norm: 0.3
   - Training utilized the Mistral chat template format

5. **Memory Efficiency**: The combination of 4-bit quantization and parameter-efficient fine-tuning (PEFT) enabled the training process to run on a single consumer GPU with 12-24GB VRAM.

## System Architecture

The system follows a streamlined pipeline:

1. **PDF Processing and Text Extraction**:
   - Utilizes PyMuPDF and PyPDF2 for robust PDF parsing
   - Implements regex-based cleaning and paragraph extraction
   - Filters non-content elements like short paragraphs and captions

2. **Critique Generation**:
   - Implements paragraph-level scholarly critique generation
   - Uses temperature-controlled sampling (0.7-0.9) for balanced feedback
   - Formats prompts using Mistral's chat template for instruction following
   - Provides actionable feedback on academic writing quality
   - Highlights specific strengths and weaknesses in each paragraph

## Technologies Used

- **Python**: Core programming language
- **Hugging Face Transformers**: For implementing pre-trained models
- **PyMuPDF & PyPDF2**: PDF parsing and text extraction
- **Mistral-7B-Instruct-v0.1**: Base model fine-tuned for academic critique generation
- **PEFT & QLoRA**: Memory-efficient parameter-efficient fine-tuning of LLMs
- **BitsAndBytes**: 4-bit quantization for running LLMs on consumer hardware


### Core Modules Implementation

1. **`fine tuning/mistral qlora`**: Implements QLoRA fine-tuning for Mistral-7B with the following features:
   - 4-bit quantization with NormalFloat4
   - Parameter-efficient training with r=16
   - Custom dataset loading and formatting
   - Checkpointing and model saving

2. **`pdf processing`**: Implements PDF text extraction with:
   - PyMuPDF and PyPDF2 integration
   - Regex-based section detection
   - Paragraph filtering and cleaning

3. **`critique generator`**: Implements the critique system with:
   - Prompt engineering for academic critique
   - Temperature-controlled LLM generation
   - Context window management

## Future Work

- Reduce hallucination in model-generated responses through confidence-based filtering or ensemble review models
- Expand the critique scope to include citation diversity and reference formatting
- Create a unified web application with drag-and-drop PDF support
- Develop domain-specific critique models for specialized fields (biomedical, legal, etc.)
- Incorporate user feedback mechanisms to improve critique quality over time


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- arXiv for sample papers
