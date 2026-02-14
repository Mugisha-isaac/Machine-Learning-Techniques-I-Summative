# Finance Assistant: Domain-Specific LLM Fine-Tuning

## Overview
This project demonstrates the process of fine-tuning a pre-trained language model for a finance assistant chatbot. The notebook is designed for Google Colab and leverages parameter-efficient fine-tuning (LoRA) to adapt a large language model to the finance domain. The workflow includes data preparation, model training, evaluation, and deployment via a user-friendly web interface.

## Project Structure
- **Finance_Assistant_Refactored.ipynb**: Main notebook containing the complete pipeline.
- **README.md**: Project documentation and instructions.

## Workflow Summary
1. **Install Dependencies**: All required libraries are installed for Colab compatibility.
2. **Import Libraries & Environment Setup**: Essential packages for modeling, visualization, and web deployment are imported.
3. **Dataset Preparation**: A curated finance instruction-response dataset is formatted and split for training and evaluation.
4. **Function & Class Definitions**: Helper functions and callback classes are defined for training and inference.
5. **Model Setup & Fine-Tuning**: A pre-trained LLM is loaded and fine-tuned using LoRA. Hyperparameters are documented.
6. **Evaluation & Visualization**: Training and validation loss, learning rate, and perplexity are tracked and visualized. BLEU score is computed for additional evaluation. Visualizations include:
   - Training Loss Curve
   - Learning Rate Schedule
   - Train vs Validation Loss
   - Loss Reduction Rate
7. **Inference & Demo**: The fine-tuned model is tested with sample questions and deployed via a Gradio chatbot interface.
8. **Experiment Table**: Hyperparameter experiments, GPU usage, and training time are summarized.
9. **Qualitative Comparison**: Responses from the base and fine-tuned models are compared for key questions.
10. **Methodology & Insights**: The approach and key findings are documented.

## Key Features
- Parameter-efficient fine-tuning (LoRA) for resource-constrained environments
- Domain-specific finance dataset for improved relevance
- Quantitative (loss, perplexity, BLEU) and qualitative evaluation
- User-friendly web interface for real-time interaction
- Experiment tracking for reproducibility

## Instructions
1. Open the notebook in Google Colab.
2. Run all cells sequentially for a complete workflow.
3. Review experiment results and evaluation metrics.
4. Interact with the deployed chatbot for demonstration.

## Requirements
- Google Colab (recommended)
- Python 3.8+
- Packages: torch, transformers, datasets, peft, trl, gradio, matplotlib, seaborn, scikit-learn, nltk, bitsandbytes

## Methodology
- Fine-tuning is performed using LoRA to efficiently adapt a pre-trained LLM to the finance domain.
- The dataset consists of instruction-response pairs covering a range of finance topics.
- Model performance is evaluated using loss, perplexity, BLEU, and qualitative comparisons.
- The Gradio interface enables easy user interaction and demonstration.

## Insights
- Domain adaptation improves response quality and relevance for finance queries.
- LoRA enables efficient training on limited hardware.
- Hyperparameter tuning impacts both performance and resource usage.
- Combining quantitative and qualitative metrics provides a comprehensive evaluation.

## License
This project is for educational purposes. Please review the license terms of any pre-trained models and datasets used.

## Contact
For questions or feedback, please contact the project maintainer.