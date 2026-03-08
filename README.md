# Molecular Property Prediction: HOMO-LUMO Gap with GATv2 GNN

This repository contains the code and implementation details for a Graph Neural Network (GNN) model designed to predict the HOMO-LUMO gap of molecules, a key quantum chemical property. The model was developed as part of a research project at UCLA.

## Project Overview

The goal of this project was to develop a model that surpasses the performance of the standard GIN-virtual baseline on the PCQM4Mv2 dataset. The final model, a 7-layer GATv2 network with a virtual node, achieved a **validation MAE of 0.1052 eV**, outperforming the GIN-virtual baseline (0.1083 eV).

## Key Features & Achievements

- **Custom GATv2 Architecture:** Implemented a Graph Attention Network (GATv2) to better incorporate edge features (chemical bonds) into the message-passing process, overcoming the limitations of simpler aggregation methods like GIN.
- **Virtual Node Enhancement:** Integrated a learnable virtual node embedding to enable global graph-level information sharing across all atoms, which significantly stabilized training and improved accuracy.
- **Jumping Knowledge:** Applied Jumping Knowledge concatenation across all 7 GAT layers to preserve multi-scale structural features and prevent over-smoothing.
- **Advanced Training Techniques:** Leveraged mixed-precision training (via `torch.cuda.amp`) and a `ReduceLROnPlateau` scheduler to optimize memory usage and convergence speed on a Google Cloud NVIDIA T4 GPU.
- **Ablation Studies:** Conducted systematic experiments to isolate the marginal effects of the virtual node and RDKit global descriptors on validation MAE.
- **Comprehensive Analysis:** Evaluated the model with loss curves, a predictions vs. true values scatter plot, t-SNE visualization of learned embeddings, and an error analysis by molecule size.

## Model Architecture

- **Core Layers:** 7-layer GATv2Conv
- **Hidden Dimension:** 256
- **Readout:** Global add pooling with Jumping Knowledge (concatenation)
- **Virtual Node:** A learnable node connected to all others for global context.
- **Input Features:** Node features (atoms) and edge features (bonds) as provided by the OGB `smiles2graph` function.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Geometric
- OGB (Open Graph Benchmark)
- RDKit
- Other dependencies listed in `requirements.txt`

### Installation & Data

1.  Clone the repository.
2.  Install the required packages: `pip install -r requirements.txt`
3.  The PCQM4Mv2 dataset will be downloaded automatically by the OGB package upon first run.

### Training the Model

To train the model on a 2-million-molecule subset, run the main training script:

```bash
python train.py
