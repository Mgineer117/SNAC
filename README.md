![SNAC logo](snac_logo.png)

# SNAC: Specialized Neurons and Architecture with Clustering

This repository presents a Hierarchical Reinforcement Learning (HRL) approach in Grid World, incorporating **Successor Features (SFs)** with the following enhancements:
- **Clustering in eigenspace** to prevent information loss by utilizing all computed eigenvectors.
- **Simultaneous reward and state feature decompositions** to adapt to both the reward structure and navigational diffusion properties of the environment.
- **Offering intuition and a foundation for Successor Features (SF) Implementation**, addressing the gap in the current field where most work focuses on Successor Representation (SR) and lacks comprehensive code and intuition for SF.

Additionally, this repository includes implementations of previous state-of-the-art (SOTA) methods, such as **EigenOption**, **CoveringOption**, and a naive **PPO** approach, serving as baselines. For further details, please refer to the workshop paper of the older version of [SNAC](https://ala2022.github.io/papers/ALA2022_paper_41.pdf).

---

## Notes on SNAC
- Due to the non-uniqueness of the sign by SVD decomposition, we count one eigenvector as two vectors such that e = (+e/-e).

## Notes on baselines

- [**EigenOption**](https://openreview.net/pdf?id=Bk8ZcAxR-) selects the top `n` eigenvectors from a diffusive-type matrix (e.g., graph Laplacian, Successor Representation, Successor Features).
- [**CoveringOption**](https://openreview.net/pdf?id=SkeIyaVtwB) selects the top 1 eigenvector and iteratively updates the diffusive matrix to find a better-explaining matrix, particularly effective in environments with hardly exploratory state-transitions.
---

## Experimental Design
**Fourroom Environment**
- Time steps: 100
- Successor Feature (SF) matrix is built using (100 trajectories x feature_dim)
- Only the goal position is stochastic while others remain constant (agent loc, grid). This is to induce the dynamics in the reward structure of the environment as the simplest case.
- Singular Value Decomposition (SVD) is applied for eigenpurpose discovery
- Intrinsic reward is calculated as the dot product of the eigenvector and the feature difference: `eigenvector^T * (next_feature - current_feature)`
- 
**CtF**
- Time steps: ??? (reasonable amount)
- Successor Feature (SF) matrix is built using (100 trajectories x feature_dim) # I assume still 100 if so no change is required
- The grid layout, agent's starting position, and enemy agents' positions are fixed, while the enemy agents move dynamically, altering the reward structure.

## Usage

To create a conda environment, run the following commands:

```bash
conda create --name myenv python==3.10.*
conda activate myenv
```
Then, install the required packages using pip:
```
pip install -r requirements.txt
```


## How to Import Pre-Trained Model

The training sequence proceeds as follows: SFs (Successor Features) -> OP (Option Training) -> HC (Hierarchical Training). Throughout this process, models are saved periodically in:

```log/train_log/```

Inside this folder, you'll find subdirectories such as `SF`, `OP`, `HC`, etc., where the trained models are stored. To import a trained model for evaluation, move the desired model from the appropriate subfolder to:

```log/eval_log/model_for_eval/```

After moving the model, rename the file (e.g., `model_n.p` or `best_model.p`) to match one of the following, depending on its type:
- sf_model
- op_model
- hc_model

Finally, enable the appropriate model import flags in the argparse file:

```import-sf-model```
```import-op-model```  
```import-hc-model```

Make sure to import the OP model, as the SF model is a prerequisite.


python3 main.py --algo-name SNAC --num-vector 10 
```
where algo-name = {SNAC, EigenOption, CoveringOption, PPO} and num-vector is the total number of eigenpurposes each algorithmn will use.
