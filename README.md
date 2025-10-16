# 🧭 FR3E: First Return, Entropy-Eliciting Explore

---

## 🔍 Overview

**FR3E (First Return, Entropy-Eliciting Explore)** is a **structured exploration framework** for reinforcement learning with large language models (LLMs).  
It identifies **high-uncertainty decision points** in reasoning trajectories and performs **targeted rollouts** to generate **semantically grounded intermediate feedback**, improving both **training stability** and **reasoning performance**.

### ✨ Key Highlights

- 🚀 **Structured Exploration:** Targeted rollouts from entropy-sensitive reasoning anchors  
- 🧠 **Adaptive Advantage Modulation:** Stabilizes policy updates and avoids early convergence  
- 🔄 **Rejection Sampling + Clip-Higher:** Improves diversity and prevents degenerate batches  
- 📈 **Stable Entropy Growth:** Prevents entropy collapse and sustains exploration  
- 🧮 **Empirical Gains:** Outperforms GRPO++ across reasoning benchmarks

---

## 🧩 Algorithm Overview

FR3E decomposes LLM reinforcement learning into **two complementary stages**:

🧭 **Stage 1: First Return — Identifying Critical Paths and High-Entropy Nodes**

In the **First Return** stage, FR3E generates complete reasoning trajectories and selectively retains both partially correct and incorrect ones for the next stage.  
We compute **token-level entropy** across the entire sequence and observe that **high-entropy positions** often correspond to **critical decision points** in reasoning — places where the model is most uncertain and prone to error.

Therefore, FR3E introduces the following strategy:

✅ Extract the **Top-K high-entropy tokens** along the reasoning trajectory  
✅ Treat these tokens as **anchors** to segment the trajectory into **semantic blocks**, forming the foundation for localized exploration  
✅ Avoid blind exploration from scratch by **focusing on local improvements around reasoning bottlenecks**

Through *First Return*, the model establishes targeted exploration starting points grounded in intrinsic uncertainty.

---

🌌 **Stage 2: Entropy-Eliciting Explore — Localized Exploration from High-Entropy Anchors**

After identifying high-entropy nodes, FR3E launches a **localized and diversified rollout mechanism** around these critical points:

✅ Perform **multiple local rollouts** from each high-entropy anchor to explore alternative reasoning branches  
✅ **Evaluate the correctness** of each branch to estimate the empirical value of the corresponding node  
✅ Use these node values as **intermediate feedback** for an **adaptive advantage modulation mechanism**, dynamically balancing exploration and exploitation and integrating scalable process-level supervision

This structured design enables FR3E to generate semantically meaningful exploration while maintaining coherence with the original trajectory.


## 🧱 Implementation

FR3E builds upon the [**VeRL**](https://github.com/volcengine/verl) reinforcement learning framework.

### Installation

```bash
git clone https://github.com/FR3E/FR3E.git
cd FR3E
conda create -n fr3e python=3.10
conda activate fr3e

pip install -r requirements.txt
