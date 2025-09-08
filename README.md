# ViConBERT: Context-Gloss Aligned Vietnamese Word Embedding for Polysemous and Sense-Aware Representations

This repository is official implementation of the paper: ViConBERT: Context-Gloss Aligned Vietnamese Word Embedding for Polysemous and Sense-Aware Representations

![](figs/architecture.jpg)
<p align="center"><em>Main architecture</em></p>

* **Abstract:**
Recent progress in contextualized word embeddings has significantly advanced tasks involving word semantics, such as Word Sense Disambiguation (WSD) and contextual semantic similarity. However, these developments have largely focused on high-resource languages like English, while low-resource languages such as Vietnamese remain underexplored. This paper introduces a novel training framework for Vietnamese contextualized word embeddings, which integrates contrastive learning (SimCLR) and distillation with the gloss embedding space to better model word meaning. Additionally, we introduce a new dataset specifically designed to evaluate semantic understanding tasks in Vietnamese, which we constructed as part of this work. Experimental results demonstrate that ViConBERT outperforms strong baselines on the WSD task (F1 = 0.87) and achieves competitive results on ViCon (AP = 0.88) and ViSim-400 (Spearman’s $\rho$ = 0.60), effectively modeling both binary and graded semantic relations in Vietnamese.
