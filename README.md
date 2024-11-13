# Log Probabilities Are a Reliable Estimate of Semantic Plausibility in Base and Instruction-Tuned Language Models
Code accompanying the paper "Log Probabilities Are a Reliable Estimate of Semantic Plausibility in Base and Instruction-Tuned Language Models" by Kauf, Chersoni, Lenci, Fedorenko, and Ivanova, to appear at the 7th BlackBoxNLP Workshop (at EMNLP2024).

This code is based on code by [Hu & Levy (2023)](https://github.com/jennhu/metalinguistic-prompting).

# Abstract
Semantic plausibility (e.g. knowing that "the actor won the award" is more likely than "the actor won the battle") serves as an effective proxy for general world knowledge. Language models (LMs) capture vast amounts of world knowledge by learning distributional patterns in text, accessible via log probabilities (LogProbs) they assign to plausible vs. implausible outputs. The new generation of instruction-tuned LMs can now also provide explicit estimates of plausibility via prompting. Here, we evaluate the effectiveness of LogProbs and basic prompting to measure semantic plausibility, both in single-sentence minimal pairs (Experiment 1) and short context-dependent scenarios (Experiment 2). We find that (i) in both base and instruction-tuned LMs, LogProbs offers a more reliable measure of semantic plausibility than direct zero-shot Prompting, which yields inconsistent and often poor results; (ii) instruction-tuning generally does not alter the sensitivity of LogProbs to semantic plausibility (although sometimes decreases it); (iii) across models, context mostly modulates LogProbs in expected ways, as measured by three novel metrics of context-sensitive plausibility and their match to explicit human plausibility judgments. We conclude that, even in the era of prompt-based evaluations, LogProbs constitute a useful metric of semantic plausibility, both in base and instruction-tuned LMs.

# Citation
If you use this work, please cite
```
@inproceedings{kauf2024log,
  title={Log Probabilities Are a Reliable Estimate of Semantic Plausibility in Base and Instruction-Tuned Language Models},
  author={Kauf, Carina and Chersoni, Emmanuele and Lenci, Alessandro and Fedorenko, Evelina and Ivanova, Anna},
  booktitle={Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP},
  pages={263--277},
  year={2024}
}
```
