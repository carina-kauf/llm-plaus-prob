# Overview of datasets

## Single sentence minimal pair datasets (Experiment 1)
We use two sentence sets adapted from previous studies and compare model scores with human plausibility judgements. The data files were obtained from [Kauf, Ivanova et al. (2023)](https://onlinelibrary.wiley.com/doi/epdf/10.1111/cogs.13386).

| Dataset        | Plausible? | Possible? | Voice  | Example                           | Source                |
|----------------|------------|-----------|--------|-----------------------------------|-----------------------|
| **EventsAdapt <br> (AA, unlikely)**   | Yes        | Yes       | Active | The nanny tutored the boy.        | [Fedorenko et al. (2020)](https://www.sciencedirect.com/science/article/abs/pii/S0010027720301670) |
|                | No         | Yes       | Passive| The boy was tutored by the nanny. |                       |
|                | Yes        | Yes       | Active | The boy tutored the nanny.        |                       |
|                | No         | Yes       | Passive| The nanny was tutored by the boy. |                       |
| **EventsAdapt <br> (AI, impossible)**   | Yes        | Yes       | Active | The teacher bought the laptop.    | [Fedorenko et al. (2020)](https://www.sciencedirect.com/science/article/abs/pii/S0010027720301670)|
|               | No        | No        | Passive| The laptop was bought by the teacher.|                     |
|                | Yes        | No        | Active | The laptop bought the teacher.    |                       |
|                | No         | No        | Passive| The teacher was bought by the laptop.|                     |
| **DTFit <br> (AI, unlikely)**           | Yes        | Yes       | Active | The actor won the award.          | [Vassallo et al. (2018)](http://lrec-conf.org/workshops/lrec2018/W9/pdf/5_W9.pdf)|
|                 | No         | Yes       | Active | The actor won the battle.         |                       |

## Context + sentence minimal pair dataset (Experiment 2)
To test the sensitivity of the LLM plausibility judgments to discourse context effects, we use a dataset from language neuroscience, collected by [Jouravlev et al. (2019)](https://evlab.mit.edu/assets/papers/Jouravlev_et_al_2018_PsychSci.pdf).

This dataset includes 100 items in three experimental conditions: a control condition (Control), in which the target sentence describes a plausible situation and the (optional) context sentence adds extra information; a semantically anomalous condition (SemAnom), in which the target sentence describes an implausible situation and the context sentence does not provide licensing information; and a critical condition (Critical), which shares the same target sentence with SemAnom, but here, the context sentence makes it plausible.

| Condition | Context sentence (optional)                         | Prefix                  | Tgt. word | Spill-over region           |
|-----------|-----------------------------------------------------|-------------------------|-----------|-----------------------------|
| Control   | The kids were looking at a canary in the pet store. | The bird had a little   | beak      | and a bright yellow tail.   |
| SemAnom   | Anna was definitely a very cute child.              | The girl had a little   | beak      | and a bright yellow tail.   |
| Critical  | The girl dressed up as a canary for Halloween.      | The girl had a little   | beak      | and a bright yellow tail.   |
