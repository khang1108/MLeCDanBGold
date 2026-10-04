# Introduction wording comparison

Date: 2026-09-21. Scope: editorial assessment of the user's proposed opening paragraph.
The manuscript has not been replaced. This is a purposive, availability-based
sample of ten MMM/VBS system papers from 2020–2025, covering five system families.
It is not a statistical language benchmark or an AI-authorship detector.

## Method and access

Compare the available opening introduction passages by topic specificity, scope,
user actions, and connection to a particular system change. P2, P4, and P6 were
read through indexed extracts of the linked primary PDFs; the other seven had
readable PDF/HTML introduction text. Do not imply that every complete paper was
read or that missing discussion in an opening passage is absent from the paper.
Publication years refer to the MMM/VBS edition, not later repository upload dates.

## Corpus and mapping

R1: replace generic progress framing with a concrete capability.
R2: scope empirical claims and distinguish observation from inference.
R3: name user actions and attribute their frequency to the actual study.
R4: name the specific problem or editable object rather than a general benefit.

Cells describe an editorial parallel or contrast, not proof of the recommendation.
No corpus paper is treated as empirical evidence for the 2026 logging result.

| ID / source | R1: opening | R2: scope | R3: actions | R4: specific focus |
| --- | --- | --- | --- | --- |
| P1 [Exquisitor 2020](https://pure.itu.dk/files/85597082/Exquisitor_VBS_2020.pdf) | VBS task setting | Competition constraints | Judge examples; update classifier | Feedback process |
| P2 [Vibro 2022](https://files.visual-computing.com/research/Efficient_Search_and_Browsing_of_Large_Scale_Video_Collections_with_Vibro.pdf) | Tasks and collection size | Observations and beliefs are distinguishable | Temporal queries; browse images | Search and display changes |
| P3 [VISIONE 2022](https://iris.cnr.it/bitstream/20.500.14243/416951/1/prod_468766-doc_189558.pdf) | Broad data-growth opening: counterexample to R1 | Named system follows broad claims | Search modalities in abstract | Textual encoding and shared index |
| P4 [VISIONE 2023](https://zenodo.org/records/8328883/files/2023_vbs2023_visione.pdf) | VBS tasks and changed conditions | Named models and edition | Select similar frames; inspect fewer frames | Models and result clustering |
| P5 [Exquisitor 2024](https://pure.uva.nl/ws/files/195689365/978-3-031-53302-0_31.pdf) | VBS retrieval paradigms | Expected benefit marked as belief | Refine information; label results | Confusing search/feedback workflow |
| P6 [VISIONE 5.0 / VBS 2024](https://iris.cnr.it/bitstream/20.500.14243/485001/1/2024_VBS2024_VISIONE_V_431.pdf) | Concrete task definitions | Edition and datasets specified | Retrieve a segment or matching shots | Named search inputs in abstract |
| P7 [diveXplore 2024](https://arxiv.org/html/2508.20560v1) | VBS setting and tasks | Expert/novice and task context | Locate clips; browse | Redesign connected to earlier experience |
| P8 [PraK 2024](https://bib.dbvis.de/uploadedFiles/VBS_2024.pdf) | VBS and named OpenCLIP setup | Current system separated from ambition | Reformulate, filter, give feedback | Data access limitation explicitly compared |
| P9 [vitrivr 2024](https://dbis.dmi.unibas.ch/publications/2024/VBS24-vitrivr/paper.pdf) | Existing system architecture | Current state versus future plans | Opening is architecture-focused, not a behavioral study | Aging engine constraints |
| P10 [Exquisitor 2025](https://pure.uva.nl/ws/files/294436154/978-981-96-2074-6_31.pdf) | Broad definition/growth: counterexample to R1 | Section 3 distinguishes observations and conjecture | Query, label, revise | Separation of interaction modes |

Seven opening paragraphs start with VBS context, one with the existing system,
and two with broader background. This is a manual classification of these ten
passages only. Familiar broad openings also appear in published work; publication
is not evidence that every sentence is a good model, nor of human/AI authorship.

## Assessment of the supplied paragraph

- R1: “Recent advances”, “substantially”, and “increasingly expressive” leave the
  model, comparison, metric, and meaning of expressiveness unspecified. State the
  capability directly; P4 and P8 provide concrete model examples. P3/P10 show
  the broad style exists in published work, not that it is necessary.
- R2: “consistently show”, “do not eliminate the need”, and “successful search
  still relies” overgeneralize. Sentences 2–3 repeat the same interaction claim.
  Replace them with one scoped observation from the VBS 2022 post-evaluation.
- R3: Replace “Recent interaction-log analyses” with the identifiable 2026 study
  and its two-system scope. Prefer “users reformulated queries and inspected
  results” to a chain of abstract nouns. Do not infer necessity from frequency.
- R4: One cannot literally “correct why” a search fails. Name what changes:
  query terms, constraints, temporal relations, or another actual editable object.
  Explainability and correction are different capabilities. An interface may expose
  constraints without faithfully explaining the causes of a ranker's errors.

## Evidence outside the ten-paper style corpus

- [Schall et al., 2024](https://doi.org/10.1007/s13735-024-00325-9): full article
  inspected; supports a scoped distinction between initial retrieval performance
  and overall interactive task performance.
- [Jäckl et al., 2026](https://pure.itu.dk/en/publications/what-drove-success-at-the-15th-video-browser-showdown-a-comprehen/):
  author abstract inspected; supports two-system reformulation/inspection claim.
- [vitrivr 2021](https://dbis.dmi.unibas.ch/publications/2021/VBS-2021/): abstract
  inspected; earlier result-explainability work, relevant to novelty positioning.
- [Lokoč et al., 2023](https://doi.org/10.1007/978-3-031-27077-2_50): publisher
  abstract inspected; class suggestions support query reformulation.

## Proposed wording, for discussion

CLIP-based models allow users to search video collections with natural-language
descriptions [P4, P8]. Yet retrieval quality alone does not determine search
success. In a post-evaluation of VBS 2022, the system with the strongest initial
text-to-image retrieval did not achieve the highest overall score; browsing and
image-based queries also contributed to performance [Schall et al., 2024].
An analysis of interaction logs from two systems at VBS 2026 found that users
predominantly reformulated text queries and inspected the returned results
[Jäckl et al., 2026]. We focus on supporting this process by allowing users to
inspect and edit the constraints used to represent a multi-part query.

The last sentence is a proposed statement of SHI's scope, conditional on the
implemented interface. It is not a finding established by the cited papers.
Verify the actual editable objects before using it as a system claim.
