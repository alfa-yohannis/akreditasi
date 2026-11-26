# Q3 journal checklist

What the paper needs before it goes to a Scopus Q3 journal. As of 2026-11-26.

Start with section 1. The answer key is on the critical path: sections 2, 3 and 5
reuse its results. Sections 4 and 7 can start today, in parallel.

Each task names who does it: **You**, **Team** (the colleagues who know the
documents) or **Claude**. Tick a task once its result is in the repository or the
paper.

Questions, labels and generated answers stay in the git-ignored private files
(`evaluation/queries.private.json`, `evaluation_results/private/`). Only aggregate
scores go into this public repository and the paper.

## 1. Correctness against a labelled answer key

Reviewers will most likely reject on this gap. The paper itself says "No
correctness evaluation yet" (Limitations section).

- [ ] **Claude**: draft about 200 candidate questions from the LAM INFOKOM
  instrument. Cover the current question types: table values, people, lists,
  narrative, curriculum, identifiers, facts, and questions the archive cannot answer.
- [ ] **Team**: review the questions so they reflect what assessors actually ask.
- [ ] **Claude**: for each question, pool the top 20 passages from every retrieval
  stage and pre-mark the likely relevant ones. That makes labelling a review, not a search.
- [ ] **Team**: two people label each pooled passage independently (relevant,
  partly, not relevant), without seeing which stage found it.
- [ ] **Claude**: compute agreement between the two labellers (Cohen's kappa; aim
  for 0.6 or higher). **Team** resolves the disagreements.
- [ ] **Team**: write the key facts of the correct answer for each question.
- [ ] **Claude**: retrieval scores for each stage separately: meaning search only,
  BM25 only, fusion, then fusion plus the reranker. Report Recall@6, Recall@24, MRR and nDCG@10.
- [ ] **Team + Claude**: score the answers for correctness against the key facts,
  for sticking to the passages, and for citation precision and recall.
  - People rate a sample without knowing which model wrote it.
  - Any automatic judge runs locally and is checked against the human ratings.
- [ ] **You**: confirm the Q12 wrong-course-code finding already used in the paper.
- [ ] **Claude**: add the correctness research question(s) and update the
  Methodology, Results and Threats to validity sections.

## 2. Baselines to compare against

- [ ] **Claude**: run an off-the-shelf RAG pipeline (default LangChain or
  LlamaIndex: fixed-size chunks, meaning search only, no reranker) on the same
  documents and questions.
- [ ] **Claude**: run other local answering models of similar size (7–9B) on the
  same prompts, including an Indonesian one (e.g. SEA-LION or Sahabat-AI, if
  available in Ollama).
- [ ] **You**: decide whether to compare against a cloud model. If yes, **Team**
  picks a set of public documents (regulations, the instrument, public web
  pages), so nothing confidential leaves the campus.
- [ ] **Claude**: put every baseline in one results table, with the same metrics
  as section 1.

## 3. More questions and statistics

- [ ] **Claude**: retrieval on the whole labelled set. The answering model on at
  least 50 questions (currently 5); one to three runs per question are enough
  for correctness. Keep 3 warm-up + 10 timed runs for speed only.
- [ ] **Claude**: 95 % bootstrap confidence intervals on every metric.
- [ ] **Claude**: paired tests (Wilcoxon signed-rank for timings, McNemar or a
  paired bootstrap for correctness), with Holm correction for multiple comparisons.
- [ ] **Claude**: report effect sizes beside the p-values.

## 4. Related work and references

The paper has 19 references; 10 of them are preprints, web pages or model cards.

- [ ] **Claude**: search the recent (2021–2026) peer-reviewed literature on:
  - RAG in education and university administration;
  - question answering and RAG in Indonesian;
  - question answering over regulatory, compliance or audit documents;
  - running language models on your own hardware, and quantisation;
  - hybrid retrieval and reranking, beyond the classic papers already cited.
- [ ] **Claude**: write a Related Work section, ending with a table comparing the
  closest systems (domain, language, local or cloud, citation granularity, evaluation).
- [ ] **Claude**: replace each preprint with its published version where one
  exists. Software (FTS5, Ollama) stays as software citations.
- [ ] **Claude**: reach about 35–50 references, mostly from 2021 onwards.
- [ ] **You**: read each new reference before submission.

## 5. Research gap and contributions

- [ ] **Claude**: state the gap in the Introduction in one paragraph. Existing RAG
  work does not cover an audited, confidential, constantly edited archive, in
  Indonesian, on one workstation.
- [ ] **Claude**: replace "not a new algorithm but a deployed design" with an
  evaluated case study.
- [ ] **Claude**: write the contributions as findings with numbers:
  - answers that can be checked by construction, with measured correctness;
  - graphics memory spilling into system memory as a hidden slowdown on Windows
    (6.3 times faster once removed);
  - mechanical grounding checks that missed misattribution (85 % of answers to one question).
- [ ] **You + Claude**: revise the title and abstract once the section 1 results are in.

## 6. Should do: user study and discussion

- [ ] **You**: check whether the university needs ethics approval. Get written
  consent from participants. Consent also covers the nine usage-log questions
  already in the paper.
- [ ] **Claude**: prepare the study tasks, the System Usability Scale (SUS)
  questionnaire in Indonesian, and the analysis script.
- [ ] **Team**: run a timed study with about 6–10 colleagues.
  - Each finds evidence for a set of claims with and without the system, in alternating order.
  - Record the time taken and whether the evidence found is right.
- [ ] **Claude**: add a Discussion section: what generalises to other institutions
  and hardware, and what is specific to this one.
- [ ] Optional, **Claude**: repeat the timing on a second machine or on Linux, to show
  how far the graphics-memory finding generalises.

## 7. Journal choice

- [ ] **Claude**: shortlist 5–8 journals with links. For each, give:
  - the current SJR quartile (Computer Science Applications, Information Systems or Education);
  - its Scopus status (not discontinued);
  - the publication fee (APC);
  - the typical review time;
  - how well it fits the paper's scope.
- [ ] **You**: check what the university counts: the quartile in which year, and
  indexing at the time of publication.
- [ ] **You**: pick the target journal and a fallback, and confirm the budget for the fee.
- [ ] **You**: reach the journal through its link on the Scopus Sources page, never
  from an email invitation, to avoid hijacked journal sites.

## 8. Packaging and submission

- [ ] **Claude**: move the paper to the journal's template, length limit and
  section structure (Introduction, Related Work, Method, Results, Discussion, Conclusion).
- [ ] **Claude**: write the data availability, code availability, ethics,
  conflict of interest, funding and author contributions (CRediT) statements.
- [ ] **Claude**: add `CITATION.cff`. **You**: connect the GitHub repository to
  Zenodo and cut a release, to get a DOI.
- [ ] **Claude**: check that nothing public (repository, released test set)
  contains confidential passages, real questions or staff names.
- [ ] **You**: decide co-authors (labellers, study contributors) and acknowledgements.
- [ ] **You**: professional English edit, or a read by a native speaker.
- [ ] **You**: similarity check with the university's Turnitin or iThenticate,
  below the journal's limit (often about 20 %).
- [ ] **You**: confirm the paper is not under review elsewhere. If a conference
  version exists, disclose it; the journal version must add substantial new material.
- [ ] **Claude**: draft the cover letter and a list of suggested reviewers.
  **You**: check the list.
- [ ] **You**: submit. **Claude** drafts the responses to the reviewers.
