# Règles méthodologiques (dictée)

Décisions tranchées avec la DEPP. Ne pas les remettre en cause sans l'accord de
l'utilisateur ; l'historique et les justifications sont dans `docs/decisions.md`.

## Grille de codage

| Code | Mots                             | Code | Ponctuation          |
|------|----------------------------------|------|----------------------|
| 1    | Orthographe correcte             | 1    | Ponctuation correcte |
| 3    | Erreur lexicale                  | 9    | Erreur               |
| 4    | Erreur grammaticale              | 0    | Absence              |
| 5    | Erreur lexicale ET grammaticale  |      |                      |
| 0    | Mot absent (oubli)               |      |                      |

- **Cible principale = grille simplifiée** : 3/4/5 regroupés en un seul code « erreur »
  → on code en **1 / erreur / 0**. La grille à 6 codes reste un objectif secondaire.
  Raison : sur le pilote, le kappa passe de 0,785 à 0,879 (on supprime la zone où les
  correcteurs humains divergent le plus).
- **Ratures** (tranché par la DEPP) : lire l'**état final**, c'est-à-dire la version
  corrigée par l'élève. Cette consigne figure dans tous les prompts.

## Méthodes comparées

- **A** HTR classique (TrOCR/PyLaia) → LLM texte ; **B** VLM transcrit → LLM texte ;
  **C** end-to-end (PRIVILÉGIÉE) : VLM image + texte de référence + grille → code par item ;
  **D** fine-tuning LoRA/QLoRA (phase ultérieure).
- Modèle principal Qwen2.5-VL : tester le **7B avant le 72B**. Baselines : TrOCR, PyLaia,
  PP-OCRv5. Repli si GPU contraint : MiniCPM-V-2.6.
- Raccourci sans GPU : `gemma4-26b-moe` (llm.lab, multimodal), candidat de première
  intention pour la méthode C. Config : `configs/scoring/dictee_REFERENCE.yaml`.
- Mode « thinking » sur llm.lab : le raisonnement est isolé dans `reasoning_content`
  (le JSON reste conforme, rien à élaguer) mais consomme le MÊME budget que la réponse →
  prévoir `max_tokens: 16384`. Mesuré le 11/09/2026 : ON par défaut sur `qwen3-6-35b-moe`
  et `qwen3-8-27b`, OFF sur `gemma4-26b-moe`, absent de `qwen3-vl`. `gemma4-26b-moe` en
  thinking s'emballe sans rendre de JSON : à réserver aux Qwen3. Voir
  `configs/scoring/dictee_end2end_thinking.yaml`.

## Confiance par item (exigence transverse)

Chaque code prédit s'accompagne d'un score de confiance (log-probs vLLM, champ structuré,
auto-cohérence ou désaccord inter-modèles) pour ne renvoyer à un expert humain que les
items incertains. Livrable décisionnel : courbe « taux de renvoi humain vs taux d'erreur
résiduel ».

## Métriques

- **Scoring** : accord brut, **kappa de Cohen**, QWK (codes ordinaux), matrice de
  confusion par code, F1 pondéré.
- **HTR** : CER, WER, taux d'« illisibles », et surtout **taux de sur-correction** (mots
  « corrigés » silencieusement par le modèle — biais critique des VLM).
- **Confiance** : courbe de fiabilité, ECE.
- Toujours comparer l'IA à la **variabilité inter-codeurs humains** (Dawid & Skene, package
  R `rater` ou équivalent Python).

## Copies écartées des métriques

Une copie **vierge** (encre sous `data.blank_ink_threshold`) ou **illisible** (code expert
`i`) est exclue des métriques, conservée dans le JSONL avec un champ `exclusion` et listée
dans `<run>_<modele>_copies_ecartees.csv` : l'expert n'ayant rendu aucun jugement
comparable, les compter imputerait au modèle un défaut d'annotation ou de numérisation.
Une copie n'est écartée en entier que si elle est majoritairement inexploitable ; sinon
seuls les items concernés sortent. **Liste à relire à l'œil** (scan trop pâle, seuil mal
réglé → vierge à tort). Détails : `docs/decisions.md`, D8.
