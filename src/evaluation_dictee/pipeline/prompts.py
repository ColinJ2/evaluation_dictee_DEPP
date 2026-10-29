"""Construction des prompts d'évaluation (décisions : `.claude/rules/methodologie.md`).

Point critique : les élisions (n', S', l', d', qu') sont des items SÉPARÉS du mot suivant ;
sans consigne explicite le modèle les recolle et décale tous les items suivants.
Chaque prompt est un template chat versionné dans Langfuse, avec repli local si indisponible.
"""

from __future__ import annotations

import json
from typing import cast

from langfuse import get_client
from langfuse.model import ChatMessageDict, ChatPromptClient

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.grid import GridItem

# Noms sous lesquels les prompts sont versionnés dans Langfuse.
PROMPT_DICTATION = "Dictation"
PROMPT_TRANSCRIPTION = "Transcription"
PROMPT_TEXT_CODING = "Text coding"

_GRILLE_SIMPLIFIEE = (
    "Pour chaque item, attribue un code :\n"
    '- "1" : le mot (ou la ponctuation) attendu est présent et correctement orthographié ;\n'
    '- "9" : le mot est présent mais comporte une erreur (orthographe, accord, '
    "conjugaison, accent, ponctuation erronée...) ;\n"
    "- \"0\" : le mot attendu est absent (l'élève ne l'a pas écrit)."
)

_GRILLE_COMPLETE = (
    "Pour chaque item, attribue un code.\n"
    "Pour un MOT :\n"
    '- "1" : correctement orthographié ;\n'
    '- "3" : erreur LEXICALE (n\'altère pas la prononciation : accent, lettre muette, '
    "mauvais graphème — ex. « soire » pour « soir ») ;\n"
    '- "4" : erreur GRAMMATICALE (accord, conjugaison, confusion de catégorie comme '
    "« on »/« ont », ou toute erreur qui change la prononciation) ;\n"
    '- "5" : erreur À LA FOIS lexicale ET grammaticale ;\n'
    "- \"0\" : le mot est absent (l'élève ne l'a pas écrit).\n"
    'Pour la PONCTUATION : "1" correcte, "9" erronée, "0" absente.'
)

_CONSIGNE_ALIGNEMENT = (
    "2 - RÈGLE D'ALIGNEMENT (la plus importante) : la liste de règles ci-dessous définit des items "
    "FIXES, un par ligne, dans l'ordre du texte. Tu dois rendre EXACTEMENT un code par "
    "item, dans le même ordre, sans en fusionner ni en omettre:\n"
    "a - Aligne-toi sur le MOT ATTENDU de chaque item, JAMAIS sur ta propre découpe de "
    "l'écriture de l'élève. L'élève peut écrire un mot avec un espace au milieu "
    "(« re trouver » au lieu de « retrouver ») ou coller deux mots (« nousles » au "
    "lieu de « nous les ») : ne te laisse pas décaler. Dans ces cas, rattache ce que "
    "tu lis au mot attendu correspondant, et continue d'aligner les items suivants "
    "sur leurs mots attendus respectifs.\n"
    "b - Les apostrophes d'élision sont des items SÉPARÉS du mot qui suit. Par exemple "
    "« n'étaient » se code en DEUX items distincts : « n' » puis « étaient ». "
    "De même « S'ils » = « S' » puis « ils » ; « l'olivier » = « l' » puis « olivier ».\n"
    "c - AVANT de coder, vérifie pour chaque item que la transcription que tu donnes "
    "correspond bien au mot attendu de CE numéro d'item ; si tu remarques un décalage, "
    "recale-toi immédiatement sur le mot attendu. Le nombre d'items de ta réponse doit "
    "être EXACTEMENT celui demandé.\n"
)

_CONSIGNE_FIDELITE = (
    "3 - Transcris EXACTEMENT ce que l'élève a écrit pour cet item, fautes comprises. "
    "Ne corrige jamais silencieusement l'orthographe : une faute non transcrite "
    "fausse l'évaluation. Si le mot écrit diffère du mot attendu, c'est une erreur (9).\n"
)

_CONSIGNE_COMPARAISON = (
    "4 - MÉTHODE DE CODAGE (à appliquer pour chaque item) : compare LETTRE À LETTRE ta "
    "transcription au mot attendu. Le code est 1 SEULEMENT si les deux sont rigoureusement "
    "identiques (mêmes lettres, mêmes accents, même terminaison). La MOINDRE différence, "
    "une lettre, un accent, une terminaison de conjugaison (ex. « mis » au lieu de "
    "« mit ») ou un singulier/pluriel, impose le code 9, même si le mot reste lisible et "
    "plausible. Ne te fie pas au sens : « mis » et « mit » se prononcent pareil mais "
    "l'un est faux. Code d'après la forme écrite exacte, pas d'après ce que l'élève "
    "voulait dire.\n"
)

_CONSIGNE_RATURES_HALLUCINATION = (
    "5 - Quand un passage est raturé/barré : ignore complètement le texte barré et lis "
    "uniquement ce que l'élève a retenu en version finale, DANS L'ORDRE OÙ C'EST ÉCRIT "
    "sur la copie. N'invente pas, ne réordonne pas les mots pour qu'ils collent au texte "
    "attendu : si l'élève a écrit les mots dans un certain ordre, transcris cet ordre réel.\n"
)

_CONSIGNE_RATURES = (
    "6 - Si l'élève a raturé puis réécrit un mot, lis uniquement la version FINALE "
    "(corrigée par l'élève), pas la version barrée.\n"
)

# TROIS branches obligatoires, dont l'absence. Avec seulement « identique » ou
# « décris la différence », un item que l'élève n'a pas écrit n'avait aucune issue :
# le modèle recopiait le mot ATTENDU et le déclarait identique (mesuré le 11/09/2026 :
# 81 items absents codés 1, et le kappa tombait de 0,705 à 0,483). Le champ obligatoire
# transformait l'absence en présence — la sur-correction que le projet cherche à éviter.
_CONSIGNE_COT = (
    "8 - AVANT de choisir le code, écris un champ « comparaison » qui décrit "
    "explicitement le rapport entre ce que TU LIS SUR LA COPIE et le mot attendu. "
    "Trois cas, et trois seulement :\n"
    "   a - tu lis le mot et il correspond lettre à lettre → « identique » (code 1) ;\n"
    "   b - tu lis le mot mais il diffère → décris la différence (code 9). "
    "Exemple : attendu « inquiets » lu « inquiet » → « il manque le 's' final » ;\n"
    "   c - le mot attendu NE FIGURE PAS sur la copie, l'élève ne l'a pas écrit → "
    "« absent », transcription vide, code 0.\n"
    "   Le cas c est le piège principal : tu as le texte de référence sous les yeux, "
    "et il est tentant d'y recopier un mot que l'élève n'a jamais écrit puis de le "
    "déclarer « identique ». N'invente JAMAIS une transcription à partir du texte de "
    "référence. Si tu ne vois rien d'écrit pour cet item sur l'image, c'est « absent ».\n"
)

_CONSIGNE_EXEMPLES = (
    "12 - FAUTES DÉJÀ OBSERVÉES : pour certains items, la liste ci-dessous indique entre "
    "crochets les formes fautives que des correcteurs ont réellement relevées sur cet "
    "item. Sers-t'en comme d'une aide à la LECTURE : ce sont les confusions à guetter "
    "sur ce mot précis, souvent une seule lettre ou un accent.\n"
    "   Deux pièges à éviter absolument :\n"
    "   a - cette liste n'est PAS exhaustive. Une forme qui n'y figure pas reste une "
    "faute si elle diffère du mot attendu. Ne code pas 1 au motif que ce que tu lis "
    "n'est pas dans la liste.\n"
    "   b - ne va JAMAIS vers ces formes par suggestion. Si l'élève a écrit "
    "correctement le mot attendu, code 1, même si une faute connue lui ressemble. "
    "Tu transcris ce que tu VOIS, pas ce qui est probable.\n"
)

# Expérimentation 3 (18/09/2026) : cible le mécanisme distinct de la ponctuation —
# elle se rate par OMISSION plutôt que par faute sur la plupart des modèles (section 7
# des rapports de bras), contrairement aux mots. Aucune règle n'était jusqu'ici propre
# à ce cas ; la méthode de codage générale (règle 4, "compare lettre à lettre") est
# pensée pour des mots, pas pour vérifier qu'un signe existe.
_CONSIGNE_PONCTUATION = (
    "13 - CAS PARTICULIER DE LA PONCTUATION : pour un item ponctuation, commence "
    "TOUJOURS par vérifier qu'un signe est réellement écrit à cet endroit précis de "
    "la copie, AVANT de juger s'il est du bon type. Un signe absent se code « 0 » "
    "(absent), jamais « 9 » (erreur) — « 9 » suppose qu'un signe existe mais que ce "
    "n'est pas le bon (un point à la place d'une virgule, par exemple).\n"
)

# Majuscules et ponctuation au bénéfice du doute (prompt `majponct`, 21/10/2026). La casse :
# sur les prédictions 2021, l'expert code 9 une majuscule manquante (« elle » pour
# « Elle » : 451 fois sur 464) comme une majuscule en trop (« Maman » pour « maman » :
# 1 020 sur 1 045), et aucune consigne n'en parlait. La ponctuation : choix de
# l'utilisateur, une absence (0) ou un mauvais signe (9) n'est compté que s'il est sûr.
_CONSIGNE_MAJUSCULES = (
    "14 - MAJUSCULES : la majuscule fait partie de l'orthographe. Compare aussi la "
    "PREMIÈRE LETTRE en majuscule/minuscule :\n"
    "   a - si le mot attendu commence par une majuscule (début de phrase, nom propre "
    "comme « Martine » ou « Papa »), une minuscule à la place est une erreur (9) : "
    "« elle » pour « Elle » ;\n"
    "   b - une majuscule écrite là où le mot attendu est en minuscule est aussi une "
    "erreur (9) : « Maman » pour « maman » au milieu de la phrase ;\n"
    "   c - l'accent est facultatif sur une majuscule : « À » pour « A » est correct (1). "
    "Une minuscule reste une erreur, accentuée ou non (« a », « à » pour « A ») ;\n"
    "   d - chez un enfant, une majuscule se reconnaît à sa FORME et à sa TAILLE par "
    "rapport aux lettres voisines. Pour les lettres dont les deux formes se ressemblent "
    "(c, o, s, v, w, x, z, parfois p ou m), juge à la taille.\n"
)

_CONSIGNE_PONCTUATION_TOLERANTE = (
    "15 - PONCTUATION, AU BÉNÉFICE DU DOUTE (cette règle prime sur la grille pour les "
    "items ponctuation) : regarde l'endroit précis de la copie où le signe est attendu.\n"
    "   a - Si un signe de ponctuation y figure, ou une marque qui peut en être un "
    "(point, virgule, trait, même petit, mal formé ou peu lisible), code 1 ;\n"
    "   b - code 9 (erreur) UNIQUEMENT si un signe est écrit et que tu es SÛR que ce "
    "n'est pas le signe attendu (un point d'interrogation net à la place d'un point, "
    "par exemple). Si tu hésites sur la nature du signe, code 1 ;\n"
    "   c - code 0 (absent) UNIQUEMENT si tu es SÛR qu'aucun signe n'est écrit à cet "
    "endroit. Si tu hésites sur sa présence, code 1.\n"
    "   Le tiret de dialogue (« - ») s'écrit en début de ligne, souvent après un retour "
    "à la ligne : cherche-le là.\n"
)

_CONSIGNE_DOUTE_MOTS = (
    "16 - MOTS, EN CAS DE DOUTE : pour tout item qui n'est PAS une ponctuation, code 1 "
    "SEULEMENT si tu lis clairement le mot attendu, lettre à lettre, accents et "
    "majuscule compris. Si tu hésites entre correct et erreur (lettre mal formée, accent "
    "incertain, terminaison peu lisible, majuscule ou minuscule incertaine), code 9. Ne "
    "laisse jamais le texte de référence trancher le doute à ta place.\n"
)

# Mêmes règles, formulées pour l'étape 2 du two_stage, qui ne voit que le texte.
_CONSIGNE_MAJUSCULES_TEXTE = (
    "5 - MAJUSCULES : la majuscule fait partie de l'orthographe. Une minuscule là où le "
    "mot attendu a une majuscule (« elle » pour « Elle ») est une erreur (9), une "
    "majuscule en trop (« Maman » pour « maman ») aussi. L'accent est facultatif sur "
    "une majuscule : « À » pour « A » est correct (1).\n"
)

_CONSIGNE_PONCTUATION_TOLERANTE_TEXTE = (
    "6 - PONCTUATION (cette règle prime sur la grille pour les items ponctuation) : si "
    "le signe attendu figure dans la transcription à cet endroit, code 1 ; si un AUTRE "
    "signe y figure à sa place, code 9 ; s'il n'y a aucun signe à cet endroit, code 0 "
    "(absent).\n"
)

# Étape 1 du two_stage : la transcription doit conserver la casse et ne perdre aucun
# signe, sinon l'étape 2 n'a aucun moyen d'appliquer les deux règles ci-dessus.
_CONSIGNE_DOUTE_MOTS_TEXTE = (
    "7 - MOTS, EN CAS DE DOUTE : pour tout item qui n'est PAS une ponctuation, code 1 "
    "seulement si la transcription donne exactement le mot attendu. Si elle est "
    "incertaine ou incomplète à cet endroit (lettre manquante, caractère illisible), "
    "code 9.\n"
)

_CONSIGNE_DOUTE_MOTS_TRANSCRIPTION = (
    "7 - MOTS, EN CAS DE DOUTE : quand une lettre ou un accent d'un mot est incertain, ne "
    "choisis pas l'orthographe correcte par défaut. Transcris ce que tu vois ; si tu ne "
    "peux pas trancher entre deux lectures, transcris celle qui s'écarte de "
    "l'orthographe correcte.\n"
)

_CONSIGNE_MAJUSCULES_TRANSCRIPTION = (
    "5 - Respecte EXACTEMENT les majuscules et les minuscules de l'élève : n'ajoute pas "
    "de majuscule en début de phrase ou à un nom propre si l'élève ne l'a pas écrite, "
    "n'en retire pas s'il en a écrit une. Ne mets un accent sur une majuscule que s'il "
    "est écrit.\n"
)

_CONSIGNE_PONCTUATION_TRANSCRIPTION = (
    "6 - PONCTUATION : transcris TOUT signe de ponctuation visible, même petit, mal formé "
    "ou douteux : une marque qui peut être un point, une virgule ou un tiret se "
    "transcrit. N'omets un signe que si tu es SÛR qu'il n'y a rien. Le tiret de "
    "dialogue (« - ») en début de ligne est un signe : transcris-le.\n"
)


# Petits mots et lettres proches (prompt `petitsmots`, 23/10/2026). Sur les fautes que
# l'end-to-end valide, il lit le mot ATTENDU (« S' », « a », « ils » à plus de 80 %) là
# où le two_stage, qui transcrit sans la référence, lit « Si », « as », « il » : la
# référence souffle la réponse quand l'image apporte peu d'indices (2-3 lettres, un
# accent de quelques pixels). D'où l'ordre imposé : lire d'abord, comparer ensuite.
_CONSIGNE_PETITS_MOTS = (
    "17 - PETITS MOTS ET LETTRES QUI SE RESSEMBLENT : c'est là que tu te trompes le plus, "
    "car l'image y donne peu d'indices et le mot attendu te souffle la réponse. Pour ces "
    "items, regarde l'image AVANT le mot attendu et transcris ce que tu vois, même si "
    "cela contredit la référence.\n"
    "   a - « à » (préposition) : code 1 SEULEMENT si un accent est visible au-dessus du "
    "« a », incliné du haut à gauche vers le bas à droite (accent grave). « a » sans "
    "accent, accent dans l'autre sens (« á ») ou posé sur une autre lettre : code 9. "
    "À l'inverse, « à » écrit pour le verbe « a » : code 9 ;\n"
    "   b - « a » (verbe avoir) : « as », « sa », « à » sont des erreurs (9) ;\n"
    "   c - « se », « ce », « le », « la », « ne », « et » : vérifie CHAQUE lettre ; « ce » "
    "pour « se », « ses » pour « se », « les » pour « le » : code 9 ;\n"
    "   d - lettres proches dans l'écriture d'un enfant, surtout « s » et « r » (aussi "
    "« n »/« u », « e »/« l », « a »/« o ») : juge à la FORME de la lettre (le « s » se "
    "referme par une boucle ou un crochet vers le bas, le « r » a une épaule ouverte vers "
    "la droite), jamais d'après ce qu'exige le mot attendu. Vérifie que le « s » final "
    "est écrit (« il » pour « ils » : code 9) ;\n"
    "   e - élision (« S' », « n' », « l' ») : un même trait ne compte qu'UNE fois, soit "
    "comme l'apostrophe, soit comme une lettre du mot voisin, jamais pour les deux. "
    "« Si ils », « Sils », « s'ils » (minuscule) : l'item « S' » vaut 9 ;\n"
    "   f - DOUBLE ERREUR en fin de phrase : un point absent se code 0 sur l'item point, "
    "ET le mot suivant, s'il commence par une minuscule alors qu'une majuscule est "
    "attendue, se code 9. Les deux fautes se comptent séparément : aucune n'excuse "
    "l'autre.\n"
)

# Étape 1 du two_stage : la lecture doit garder ce que l'étape 2 doit juger.
_CONSIGNE_PETITS_MOTS_TRANSCRIPTION = (
    "8 - PETITS MOTS ET LETTRES PROCHES : transcris les accents tels qu'ils sont écrits "
    "(« a » sans accent reste « a », un accent dans l'autre sens s'écrit « á »). Pour les "
    "lettres qui se ressemblent (« s »/« r », « n »/« u », « e »/« l »), transcris la "
    "lettre que tu VOIS d'après sa forme, pas celle que le sens suggère, et ne rajoute "
    "jamais un « s » final absent. Un même trait ne se transcrit qu'une fois : soit "
    "apostrophe, soit lettre (« Si ils » et « S'ils » sont deux lectures distinctes).\n"
)

# Étape 2 du two_stage : mêmes règles, formulées sur le texte.
_CONSIGNE_PETITS_MOTS_TEXTE = (
    "8 - PETITS MOTS : « a » ou « á » pour « à », « à », « as » ou « sa » pour « a », "
    "« ce » ou « ses » pour « se », « il » pour « ils », « Si » ou « s' » pour « S' » "
    "sont des erreurs (9). DOUBLE ERREUR : un point absent se code 0 ET le mot suivant, "
    "s'il est en minuscule alors qu'une majuscule est attendue, se code 9.\n"
)


# Signal de doute (prompt `doute`, 23/10/2026). La confiance auto-déclarée ne
# discriminait rien (AUC 0,53 en 2015, 1,0 sur 91 % des items) : on demande à la place
# une lecture concurrente, concrète et vérifiable. Tout l'enjeu est de NE PAS la
# surremplir (a) : un doute de précaution sur chaque item ne signalerait plus rien.
_CONSIGNE_DOUTE_LECTURE = (
    "18 - HÉSITATION RÉELLE ENTRE DEUX LECTURES : pour chaque item, remplis "
    "« lecture_alternative » AVANT le code.\n"
    '   a - Laisse-le VIDE ("") dans la très grande majorité des cas : dès qu\'une lecture '
    "s'impose en regardant les traits, il n'y a pas d'hésitation. N'en déclare pas par "
    "prudence, ni parce que ce que tu lis diffère du mot attendu, ni parce que "
    "l'écriture est maladroite mais lisible ;\n"
    "   b - remplis-le SEULEMENT si, en regardant les traits eux-mêmes, deux lectures "
    "restent vraiment possibles et que tu ne peux pas trancher, par exemple une lettre "
    "dont la forme est à mi-chemin entre « s » et « r », ou entre « n » et « m ». Écris "
    "alors l'autre lecture du mot entier (transcription « porte », lecture_alternative "
    "« poste ») ;\n"
    "   c - n'écris jamais le mot attendu comme alternative s'il n'est pas réellement "
    "lisible ainsi sur la copie ;\n"
    "   d - code : si l'hésitation porte sur la FORME d'une lettre que l'écriture d'un "
    "enfant rend ambiguë (« s »/« r », « n »/« m », « n »/« u », « e »/« l », "
    "« a »/« o ») et que l'une des deux lectures est le mot attendu, code 1 : le doute "
    "profite à l'élève. Cette règle prime sur toute règle de doute précédente. Pour "
    "toute autre hésitation, applique les règles précédentes.\n"
)


def _ajouter_au_systeme(messages: list[ChatMessageDict], blocs: list[str]) -> list[ChatMessageDict]:
    """Ajoute des consignes à la fin du message system.

    Les templates « Transcription » et « Text coding » versionnés dans Langfuse n'ont
    pas de variable pour des consignes optionnelles : les ajouter après compilation
    les applique quelle que soit la version servie, sans modifier Langfuse.
    """
    if not blocs:
        return messages
    out = [dict(m) for m in messages]
    for message in out:
        if message["role"] == "system":
            message["content"] = str(message["content"]).rstrip() + "\n\n" + "\n".join(blocs)
            break
    return cast("list[ChatMessageDict]", out)


_CONSIGNE_COMPTAGE = (
    "9 - COMMENCE PAR COMPTER. Avant de coder quoi que ce soit, parcours l'image et "
    "compte combien d'items l'élève a RÉELLEMENT écrits : chaque mot écrit compte pour "
    "1, chaque signe de ponctuation écrit compte pour 1. Écris ce total dans le champ "
    "« n_items_lus », qui est le PREMIER champ de ta réponse.\n"
    "   Compare-le ensuite au nombre d'items attendus, qui t'est donné plus bas. S'il "
    "est plus petit, c'est que l'élève a écrit moins que la dictée complète : il DOIT "
    "donc y avoir au moins autant d'items codés « 0 » (absent) que la différence entre "
    "les deux nombres. Un élève qui s'arrête au milieu de la dictée laisse tous les "
    "items suivants absents.\n"
    "   Tu as le texte de référence sous les yeux : ne t'en sers JAMAIS pour compléter "
    "ce que l'élève n'a pas écrit. Le comptage est là pour t'en empêcher.\n"
)

_CONSIGNE_COHERENCE_COMPTAGE = (
    "10 - FAIS COÏNCIDER TON CODAGE AVEC TON COMPTAGE. Le nombre d'items que tu codes "
    "« 0 » (absent) doit être EXACTEMENT égal au nombre d'items attendus moins "
    "« n_items_lus ». Si tu annonces avoir lu 60 items sur 83 attendus, tu dois coder "
    "exactement 23 items « 0 ». Avant de rendre ta réponse, recompte tes codes « 0 » : "
    "s'ils ne concordent pas avec ton comptage, l'un des deux est faux, corrige-le. Un "
    "comptage annoncé puis contredit par le codage ne sert à rien.\n"
)

_CONSIGNE_VOISINAGE = (
    "11 - VÉRIFICATION DE VOISINAGE, pour CHAQUE item N (pas seulement en cas de doute) : "
    "une fois ton code choisi, vérifie que ta transcription de l'item N ressemble bien au "
    "mot attendu de l'item N — puis que c'est AUSSI le cas pour l'item N-1 et pour l'item "
    "N+1. Si l'item N correspond mais que N-1 et N+1 sont décalés d'un cran, c'est que tu "
    "as sauté ou dupliqué un item : recale-toi immédiatement.\n"
    "   Un mot que l'élève n'a pas écrit décale TOUT ce qui suit si tu ne le codes pas "
    "« 0 » : au lieu d'un seul item faux, tu en produis vingt. C'est la première cause "
    "d'erreur en chaîne, et la vérification de voisinage est ce qui la rattrape.\n"
)

# Format de sortie JSON de la méthode C : UN SEUL exemple, construit dynamiquement
# par `_format_sortie` selon les options actives. Avant, chaque combinaison avait son
# texte tapé à la main : count_items concaténait un rappel devant un bloc déjà complet,
# produisant deux « Réponds UNIQUEMENT... » successifs avec deux exemples divergents
# (le second oubliait `n_items_lus`). Un seul dict Python, sérialisé une fois, ne peut
# plus diverger de lui-même — et il ne contient plus de champ absent du schéma
# (l'ancien `reason`, jamais dans `_items_json_schema`, ni de score de confiance
# auto-déclaré, jugé peu fiable et retiré : voir docs/decisions.md).

# Consigne « ratures » propre à l'étape de transcription (formulation dédiée).
_CONSIGNE_RATURES_TRANSCRIPTION = (
    "Si l'élève a raturé puis réécrit, transcris uniquement la version "
    "FINALE (non barrée). Ignore complètement le texte barré.\n"
)

# Consigne « ratures » propre à l'étape d'évaluation directe
_CONSIGNE_RATURES_DICTATION = (
    "Si l'élève a raturé puis réécrit, ignore complètement le texte barré dans ton output."
    "Il s'agit d'une correction faite par l'élève, les mots raturés ne doivent pas être pris "
    "en compte dans la correction."
)


# ─────────────────────────────────────────────────────────────────────────────
# Templates chat (masques versionnés dans Langfuse, {{...}} remplis par les build_*).
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE_DICTATION: list[ChatMessageDict] = [
    {
        "role": "system",
        "content": (
            "Tu fais partie d'un groupe d'expert composé de professeurs et de formateurs pour une "
            "évaluation nationale de dictée.\n"
            "Ta tâche : On te montre l'image manuscrite de la dictée d'un élève de primaire. "
            "Tu dois noter chaque item de la dictée à l'aide des éléments définis "
            "dans la grille de notation.\n\n"
            "Règles à respecter impérativement :\n"
            "1 - Tu dois noter chaque item de la dictée avec la grille de notation disponible : "
            "{{grille}}\n"
            + _CONSIGNE_ALIGNEMENT
            + "{{consignes_optionnelles}}\n\n"
            + "{{consigne_cot}}\n\n"
            + _CONSIGNE_RATURES_DICTATION
        ),
    },
    {
        "role": "user",
        "content": (
            "{{bloc_reference}}"
            "# Items à coder, dans l'ordre. Chaque ligne = un item fixe "
            "« identifiant → mot attendu » :\n"
            "{{items_list}}\n\n"
            "# Tu dois rendre EXACTEMENT {{n_items}} items, dans cet ordre.\n"
            "{{format_sortie}}"
        ),
    },
]

# Bloc "texte de référence" par défaut : la phrase continue, entre guillemets.
_BLOC_REFERENCE_PHRASE = (
    "# Texte de référence (ce que l'élève devait écrire) :\n« {reference_text} »\n\n"
)

# Expérimentation 1 (18/09/2026) : PAS de phrase continue. Vise la tension identifiée
# entre donner le texte de référence et le biais dominant mesuré (sous-détection) :
# une phrase familière invite à la reconnaître par lecture fluide plutôt qu'à examiner
# l'image lettre à lettre. Le texte de référence reste entièrement présent — mais
# UNIQUEMENT dans la liste d'items juste en dessous, un mot isolé par ligne.
_BLOC_REFERENCE_ITEMS_SEULS = (
    "# PAS de texte de référence en phrase continue : volontaire. Base-toi "
    "UNIQUEMENT sur le mot attendu de chaque item ci-dessous. Ne reconstitue pas "
    "mentalement la phrase complète pour deviner un mot par le sens ou l'habitude — "
    "juge chaque item sur ce que tu VOIS écrit à cet endroit précis de l'image.\n\n"
)

_TEMPLATE_TRANSCRIPTION: list[ChatMessageDict] = [
    {
        "role": "system",
        "content": (
            "Tu es un expert en lecture d'écriture manuscrite d'enfants.\n"
            "On te montre l'image manuscrite de la dictée d'un élève de primaire.\n\n"
            "Règles à respecter impérativement :\n"
            "1 - Transcris EXACTEMENT le texte écrit par l'élève, mot pour mot, "
            "FAUTES D'ORTHOGRAPHE COMPRISES.\n "
            "2 - Ne corrige rien, ne complète rien, "
            "ne réordonne rien.\n "
            " 3 - Reproduis fidèlement les erreurs, y compris les "
            "accents manquants, les mots mal orthographiés et la ponctuation.\n"
            "4 - Respecte l'ordre exact des items écrits sur la copie "
            "(mots, ponctuation, chiffres, ...).\n"
            "{{consigne_ratures}}"
        ),
    },
    {
        "role": "user",
        "content": (
            "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :\n"
            '{"transcription": "le texte exact écrit par l\'élève"}'
        ),
    },
]

_TEMPLATE_TEXT_CODING: list[ChatMessageDict] = [
    {
        "role": "system",
        "content": (
            "Tu fais partie d'un groupe d'expert, composé de professeur et de formateurs "
            "pour une évaluation nationale de dictée.\n"
            "Ta tâche : Tu ne vois PAS l'image, on te donne uniquement la transcription "
            "de ce que l'élève a écrit (produite par un système de lecture), et le "
            "texte de référence.\n"
            "Compare la transcription au mot attendu de chaque item. Si un mot attendu "
            "n'apparaît pas dans la transcription, code-le 0 (absent).\n\n"
            "Règles à respecter impérativement :\n"
            "1 - Tu dois noter chaque item de la dictée avec la grille de notation "
            "disponible ci-dessous : "
            "{{grille}}\n\n" + _CONSIGNE_ALIGNEMENT + "\n\n" + _CONSIGNE_COMPARAISON + "\n\n"
        ),
    },
    {
        "role": "user",
        "content": (
            "Texte de référence (ce que l'élève devait écrire) :\n"
            "« {{reference_text}} »\n\n"
            "Transcription de la copie de l'élève (fautes comprises) :\n"
            "« {{transcription}} »\n\n"
            "Items à coder, dans l'ordre. Chaque ligne = un item "
            "« identifiant → mot attendu » :\n"
            "{{items_list}}\n\n"
            "Tu dois rendre EXACTEMENT {{n_items}} items, dans cet ordre.\n"
            "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :\n"
            '{"items": [{"item_id": "...", "transcription": "mot lu pour cet item", '
            '"code": "1"}, ...]}'
        ),
    },
]

# Registre exposé pour l'initialisation Langfuse (utils/add_langfuse_prompt.py).
PROMPT_TEMPLATES: dict[str, list[ChatMessageDict]] = {
    PROMPT_DICTATION: _TEMPLATE_DICTATION,
    PROMPT_TRANSCRIPTION: _TEMPLATE_TRANSCRIPTION,
    PROMPT_TEXT_CODING: _TEMPLATE_TEXT_CODING,
}


def _format_sortie(chain_of_thought: bool, count_items: bool, report_doubt: bool = False) -> str:
    """Bloc « format de sortie » du prompt : UN SEUL exemple JSON, cohérent.

    Construit un unique dict Python puis le sérialise une fois, plutôt que de coller
    des morceaux de texte pré-écrits par option : sans quoi deux options combinées
    produisaient deux blocs « Réponds UNIQUEMENT... » successifs, avec deux exemples
    divergents (le second oubliait le champ ajouté par le premier). Un seul objet ne
    peut plus se contredire lui-même, et son ordre de clés reflète exactement
    `_items_json_schema` (dict Python = ordre d'insertion = ordre JSON).

    Args:
        chain_of_thought: ajoute le champ « comparaison » par item, avant « code ».
        count_items: ajoute le champ « n_items_lus » en tête de réponse.
        report_doubt: ajoute le champ « lecture_alternative » par item, avant « code ».

    Returns:
        Le texte décrivant le JSON attendu.
    """
    item: dict[str, str] = {"item_id": "...", "transcription": "ce que l'élève a écrit"}
    if report_doubt:
        item["lecture_alternative"] = ""
    if chain_of_thought:
        item["comparaison"] = 'identique OU description brève de la différence OU "absent"'
    item["code"] = "1"

    racine: dict[str, object] = {}
    if count_items:
        racine["n_items_lus"] = "<nombre entier d'items que tu as VUS sur la copie>"
    racine["items"] = [item]

    consignes = [
        "Réponds UNIQUEMENT par un objet JSON, sans texte autour ni de notes, de la forme :",
        json.dumps(racine, ensure_ascii=False) + ".",
    ]
    if count_items:
        # Rappelé en clair : l'ordre dans l'exemple JSON seul ne suffit pas toujours à
        # faire respecter l'ordre de génération réel (mesuré).
        consignes.append("« n_items_lus » vient EN PREMIER, avant la liste des items.")
    if report_doubt:
        consignes.append(
            '« lecture_alternative » vient AVANT « code » et reste "" sauf hésitation '
            "réelle entre deux lectures (règle 18)."
        )
    if chain_of_thought:
        consignes.append(
            "RESPECTE CET ORDRE DE CLÉS : « comparaison » vient AVANT « code ». Tu dois "
            "avoir écrit la différence avant de choisir le code, sinon tu ne fais que "
            "justifier une décision déjà prise."
        )
    consignes.append(
        "Tout ajout de texte hors de la structure du JSON sera pris comme une erreur "
        "par le pipeline."
    )
    return "\n".join(consignes)


def _fautes_connues(item: GridItem, scheme: str, contraste: bool = False) -> str:
    """Fautes déjà observées sur cet item par les correcteurs, prêtes pour le prompt.

    En grille simplifiée, les trois familles (lexicale, grammaticale, mixte) sont
    fusionnées : elles y reçoivent toutes le même code « erreur ». En grille complète
    elles restent distinctes, puisque c'est précisément ce que le modèle doit trancher.

    Args:
        item: item de la grille.
        scheme: schéma de codage cible.
        contraste: si True (expérimentation 4), rappelle le mot ATTENDU juste à côté
            des fautes connues, plutôt que les fautes seules — pour contrebalancer le
            risque de suggestion (le modèle ancré sur la seule liste de fautes).

    Returns:
        Le fragment de ligne à accoler à l'item, vide si aucune faute n'est connue.
    """
    if scheme == "complete":
        groupes = [
            ("lexicales", item.ex_lexicale),
            ("grammaticales", item.ex_grammaticale),
            ("lexicales ET grammaticales", item.ex_les_deux),
        ]
        parts = [
            f"{libelle} : " + ", ".join(f"« {f} »" for f in formes)
            for libelle, formes in groupes
            if formes
        ]
        if not parts:
            return ""
        if contraste:
            return (
                f"  [attendu : « {item.attendu} » — confusions fréquentes — "
                + " ; ".join(parts)
                + "]"
            )
        return "  [fautes déjà observées — " + " ; ".join(parts) + "]"

    # Dédoublonnage en conservant l'ordre : une même forme peut figurer dans deux
    # familles, et la répéter au modèle n'apporte rien.
    formes = list(dict.fromkeys([*item.ex_lexicale, *item.ex_grammaticale, *item.ex_les_deux]))
    if not formes:
        return ""
    liste_formes = ", ".join(f"« {f} »" for f in formes)
    if contraste:
        return f"  [attendu : « {item.attendu} » — confusions fréquentes : {liste_formes}]"
    return f"  [fautes déjà observées : {liste_formes}]"


def _format_items(
    items: list[GridItem],
    show_error_examples: bool = False,
    scheme: str = "simplifiee",
    contrastive_examples: bool = False,
) -> str:
    """Formate la liste des items « N. identifiant -> « mot » (nature) », un par ligne.

    Args:
        items: items de la grille à formater.
        show_error_examples: si True, accole à chaque item les fautes déjà observées.
        scheme: schéma de codage cible (sert au regroupement des fautes connues).
        contrastive_examples: si True (expérimentation 4), présente le mot attendu en
            contraste des fautes connues au lieu des fautes seules.

    Returns:
        Le texte des items numérotés, une ligne par item.
    """
    lignes = []
    for idx, it in enumerate(items, 1):
        nature = "ponctuation" if it.type == "ponctuation" else "mot"
        suffixe = _fautes_connues(it, scheme, contrastive_examples) if show_error_examples else ""
        lignes.append(f"  {idx:>2}. {it.item_id} → « {it.attendu} » ({nature}){suffixe}")
    return "\n".join(lignes)


def _render_local(
    messages: list[ChatMessageDict], variables: dict[str, object]
) -> list[ChatMessageDict]:
    """Compile un template localement (repli si Langfuse est indisponible).

    Args:
        messages: messages du template, avec placeholders `{{clé}}`.
        variables: valeurs à substituer aux placeholders.

    Returns:
        Les messages avec les placeholders remplacés par leurs valeurs.
    """
    rendered: list[ChatMessageDict] = []
    for message in messages:
        content = message["content"]
        for key, value in variables.items():
            content = content.replace("{{" + key + "}}", str(value))
        rendered.append({"role": message["role"], "content": content})
    return rendered


def _compile_prompt(
    name: str,
    fallback: list[ChatMessageDict],
    variables: dict[str, object],
) -> list[ChatMessageDict]:
    """Récupère et compile un prompt chat Langfuse (repli sur `fallback` local si indisponible).

    Args:
        name: nom du prompt versionné dans Langfuse.
        fallback: template local utilisé si Langfuse est indisponible.
        variables: valeurs à injecter dans le template.

    Returns:
        Les messages compilés (role/content uniquement, placeholders vides écartés).
    """
    try:
        prompt = get_client().get_prompt(name, type="chat", fallback=fallback)
        messages = prompt.compile(**variables)
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        messages = _render_local(fallback, variables)
    # compile() peut renvoyer des placeholders sans "content" ; on ne garde que role/content.
    plain = cast("list[dict[str, object]]", messages)
    compiled: list[ChatMessageDict] = []
    for message in plain:
        if message.get("content"):
            compiled.append({"role": str(message["role"]), "content": str(message["content"])})
    return compiled


def attach_image(messages: list[ChatMessageDict], image_data_url: str) -> list[dict[str, object]]:
    """Attache une image au dernier message user, en contenu structuré (format multimodal vLLM).

    Args:
        messages: messages du prompt ; le dernier doit être le message user.
        image_data_url: image encodée en data URL à joindre.

    Returns:
        Les messages avec l'image ajoutée au contenu du dernier message user.
    """
    *head, user = messages
    out: list[dict[str, object]] = [dict(message) for message in head]
    out.append(
        {
            "role": user["role"],
            "content": [
                {"type": "text", "text": user["content"]},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    )
    return out


def fetch_prompt(name: str) -> ChatPromptClient | None:
    """Renvoie l'objet prompt Langfuse pour lier une trace à sa version (None si repli local).

    Args:
        name: nom du prompt versionné dans Langfuse.

    Returns:
        L'objet prompt Langfuse, ou None si Langfuse est indisponible ou renvoie le repli.
    """
    try:
        prompt = get_client().get_prompt(name, type="chat", fallback=PROMPT_TEMPLATES[name])
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        return None
    return None if prompt.is_fallback else prompt


def build_dictation_prompt(
    reference_text: str,
    items: list[GridItem],
    config: PromptConfig,
    scheme: str = "simplifiee",
) -> list[ChatMessageDict]:
    """Construit le prompt d'évaluation d'une dictée (méthode C) ; image jointe par l'appelant.

    Args:
        reference_text: texte de référence de la dictée.
        items: items de la grille à coder.
        config: options de prompt (fidélité, ratures, chain-of-thought, comptage).
        scheme: schéma de grille, "complete" ou "simplifiee" (défaut).

    Returns:
        Les messages du prompt compilés, prêts à recevoir l'image.
    """
    grille = _GRILLE_COMPLETE if scheme == "complete" else _GRILLE_SIMPLIFIEE

    blocs: list[str] = []
    if config.enforce_faithful:
        blocs += [_CONSIGNE_FIDELITE, _CONSIGNE_COMPARAISON]
    if config.read_final_state:
        blocs += [_CONSIGNE_RATURES, _CONSIGNE_RATURES_HALLUCINATION]
    if config.count_items:
        blocs.append(_CONSIGNE_COMPTAGE)
    if config.enforce_count and config.count_items:
        blocs.append(_CONSIGNE_COHERENCE_COMPTAGE)
    if config.check_neighbours:
        blocs.append(_CONSIGNE_VOISINAGE)
    if config.show_error_examples:
        blocs.append(_CONSIGNE_EXEMPLES)
    if config.check_punctuation_presence and not config.lenient_punctuation:
        blocs.append(_CONSIGNE_PONCTUATION)
    if config.case_rules:
        blocs.append(_CONSIGNE_MAJUSCULES)
    if config.lenient_punctuation:
        blocs.append(_CONSIGNE_PONCTUATION_TOLERANTE)
    if config.doubt_word_is_error:
        blocs.append(_CONSIGNE_DOUTE_MOTS)
    if config.small_words_rules:
        blocs.append(_CONSIGNE_PETITS_MOTS)
    if config.report_doubt:
        blocs.append(_CONSIGNE_DOUTE_LECTURE)
    consignes_optionnelles = ("\n\n" + "\n\n".join(blocs)) if blocs else ""

    consigne_cot = ("\n\n" + _CONSIGNE_COT) if config.chain_of_thought else ""
    format_sortie = _format_sortie(config.chain_of_thought, config.count_items, config.report_doubt)
    bloc_reference = (
        _BLOC_REFERENCE_ITEMS_SEULS
        if config.reference_items_only
        else _BLOC_REFERENCE_PHRASE.format(reference_text=reference_text)
    )

    return _compile_prompt(
        PROMPT_DICTATION,
        fallback=_TEMPLATE_DICTATION,
        variables={
            "grille": grille,
            "consignes_optionnelles": consignes_optionnelles,
            "consigne_cot": consigne_cot,
            "bloc_reference": bloc_reference,
            "items_list": _format_items(
                items, config.show_error_examples, scheme, config.contrastive_examples
            ),
            "n_items": len(items),
            "format_sortie": format_sortie,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompts pour l'APPROCHE 1 (deux étapes) : transcription HTR puis codage textuel
# ─────────────────────────────────────────────────────────────────────────────


def build_transcription_prompt(
    read_final_state: bool = True,
    case_rules: bool = False,
    lenient_punctuation: bool = False,
    doubt_word_is_error: bool = False,
    small_words_rules: bool = False,
) -> list[ChatMessageDict]:
    """Prompt ÉTAPE 1 (HTR) : transcrire l'image sans coder ni voir la référence (non biaisé).

    Args:
        read_final_state: si True, ajoute la consigne de ne lire que l'état final (ratures).
        case_rules: si True, impose de conserver la casse exacte de l'élève.
        lenient_punctuation: si True, impose de transcrire tout signe douteux.
        doubt_word_is_error: si True, un doute sur un mot se transcrit vers la faute.
        small_words_rules: si True, transcrit accents, lettres proches et apostrophes
            tels qu'écrits.

    Returns:
        Les messages du prompt de transcription compilés.
    """
    consigne_ratures = ("\n" + _CONSIGNE_RATURES_TRANSCRIPTION) if read_final_state else ""
    messages = _compile_prompt(
        PROMPT_TRANSCRIPTION,
        fallback=_TEMPLATE_TRANSCRIPTION,
        variables={"consigne_ratures": consigne_ratures},
    )
    blocs = []
    if case_rules:
        blocs.append(_CONSIGNE_MAJUSCULES_TRANSCRIPTION)
    if lenient_punctuation:
        blocs.append(_CONSIGNE_PONCTUATION_TRANSCRIPTION)
    if doubt_word_is_error:
        blocs.append(_CONSIGNE_DOUTE_MOTS_TRANSCRIPTION)
    if small_words_rules:
        blocs.append(_CONSIGNE_PETITS_MOTS_TRANSCRIPTION)
    return _ajouter_au_systeme(messages, blocs)


def build_text_coding_prompt(
    reference_text: str,
    transcription: str,
    items: list[GridItem],
    scheme: str = "simplifiee",
    case_rules: bool = False,
    lenient_punctuation: bool = False,
    doubt_word_is_error: bool = False,
    small_words_rules: bool = False,
) -> list[ChatMessageDict]:
    """Prompt ÉTAPE 2 : coder à partir du texte transcrit (sans image, modèle texte seul).

    Args:
        reference_text: texte de référence de la dictée.
        transcription: transcription de la copie produite à l'étape 1.
        items: items de la grille à coder.
        scheme: schéma de grille, "complete" ou "simplifiee" (défaut).
        case_rules: si True, ajoute la règle des majuscules.
        lenient_punctuation: si True, ajoute la règle de ponctuation tolérante.
        doubt_word_is_error: si True, une transcription incertaine d'un mot vaut 9.
        small_words_rules: si True, ajoute les erreurs types des petits mots et la
            double erreur point absent + minuscule.

    Returns:
        Les messages du prompt de codage textuel compilés.
    """
    grille = _GRILLE_COMPLETE if scheme == "complete" else _GRILLE_SIMPLIFIEE
    blocs = []
    if case_rules:
        blocs.append(_CONSIGNE_MAJUSCULES_TEXTE)
    if lenient_punctuation:
        blocs.append(_CONSIGNE_PONCTUATION_TOLERANTE_TEXTE)
    if doubt_word_is_error:
        blocs.append(_CONSIGNE_DOUTE_MOTS_TEXTE)
    if small_words_rules:
        blocs.append(_CONSIGNE_PETITS_MOTS_TEXTE)
    messages = _compile_prompt(
        PROMPT_TEXT_CODING,
        fallback=_TEMPLATE_TEXT_CODING,
        variables={
            "grille": grille,
            "reference_text": reference_text,
            "transcription": transcription,
            "items_list": _format_items(items),
            "n_items": len(items),
        },
    )
    return _ajouter_au_systeme(messages, blocs)
