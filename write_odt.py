"""
Write the completed AML assignment draft as an .odt file.

This continues the user's original `Untitled 1.odt` draft in their own
loose first-person voice (they are an L2 English writer) but with
cleaner grammar, the actual experimental results, and — importantly —
a deeper justification for *why each method was chosen over the
alternatives that were considered and rejected*.

The output goes to a new file alongside the original so we do not
clobber it: `Untitled 1 - completed.odt`.

After saving, the script re-opens the .odt, extracts every text run,
and reports the word count using a library-based tokeniser
(odf text extraction + str.split) so the length is measured, not
estimated.
"""

from __future__ import annotations

from pathlib import Path

from odf.opendocument import OpenDocumentText, load
from odf.style import ParagraphProperties, Style, TextProperties
from odf.text import H, P
from odf import teletype

OUT = Path(__file__).resolve().parent / "Untitled 1 - completed.odt"

doc = OpenDocumentText()

# --- Styles ---
h1 = Style(name="H1", family="paragraph", parentstylename="Heading 1")
h1.addElement(TextProperties(fontsize="18pt", fontweight="bold"))
doc.styles.addElement(h1)

h2 = Style(name="H2", family="paragraph", parentstylename="Heading 2")
h2.addElement(TextProperties(fontsize="14pt", fontweight="bold"))
doc.styles.addElement(h2)

body = Style(name="Body", family="paragraph")
body.addElement(ParagraphProperties(margintop="0.06in", marginbottom="0.06in"))
body.addElement(TextProperties(fontsize="11pt"))
doc.styles.addElement(body)


def heading(level: str, text: str) -> None:
    style = "H1" if level == "1" else "H2"
    doc.text.addElement(H(outlinelevel=int(level), stylename=style, text=text))


def para(text: str) -> None:
    doc.text.addElement(P(stylename="Body", text=text))


# ============================== CONTENT ==============================

heading("1", "AML assignment draft")

# ---------------------------------------------------------------- T1
heading("1", "Part 1 — NLP (sentiment analysis)")

heading("2", "Choice of algorithm — and what I rejected")

para(
    "Normally in 2026, for an NLP task with no real performance "
    "constraint, I would just throw a large language model at the "
    "problem and call it a day — modern LLMs contain most of the "
    "public internet as base knowledge, so classifying a movie review "
    "is already inside their training distribution. I decided against "
    "this for two reasons. The first is academic honesty: an LLM has "
    "almost certainly already seen the Pang & Lee review sentences "
    "during pre-training, so any score it produces is contaminated by "
    "memorisation and tells me nothing about a system I actually "
    "designed. The second is that the module is assessed on the "
    "reasoning behind design decisions, and an LLM call hides every "
    "decision inside a black box I cannot inspect, defend, or debug. "
    "So I built a deliberately classical, fully transparent system "
    "where every component can be justified individually."
)

para(
    "The dataset has a twist: the review corpus is mixed with roughly "
    "26% Enron-style business email whose 0/1 label is assigned at "
    "random. I confirmed this empirically — the spam is split almost "
    "exactly evenly between the two classes (25.6% vs 26.0% in "
    "training), which means the spam carries no usable label "
    "information at all. Two architectures were on the table. The "
    "first is a single classifier trained on the noisy labels, "
    "trusting regularisation to absorb the noise. The second is a "
    "two-stage pipeline: a spam detector first, then a sentiment "
    "classifier that only ever sees genuine reviews. I chose the "
    "two-stage design, and the reasoning is worth spelling out "
    "because it is the single most important decision in Task 1."
)

para(
    "Random labels on the spam mean roughly 50% label noise applied "
    "to about a quarter of the training set. Frenay & Verleysen "
    "(2014), in their survey of label noise, show that linear "
    "classifiers stay well-behaved up to maybe 10–15% symmetric noise "
    "but degrade sharply beyond that — and the noise here, "
    "concentrated on a recognisable sub-population, is worse than "
    "uniform noise because the model can actually learn to be "
    "confidently wrong on email-shaped text. A single-stage model "
    "would also waste capacity learning a decision boundary that "
    "spans two completely different text distributions (terse "
    "one-sentence film reviews versus multi-line business email). "
    "Finally, the brief explicitly requires that spam in the test set "
    "receives a dummy label rather than 0 or 1; only an architecture "
    "that can abstain satisfies this, and a two-stage pipeline gives "
    "abstention for free. The cost of the two-stage design is that "
    "errors compound, but since stage one turned out to be almost "
    "perfect, that cost is negligible in practice."
)

heading("2", "Dataset and why I did not augment it")

para(
    "My first instinct was to enrich the training data with "
    "synthetically generated reviews from an LLM, on the principle "
    "that machine-learning models interpolate well between data "
    "points but extrapolate badly, so broader coverage should help. "
    "After actually inspecting the data I dropped this plan. The "
    "review portion is the Pang & Lee 2005 sentence-polarity corpus, "
    "which is already balanced, curated, and broad in vocabulary; the "
    "spam portion is Enron email with an extremely consistent surface "
    "signature. Injecting synthetic data risked two harms with little "
    "upside. It would shift the training distribution away from the "
    "held-out validation and test distributions (so my validation "
    "score would stop predicting test performance), and LLM-written "
    "reviews tend to be cleaner and more formulaic than human ones, "
    "which would teach the model the wrong stylistic priors. The "
    "honest broad-coverage test is not synthetic data anyway — it is "
    "the external NLTK corpus described later, which is real human "
    "writing from a genuinely different distribution."
)

heading("2", "Stage 1 — spam detector")

para(
    "For features I used character n-grams of length 3 to 5, weighted "
    "by TF-IDF, rather than word n-grams. This is a considered choice. "
    "The spam is not distinguished by its topic so much as by its "
    "surface form: carriage-return/line-feed pairs, runs of digits "
    "inside deal identifiers such as 98-6736, and email furniture like "
    "the literal token 'Subject:'. Word tokenisers throw most of this "
    "away — they discard newlines and split on punctuation, so the "
    "very signal that separates email from a film review is lost "
    "before the classifier sees it. Character n-grams keep it: a "
    "newline simply becomes part of a character triple, and 'subj' is "
    "captured regardless of casing. Character n-grams are also robust "
    "to the variety of digit and punctuation patterns in the email "
    "without me hand-writing a brittle regular expression for each one."
)

para(
    "For the classifier I used logistic regression. I considered a "
    "linear SVM and Naive Bayes as alternatives. Naive Bayes assumes "
    "conditional independence between features, which is badly "
    "violated by overlapping character n-grams, so I expected it to "
    "be poorly calibrated here. A linear SVM would match logistic "
    "regression on raw accuracy, but it does not produce probabilities "
    "without an extra calibration step, and I wanted a genuine "
    "probability so the pipeline can apply a tunable abstention "
    "threshold. Logistic regression gives a calibrated probability "
    "directly, is convex and so has no local-minima worries, and its "
    "coefficients are directly readable. The training labels came "
    "from a weak-supervision rule: an item is spam if it begins with "
    "'Subject:'. I verified that this rule is essentially noise-free "
    "on this data — 100% of suspected spam items carry both that "
    "prefix and CRLF newlines, and 0% of reviews carry either, so the "
    "two populations are perfectly separable on the surface. I still "
    "trained a model on top of the rule rather than just applying the "
    "rule, because the learned character-n-gram model generalises to "
    "the email *style* and would still flag a spam item that, for "
    "whatever reason, did not start with that exact prefix. This is "
    "defence-in-depth, following the data-programming idea of Ratner "
    "et al. (2016)."
)

para(
    "Results justify the effort: F1 of 0.9990 in five-fold "
    "cross-validation on training and 0.9985 on the held-out "
    "validation set, with exactly one rule/model disagreement on "
    "validation (a short personal email beginning 'Subject: "
    "pictures!' that the learned model judged not-spam). For all "
    "practical purposes stage one is solved, so the compounding-error "
    "risk of the two-stage design never materialises."
)

heading("2", "Stage 2 — sentiment classifier, and the representation question")

para(
    "Once spam is removed, training is 8,530 perfectly balanced "
    "review sentences. The central question for stage two is the "
    "representation — how to turn a sentence into a feature vector — "
    "so I compared two whole families under one controlled protocol "
    "(identical split, identical evaluation, linear classifier held "
    "as constant as possible)."
)

para(
    "The first family is sparse bag-of-n-grams with TF-IDF weighting. "
    "I used TF-IDF rather than raw counts because raw counts let "
    "ubiquitous words such as 'the' or 'movie' dominate the vector "
    "while contributing nothing to polarity; TF-IDF down-weights "
    "exactly those terms, and the sublinear variant further damps the "
    "effect of a word being repeated. I deliberately did not apply "
    "stop-word removal or stemming. Standard stop-word lists delete "
    "negation words — 'no', 'not', 'nor' — which are central to "
    "sentiment, and Pang et al. (2002) found stemming gives no "
    "benefit on this task and can even merge an emphatic form with a "
    "neutral one. Instead I let 'max_df' drop terms that appear in "
    "almost every document, which is a data-driven stop-list that "
    "cannot accidentally remove a polarity word. I also kept "
    "bigrams, not just unigrams, because a bigram is the cheapest way "
    "to capture short-range word order — 'not good' is a different "
    "feature from 'good'."
)

para(
    "The second family is dense word embeddings. I used pre-trained "
    "GloVe-100 vectors (Pennington et al. 2014), which cover 96.7% of "
    "the review tokens, and built a sentence vector by averaging the "
    "word vectors. I tested two pooling rules: a plain mean, and an "
    "IDF-weighted mean in the spirit of Arora et al. (2017), which "
    "down-weights common words during the average. Embeddings are the "
    "representation people reach for first in 2026, so it mattered to "
    "test them honestly rather than assume."
)

para(
    "On classifiers I compared logistic regression, a linear SVM, and "
    "Multinomial Naive Bayes. Naive Bayes is included precisely "
    "because Pang et al. (2002) and Wang & Manning (2012) found it is "
    "a stubbornly strong baseline on short reviews despite its naive "
    "independence assumption — it would be intellectually dishonest "
    "to omit it just because it is simple. The linear SVM tests "
    "whether a max-margin objective behaves differently from logistic "
    "regression's log-loss on this sparse, high-dimensional space, as "
    "Joachims (1998) argued SVMs should. Logistic regression is the "
    "reference point: convex, calibrated, interpretable. I kept all "
    "models linear on purpose, so that any score difference is "
    "attributable to the representation and not to model capacity."
)

para(
    "The headline numbers, on the held-out validation set with "
    "95% bootstrap confidence intervals from 1000 resamples: word "
    "TF-IDF with logistic regression reached 74.8%; Naive Bayes "
    "77.6%; the linear SVM 74.7%; the word-plus-character TF-IDF "
    "union with logistic regression was best at 78.2% (CI 0.757 to "
    "0.806). The two dense GloVe models were the worst of the six — "
    "72.7% for the plain mean and 70.8% for the IDF-weighted mean. "
    "The character n-grams in the winning model add a small but "
    "reliable lift over words alone because they capture sub-word "
    "style — suffixes like '-less' or '-ing', emphatic punctuation — "
    "that a word tokeniser cannot see."
)

para(
    "The most interesting result is that the dense embeddings lost, "
    "and it is worth saying why, because the intuition that "
    "'embeddings are modern, therefore better' is exactly the kind of "
    "assumption the module wants me to interrogate. Averaging a "
    "sentence into one vector destroys the very thing sentiment "
    "depends on. A review is short and its polarity is usually "
    "carried by one or two strong words; in a bag-of-n-grams model "
    "the classifier can place a large dedicated weight on 'horrible' "
    "or 'fantastic', but in a mean-pooled embedding those words are "
    "diluted by every neutral word around them, and the polarity "
    "signal is smeared toward the centroid. The IDF-weighted mean did "
    "not rescue this — it actually scored lower — because IDF "
    "up-weights rare nouns (proper names, obscure film vocabulary) "
    "that are rare precisely because they are uninformative about "
    "sentiment. The genuine advantage of embeddings, generalising to "
    "words unseen in training, barely applies here because training "
    "and validation share almost all of their vocabulary."
)

heading("2", "Failure analysis and bias")

para(
    "The best model still misclassifies 232 of 1066 validation "
    "reviews. Tagging the errors shows the dominant modes are all "
    "compositional: contrast clauses (50 cases with 'but', 'however', "
    "'although'), negation (38 cases), and very short reviews (27). "
    "The highest-confidence mistakes are the most telling. Idiomatic "
    "negation: 'you can do no wrong with jason x' is positive, but "
    "the model sees 'no' and 'wrong' and votes negative. The "
    "'too X to Y' construction: 'too savvy a filmmaker to let this "
    "morph into a typical romantic triangle' is praise, but 'too' is "
    "the model's single strongest negative feature. Concession: "
    "'hilarious comedy though stymied by accents thick as mud' is a "
    "negative verdict, but 'hilarious' wins the vote. These are all "
    "cases where meaning depends on word order and sentence "
    "structure, which a bag-of-words model discards by construction. "
    "A contextual transformer would fix most of them, but at a large "
    "cost in compute and interpretability — and it would reintroduce "
    "the memorisation concern that made me reject an LLM in the first "
    "place. The systematic bias to flag for the assignment is the "
    "'too' coefficient: a single feature weight that is right on "
    "average but wrong on a recognisable, well-defined slice of "
    "inputs."
)

para(
    "On the external NLTK movie_reviews corpus — 2000 full-length "
    "reviews, around 750 words each, against the single sentences I "
    "trained on — the best model holds 75.6% accuracy, only about "
    "three points below validation. But its recall is badly "
    "asymmetric: 98% on negative reviews, 54% on positive. Long "
    "positive reviews routinely contain critical asides ('the second "
    "act drags'), and a model trained only on single, "
    "strongly-polarised sentences has never learned that a positive "
    "verdict can survive local negativity. This is a length-driven "
    "domain shift, and it is a more honest generalisation test than "
    "any synthetic data I could have generated."
)

# ---------------------------------------------------------------- T2
heading("1", "Part 2 — Computer Vision (face alignment)")

heading("2", "Problem framing and what I rejected")

para(
    "The task is to predict five facial landmarks — the two eyes, the "
    "nose tip, and the two mouth corners — from a 256x256 face image. "
    "This is multi-output regression: ten continuous numbers per "
    "face, not a classification. The brief warns of unwanted "
    "variability from transformations that distort the images, and "
    "inspecting the data confirms substantial in-plane head rotation "
    "and scale variation, so robustness has to be designed in, not "
    "hoped for. I treated it as an explicit, measured part of the "
    "evaluation rather than an afterthought."
)

para(
    "The obvious 2026 choice is a convolutional network with a "
    "landmark-regression head. I deliberately did not start there. "
    "With only 2600 training images, a deep network trained from "
    "scratch would overfit badly, and a transfer-learned one would "
    "again hide the design decisions inside pre-trained weights — the "
    "same transparency problem as the LLM in Task 1. A classical "
    "pipeline of a hand-crafted descriptor plus a linear regressor is "
    "exactly the regime where small data with strong, regular "
    "structure still competes, and every component stays inspectable. "
    "So I built three models of increasing sophistication and "
    "compared them honestly."
)

heading("2", "Pre-processing — and the alternatives")

para(
    "Every image is converted to greyscale, histogram-equalised, and "
    "resized to 128x128. Greyscale is justified because facial "
    "landmarks are defined by shape and edge structure, not colour; "
    "keeping three channels would triple the feature cost for no "
    "located-landmark benefit. Histogram equalisation was chosen over "
    "the alternatives of doing nothing, simple mean/standard-deviation "
    "normalisation, or the locally-adaptive CLAHE. Doing nothing "
    "leaves the descriptor sensitive to absolute brightness, and the "
    "dataset's lighting varies enormously. Mean/std normalisation only "
    "shifts and scales intensities, so it cannot fix a non-linear "
    "contrast change. CLAHE is more powerful but introduces tile-size "
    "parameters and can amplify noise; plain global equalisation maps "
    "every image onto one canonical intensity histogram with no "
    "parameters to tune, and the robustness study later confirms this "
    "was the right call. The 128x128 resize is a compute compromise: "
    "at full 256x256 the HOG descriptor is roughly four times larger "
    "with no measurable accuracy gain, while 128 still resolves the "
    "eye and mouth structures comfortably."
)

heading("2", "Feature representation — why HOG")

para(
    "For the image representation I chose the Histogram of Oriented "
    "Gradients (Dalal & Triggs 2005). The alternatives I weighed were "
    "raw pixels, Local Binary Patterns, and SIFT-style keypoint "
    "descriptors. Raw pixels are a poor regression input because they "
    "are dominated by lighting and by tiny translations — shifting a "
    "face one pixel changes every value — and they leave the linear "
    "model no edge structure to grab. LBP is excellent for texture "
    "classification but encodes fine micro-texture rather than the "
    "larger edge geometry that locates an eye corner. SIFT is built "
    "for sparse keypoint matching between images, not for a "
    "fixed-length dense description of a whole aligned crop. HOG sits "
    "exactly where this problem lives: it summarises the dominant "
    "gradient orientation in each small cell and normalises in "
    "overlapping blocks, so it captures the structured edges around "
    "eyes, nose and mouth while being stable to illumination. After "
    "histogram equalisation, HOG over the 128x128 image is a strong, "
    "compact, and interpretable descriptor."
)

heading("2", "Models compared")

para(
    "M0, the mean-shape baseline, predicts the average training "
    "landmark positions for every image and ignores the picture "
    "entirely. It exists to set the floor: any real model must beat "
    "the score obtainable by knowing nothing about the specific face."
)

para(
    "M1 is global HOG followed by ridge regression. The regression "
    "choice deserves explanation. With about 8100 HOG features and "
    "only 2600 training images the problem is wide — more unknowns "
    "than equations — so ordinary least squares is underdetermined "
    "and would overfit catastrophically. I considered Lasso (L1), "
    "ridge (L2), and a random forest. Lasso forces a sparse solution, "
    "but HOG features are correlated and jointly informative, so "
    "zeroing most of them throws away signal; L1 also has no "
    "closed-form solution. A random forest is non-linear and could in "
    "principle do better, but it would obscure the contribution of "
    "the representation, which is the comparison I am actually trying "
    "to make. Ridge regression is the right tool: it has a stable "
    "closed-form solution even when features outnumber samples, it "
    "handles correlated features gracefully by shrinking them "
    "together, and its single regularisation strength is easy to "
    "tune. I swept that strength over five orders of magnitude and "
    "picked the value that minimised validation error."
)

para(
    "M2 is a cascaded shape regression. The motivation, following Cao "
    "et al. (2014) and Kazemi & Sullivan (2014), is a real weakness "
    "of M1: a global descriptor regressed in one shot has to explain "
    "every pixel at once, and the same pixel means different things "
    "depending on where the face actually sits in the frame. A "
    "cascade fixes this by iterating. It starts from the M1 "
    "prediction, then at each of three stages extracts small HOG "
    "patches centred on the *current* estimate of each landmark and "
    "regresses only the residual correction. Because the patches are "
    "indexed to the current shape, the features become increasingly "
    "relevant as the estimate improves — the model gets to 'look "
    "again, locally' instead of guessing globally. I used three "
    "stages because that is where the returns flatten: stage zero "
    "gave NME 0.052, stage one 0.042, stage two 0.040, stage three "
    "0.040. A fourth stage would add compute for no measurable gain."
)

para(
    "Each regressor is trained by ridge — minimising squared residual "
    "error plus an L2 penalty on the weights — and there is no "
    "non-convex optimisation anywhere in the pipeline, so training is "
    "fast and deterministic. The whole of M2, including feature "
    "extraction and test inference, trains in about sixteen seconds "
    "on a CPU; no GPU is required."
)

heading("2", "Results")

para(
    "The primary metric is Normalised Mean Error: the mean Euclidean "
    "distance between predicted and true landmarks for a face, "
    "divided by that face's inter-ocular (eye-to-eye) distance. "
    "Normalising this way is the standard 300W-benchmark convention "
    "and makes the score comparable across faces of different sizes "
    "in the frame; a raw pixel error would unfairly punish larger "
    "faces. On the 211-image validation set, M0 scores NME 0.127 "
    "(12.24 pixels of mean error); M1 scores 0.052 (4.99 pixels), a "
    "59% reduction; and M2 scores 0.0395 (3.76 pixels), a further 24% "
    "reduction. The cumulative error distribution curve shows M2 "
    "placing more than 85% of validation faces under NME 0.05, where "
    "M0 reaches only about 30%."
)

para(
    "Broken down per landmark, the eyes are located most accurately "
    "(median error around 2.5 pixels) and the mouth corners least "
    "accurately (around 3.5 pixels), with the nose between. This "
    "matches intuition: the eye is the highest-contrast facial "
    "feature, with sharp sclera-iris-pupil boundaries that HOG "
    "captures cleanly, whereas the mouth corner moves a great deal "
    "between expressions, so its position is genuinely more variable "
    "and therefore harder to pin down. The worst whole-face "
    "predictions are all strong-pose images — profile and "
    "three-quarter views — where a landmark is effectively off-frame; "
    "the mean-shape initialisation pulls the estimate toward a "
    "frontal centroid and the local patches have nothing to lock "
    "onto. That is the model's clearest systematic bias."
)

heading("2", "Robustness analysis")

para(
    "Because the brief asks for it explicitly, I retrained M2 once on "
    "clean data and then evaluated it on validation images perturbed "
    "three ways. Under additive Gaussian noise the error grows "
    "roughly linearly with the noise standard deviation, from NME "
    "0.039 with no noise to 0.087 at a heavy sigma of 60. HOG is "
    "computed from local intensity gradients, and noise inflates "
    "those gradients, so some degradation is unavoidable — but the "
    "linear, gentle slope shows the block normalisation inside HOG is "
    "absorbing most of it. Under in-plane rotation the error climbs "
    "steeply: 0.046 at ten degrees, about 0.07 at twenty, and 0.121 "
    "at thirty. This is expected and informative — HOG bins gradients "
    "into orientation channels, so rotating the input literally "
    "shifts the histogram into different bins, and the cascade cannot "
    "recover because its shape-indexed patches inherit the same "
    "rotation. Under gamma contrast change the error is essentially "
    "flat, staying between 0.039 and 0.040 across a wide gamma range. "
    "That flat line is the clearest single vindication of a design "
    "decision in the whole project: the histogram-equalisation step "
    "maps every image, however it was lit, onto the same intensity "
    "distribution before HOG ever sees it, so contrast variation is "
    "removed at the door. The rotation weakness has a known remedy — "
    "augmenting training with randomly rotated copies, or using a "
    "rotation-invariant descriptor — which could be added without "
    "redesigning the pipeline."
)

heading("2", "Test submission and reflection")

para(
    "Task 1 test predictions are written to "
    "sentiment_test_predictions.csv with the dummy label 2 reserved "
    "for detected spam; Task 2 predictions are written to "
    "results_task2.csv via the worksheet's save_as_csv helper, using "
    "model M2. Both systems are deliberately classical and small, and "
    "that was the point: every choice — decoupling spam because the "
    "noise rate makes joint training unsafe, preferring sparse TF-IDF "
    "to dense embeddings because the reviews are short and lexical, "
    "choosing HOG over raw pixels because gradients beat intensities "
    "for locating structure, cascading because local patches beat one "
    "global guess, equalising histograms to buy contrast invariance "
    "for free — is a decision I can name, defend, and show evidence "
    "for. The remaining failure modes, compositional sentiment in "
    "Task 1 and strong-pose faces with rotation sensitivity in "
    "Task 2, are well understood and have known modern remedies that "
    "could be bolted on incrementally."
)


doc.save(str(OUT))
print("wrote", OUT)

# ---------- measure word count with a library extractor ----------
reloaded = load(str(OUT))
words = 0
for el in reloaded.getElementsByType(P) + reloaded.getElementsByType(H):
    text = teletype.extractText(el)
    words += len(text.split())
print(f"word count (odf teletype extractor + str.split): {words}")
