"""Context, scholarly retrieval, pretrained inference, and attributable answers."""

import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.parse import urlencode, urlsplit
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import faiss
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

try:
    import spacy
except ImportError:
    spacy = None
    logging.getLogger(__name__).warning("spaCy unavailable; using basic English tokenization")

MODEL_REPOSITORY = "Xenova/all-MiniLM-L6-v2"
MODEL_REVISION = "751bff37182d3f1213fa05d7196b954e230abad9"
MODEL_FILES = ("tokenizer.json", "onnx/model_quantized.onnx")
FAISS_THREADS = 1


def scholarly_url(value):
    if not isinstance(value, str) or any(char.isspace() for char in value):
        return ""
    try:
        url = urlsplit(value)
        if url.scheme == "https" and url.netloc in {"doi.org", "openalex.org", "arxiv.org"} and url.path.strip("/"):
            return value
    except ValueError:
        pass
    return ""


@dataclass(frozen=True)
class Paper:
    title: str
    authors: str
    year: int | None
    url: str
    abstract: str
    provider: str
    work_id: str = ""

    def reference(self):
        return {field: value for field, value in asdict(self).items() if field != "abstract"}


class QueryProcessor:
    # A follow-up refers back to the topic ("it", "they", "this"), continues it ("and…", "tell me more",
    # "give an example"), or asks about an aspect without naming any topic ("What are the challenges?").
    REFERS_BACK = re.compile(r"\b(?:it|its|they|them|their|these|those|this)\b", re.IGNORECASE)
    # "it" that refers to nothing, as in "Is it true that FAISS needs a GPU?", is not a reference to the topic.
    EXPLETIVE = re.compile(
        r"\b(?:is it|it is|it's|was it|would it be|can it be)\s+(?:true|possible|necessary|important|better|worth|"
        r"common|safe|likely|known|correct|the case|fair|reasonable)\b|\bhow is it that\b", re.IGNORECASE)
    CONTINUES = re.compile(
        r"^(?:and|also|so|then|what about|how about|tell me more|more on|more about|can you elaborate|elaborate|"
        r"explain (?:more|further|that|in detail)|go on|continue|why|give (?:me )?(?:an? |some )?examples?|"
        r"any (?:examples?|more))\b", re.IGNORECASE)
    ASPECTS = {"limitation", "limitations", "challenge", "challenges", "advantage", "advantages", "disadvantage",
               "disadvantages", "benefit", "benefits", "drawback", "drawbacks", "application", "applications", "use",
               "uses", "used", "example", "examples", "detail", "details", "more", "why", "work", "works", "working",
               "pros", "cons", "future", "importance", "important", "purpose", "history", "types", "steps", "simple",
               "simply", "terms", "briefly", "again", "elaborate", "further"}
    FILLER = {"example", "examples", "detail", "details", "more", "why", "simple", "simply", "terms", "briefly",
              "again", "elaborate", "further"}
    GENERIC = {"compare", "describe", "explain", "find", "help", "paper", "papers", "please",
               "research", "show", "studies", "study", "summarize", "tell", "work"}
    # Used only when spaCy cannot be loaded; close to spaCy's own English stop words for question text.
    COMMON = set("a about also an and any are as at be been being but by can could did do does doing for from "
                 "get give go had has have how i in into is it its me my of on or our should so some than that the their "
                 "them then there these they this those to too used using very was we were what when where which "
                 "who whom why will with would you your".split())

    # The synopsis's "NLP Analysis" step: the kind of answer the question asks for.
    INTENTS = (
        ("comparison", re.compile(r"\b(?:compare[sd]?|comparison|versus|vs\.?|difference between|differ|better than)\b", re.I)),
        ("limitation", re.compile(r"\b(?:limitations?|weakness(?:es)?|drawbacks?|disadvantages?|challenges?|problems?|"
                                  r"trade[- ]?offs?)\b", re.I)),
        ("application", re.compile(r"\b(?:applications?|used for|use cases?|uses of|applied to)\b", re.I)),
        ("mechanism", re.compile(r"^(?:how (?:does|do|is|are|can)|explain how|in what way)\b", re.I)),
        ("definition", re.compile(r"^(?:what (?:is|are)|what does .{1,40} mean|define|definition of|meaning of|explain)\b",
                                  re.I)),
    )
    MINOR = {"a", "an", "and", "for", "from", "in", "of", "on", "the", "to", "with", "via", "by"}

    def __init__(self):
        self.nlp = spacy.blank("en") if spacy else None
        if self.nlp:
            self.nlp.add_pipe("sentencizer")

    def terms(self, value):
        if self.nlp:
            tokens = (token.lower_ for token in self.nlp.make_doc(value) if
                      not token.is_stop and (token.is_alpha or token.like_num))
        else:
            tokens = (token for token in re.findall(r"[^\W_]+", value.lower()) if token not in self.COMMON)
        return list(dict.fromkeys(token for token in tokens if len(token) > 1 and token not in self.GENERIC))

    def is_follow_up(self, question):
        if self.REFERS_BACK.search(self.EXPLETIVE.sub(" ", question)) or self.CONTINUES.search(question.strip()):
            return True
        # Nothing left once aspect words are removed: the question has no topic of its own.
        return not [term for term in self.terms(question) if term not in self.ASPECTS]

    def intent(self, question):
        return next((name for name, pattern in self.INTENTS if pattern.search(question.strip())), "general")

    @staticmethod
    def acronyms(text):
        """Abbreviations such as RAG, BM25 or FAISS: 2 to 8 characters, almost all capitals or digits.

        Mixed-case names such as OpenAlex, PyTorch or GitHub are not abbreviations and are left alone.
        """
        found = re.findall(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*s?\b", text)
        def capitals(word):
            core = word[:-1] if word.endswith("s") else word
            return sum(c.isupper() or c.isdigit() for c in core) >= max(2, len(core) - 1)
        return [word for word in dict.fromkeys(found) if 2 <= len(word) <= 8 and capitals(word)]

    @classmethod
    def expansion(cls, acronym, texts):
        """Find the long form an author gives an abbreviation, as in "Retrieval-Augmented Generation (RAG)"."""
        letters = [char.lower() for char in acronym if char.isupper()]
        pattern = re.compile(r"([A-Za-z][\w' -]{2,120}?)\s*\(\s*" + re.escape(acronym) + r"s?\s*\)")
        counts = Counter()
        for text in texts:
            for match in pattern.finditer(text):
                words = list(re.finditer(r"[A-Za-z][\w']*", match.group(1)))
                chosen = []
                for word in reversed(words):
                    if word.group().lower() in cls.MINOR and chosen:
                        continue
                    chosen.insert(0, word)
                    if len(chosen) == len(letters):
                        break
                if [word.group()[0].lower() for word in chosen] == letters:
                    phrase = match.group(1)[chosen[0].start():].strip()
                    counts[re.sub(r"\s+", " ", phrase).lower()] += 1
        return counts.most_common(1)[0][0] if counts else ""

    def sentences(self, value):
        text = value[:6000]
        if self.nlp:
            return [span.text.strip() for span in self.nlp(text).sents]
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)]


class ConversationContext:
    def __init__(self, processor):
        self.processor = processor

    @staticmethod
    def topic(resolved):
        """The conversation topic a resolved question carries: its own text, or the topic it follows up."""
        return resolved.split("\nFollow-up:", 1)[0].strip()

    def resolve(self, question, history):
        if history and self.processor.is_follow_up(question):
            # The topic of the latest turn: its own question, or the topic it was itself following up.
            # Earlier turns are searched backwards so that a turn without a stored topic is skipped.
            anchor = next((self.topic(turn.get("resolved") or turn.get("question") or "")
                           for turn in reversed(history) if turn.get("resolved") or turn.get("question")), "")
            if anchor:
                return f"{anchor[:450]}\nFollow-up: {question}"
        return question

    def search_text(self, resolved):
        # OpenAlex's search is keyword-based: the conversation's topic plus what the follow-up adds,
        # so "How does it compare with fine-tuning?" searches for the topic together with fine-tuning.
        anchor, _, follow = resolved.partition("\nFollow-up:")
        topic = self.processor.terms(anchor)
        extra = [term for term in self.processor.terms(follow)
                 if term not in topic and term not in self.processor.FILLER][:3]
        return " ".join(topic[:8 - len(extra)] + extra) or anchor[:160]


class HuggingFaceEncoder:
    def __init__(self, cache=None):
        self.cache = Path(cache or os.getenv("MODEL_CACHE_DIR") or Path.home() / ".cache" / "ignou-minilm")
        self.session = None
        self.lock = Lock()

    def _asset(self, relative_path):
        target = self.cache / Path(relative_path).name
        if target.is_file():
            return target
        url = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{relative_path}"
        self.cache.mkdir(parents=True, exist_ok=True)
        partial = None
        try:
            with urlopen(url, timeout=15) as response, NamedTemporaryFile(dir=self.cache, delete=False) as stream:
                partial = Path(stream.name)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            os.replace(partial, target)
        finally:
            if partial:
                partial.unlink(missing_ok=True)
        return target

    def load(self):
        with self.lock:
            if self.session is not None:
                return
            tokenizer_path, model_path = (self._asset(name) for name in MODEL_FILES)
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.enable_truncation(max_length=256)
            tokenizer.enable_padding()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.tokenizer = tokenizer
            self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
            self.inputs = {entry.name for entry in self.session.get_inputs()}

    def encode(self, texts):
        self.load()
        batches = []
        for start in range(0, len(texts), 12):
            tokens = self.tokenizer.encode_batch(texts[start:start + 12])
            masks = np.asarray([token.attention_mask for token in tokens], dtype=np.int64)
            values = {
                "input_ids": np.asarray([token.ids for token in tokens], dtype=np.int64),
                "attention_mask": masks,
                "token_type_ids": np.asarray([token.type_ids for token in tokens], dtype=np.int64),
            }
            hidden = self.session.run(None, {name: values[name] for name in self.inputs})[0]
            weighted = masks[:, :, None]
            pooled = (hidden * weighted).sum(axis=1) / weighted.sum(axis=1).clip(min=1)
            batches.append(pooled.astype(np.float32))
        vectors = np.ascontiguousarray(np.concatenate(batches))
        faiss.normalize_L2(vectors)
        return vectors


class OpenAlexSource:
    SELECT = "id,doi,display_name,publication_year,authorships,abstract_inverted_index,is_retracted"

    CACHE_SECONDS = 600

    def __init__(self, key=None, timeout=5):
        self.key = key
        self.timeout = timeout
        self.cache = {}
        self.cache_lock = Lock()

    def search(self, query):
        """Search OpenAlex, reusing a result for the same query for ten minutes."""
        key = query.strip().casefold()
        with self.cache_lock:
            hit = self.cache.get(key)
            if hit and time.monotonic() - hit[0] < self.CACHE_SECONDS:
                return hit[1]
        papers = self.fetch(query)
        with self.cache_lock:
            if len(self.cache) >= 256:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = (time.monotonic(), papers)
        return papers

    @staticmethod
    def restore_abstract(index):
        if not isinstance(index, dict):
            return ""
        terms = {position: word for word, positions in index.items()
                 if isinstance(positions, list) for position in positions
                 if isinstance(position, int) and 0 <= position < 1000}
        return " ".join(terms[position] for position in sorted(terms))

    def fetch(self, query):
        params = urlencode({"search": query, "per-page": 12, "select": self.SELECT})
        headers = {"User-Agent": "BCSP-064 Research Assistant/1.0", "Accept": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        request = Request(f"https://api.openalex.org/works?{params}", headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read(2_000_000))
        papers = []
        for record in payload["results"]:
            title = record.get("display_name")
            work_id = scholarly_url(record.get("id"))
            url = scholarly_url(record.get("doi")) or work_id
            if not title or not url or record.get("is_retracted"):
                continue
            # Some records omit the author object; skip those names instead of dropping every live result.
            authors = [(entry.get("author") or {}).get("display_name")
                       for entry in (record.get("authorships") or [])[:6] if isinstance(entry, dict)]
            papers.append(Paper(title, ", ".join(filter(None, authors)), record.get("publication_year"),
                                url, self.restore_abstract(record.get("abstract_inverted_index")),
                                "OpenAlex", work_id))
        return papers


class LocalCorpus:
    """Original summaries with publication identifiers for the offline demo."""

    PAPERS = (
        Paper("Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "Patrick Lewis et al.", 2020,
              "https://arxiv.org/abs/2005.11401",
              "Retrieval-augmented generation (RAG) combines a pretrained sequence-to-sequence model with a dense index of Wikipedia passages. "
              "The authors evaluate the combination on knowledge-intensive tasks and report more specific and factual text than a parametric-only baseline. "
              "Retrieval-Augmented Generation uses a fixed Wikipedia snapshot in the reported experiments, so its coverage depends on the indexed passages.",
              "Local corpus"),
        Paper("Dense Passage Retrieval for Open-Domain Question Answering", "Vladimir Karpukhin et al.", 2020,
              "https://arxiv.org/abs/2004.04906",
              "Dense Passage Retrieval (DPR) learns separate question and passage encoders and retrieves passages through vector similarity. "
              "On the reported open-domain question-answering datasets it improves top-20 passage retrieval accuracy over a BM25 baseline. "
              "A retrieved passage is a candidate for the answer and still needs to be interpreted by a reader model.", "Local corpus"),
        Paper("Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", "Nils Reimers and Iryna Gurevych", 2019,
              "https://doi.org/10.18653/v1/D19-1410",
              "Sentence-BERT uses siamese and triplet networks to obtain sentence embeddings that can be compared with cosine similarity. "
              "Computing embeddings separately avoids scoring each possible sentence pair with a joint BERT forward pass. "
              "The work evaluates sentence embeddings on semantic textual similarity and transfer tasks.", "Local corpus"),
        Paper("BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
              "Nandan Thakur et al.", 2021, "https://arxiv.org/abs/2104.08663",
              "The BEIR benchmark compares lexical, dense, sparse, late-interaction, and reranking retrieval systems across heterogeneous datasets. "
              "BM25 provides a competitive baseline across domains, while strong reranking methods often add substantial computational cost. "
              "The evaluation shows that retrieval quality varies by dataset and domain.", "Local corpus"),
        Paper("Attention Is All You Need", "Ashish Vaswani et al.", 2017, "https://arxiv.org/abs/1706.03762",
              "The Transformer architecture uses attention rather than recurrence to relate positions in a sequence. "
              "Multi-head attention lets the model attend to information from different representation subspaces. "
              "The original experiments compare machine translation quality and training efficiency.", "Local corpus"),
        Paper("Lost in the Middle: How Language Models Use Long Contexts", "Nelson F. Liu et al.", 2023,
              "https://arxiv.org/abs/2307.03172",
              "In multi-document question answering, models often perform better when relevant information appears near the beginning or end of the input. "
              "Performance can decline when the same evidence is placed in the middle of a long context. "
              "The position effect is a limitation for applications that supply many retrieved passages at once.", "Local corpus"),
        Paper("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "Jacob Devlin et al.", 2019,
              "https://doi.org/10.18653/v1/N19-1423",
              "BERT pretrains deep bidirectional representations and adapts them to language understanding tasks. "
              "Masked language modeling uses context on both sides of a masked token. "
              "The paper evaluates the resulting representations on question answering and other language understanding benchmarks.",
              "Local corpus"),
        Paper("Billion-scale similarity search with GPUs", "Jeff Johnson, Matthijs Douze and Hervé Jégou", 2017,
              "https://arxiv.org/abs/1702.08734",
              "FAISS provides efficient similarity search algorithms for vectors. "
              "The paper describes GPU implementations of exact and approximate nearest-neighbor search and evaluates large vector indexes. "
              "Approximate search can trade recall for speed and index size.", "Local corpus"),
    )

    def search(self, _query):
        return self.PAPERS


class SemanticRetriever:
    # Cosine similarity below which a paper is not treated as evidence for the question.
    THRESHOLD = 0.30

    def __init__(self, encoder, local, external):
        self.encoder = encoder
        self.local = local
        self.external = external

    @staticmethod
    def distinct(papers):
        seen = set()
        for paper in papers:
            keys = {paper.url.casefold(), re.sub(r"\W", "", paper.title.casefold())}
            if paper.work_id:
                keys.add(paper.work_id.casefold())
            if not keys & seen:
                yield paper
            seen.update(keys)

    def retrieve(self, resolved, query):
        notice = ""
        online = []
        if self.external:
            # One retry covers OpenAlex's brief rate limits and slow responses.
            failed = False
            for attempt in range(2):
                try:
                    online = self.external.search(query)
                    failed = False
                    break
                except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
                    failed = True
                    logging.getLogger(__name__).warning("OpenAlex search failed (attempt %d): %s", attempt + 1, error)
                    # Only rate limits, server errors and timeouts are worth a second try.
                    if isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                        break
            if failed:
                notice = "OpenAlex is temporarily unavailable. Showing matches from the local collection."
        if not self.external:
            notice = "Local collection only."
        local = self.local.search(query)
        # Only papers with an abstract can be quoted or written from, so only they compete for the top five.
        candidates = [paper for paper in self.distinct([*online, *local]) if paper.abstract]
        if not candidates:
            return [], notice
        vectors = self.encoder.encode([resolved, *(p.title + ". " + p.abstract for p in candidates)])
        faiss.omp_set_num_threads(FAISS_THREADS)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors[1:])
        scores, positions = index.search(vectors[:1], min(5, len(candidates)))
        return [(candidates[pos], float(score)) for pos, score in zip(positions[0], scores[0])
                if score >= self.THRESHOLD], notice


class EvidenceComposer:
    """The synopsis's "Response Structuring" step: a direct answer first, then supporting evidence.

    Every sentence is quoted from a retrieved paper and carries its citation number, so the answer never
    states anything that cannot be checked against a source.
    """

    LIMITATION = re.compile(r"\b(?:limitations?|weakness(?:es)?|trade[- ]?offs?|drawbacks?|challenges?)\b", re.I)
    CUES = {
        "definition": re.compile(r"\b(?:is|are|refers? to|combines?|describes?|denotes?|means?|defined as)\b", re.I),
        "mechanism": re.compile(r"\b(?:uses?|using|by|through|retriev\w*|learns?|encod\w*|combin\w*|comput\w*)\b", re.I),
        "comparison": re.compile(r"\b(?:than|compared|outperform\w*|versus|whereas|baseline|better|worse)\b", re.I),
        "limitation": re.compile(r"\b(?:decline|depends? on|limited|limitation|fixed|cost|trade|weak|harder|fail\w*)\b", re.I),
        "application": re.compile(r"\b(?:used|applied|applications?|tasks?|domains?|question answering)\b", re.I),
    }
    BONUS = {"definition": .08, "mechanism": .05, "comparison": .08, "limitation": .10, "application": .05}
    HEADINGS = {"definition": "Definition", "mechanism": "How it works", "comparison": "Comparison",
                "limitation": "Limitations", "application": "Applications", "general": "Answer"}

    def __init__(self, processor, encoder):
        self.processor = processor
        self.encoder = encoder

    def best_sentence(self, resolved, paper, intent):
        sentences = [text for text in self.processor.sentences(paper.abstract) if 35 <= len(text) <= 400][:22]
        if not sentences:
            return None
        vectors = self.encoder.encode([resolved, *sentences])
        scores = vectors[1:] @ vectors[0]
        ranking = scores.copy()
        if intent in self.CUES:
            ranking += [self.BONUS[intent] * bool(self.CUES[intent].search(text)) for text in sentences]
        best = int(ranking.argmax())
        # A curated summary opens with its definition; prefer it for "what is" and "how" questions.
        if (intent in {"definition", "mechanism", "general"} and paper.provider == "Local corpus"
                and scores[0] >= max(0.27, scores[best] - 0.15)):
            best = 0
        if scores[best] < 0.27:
            return None
        return float(ranking[best]), sentences[best]

    def compose(self, resolved, ranked, notice, intent="general"):
        statements = []
        for order, (paper, _) in enumerate(ranked):
            chosen = self.best_sentence(resolved, paper, intent)
            if chosen:
                statements.append((chosen[0], order, paper, chosen[1]))
        statements = statements[:4]
        if statements:
            # Lead with the sentence that best answers the question; keep the rest in relevance order.
            lead = max(statements, key=lambda item: item[0])
            statements = [lead] + [item for item in statements if item is not lead]
        sources, cited = [], []
        for _, _, paper, text in statements:
            sources.append(paper.reference())
            cited.append({"text": text, "citation": len(sources)})
        summary, findings = cited[:2], cited[2:]
        cited_urls = {source["url"] for source in sources}
        leads = [paper.reference() for paper, _ in ranked if paper.url not in cited_urls]
        if cited:
            overview = (f"This answer is drawn from {len(sources)} paper{'s' if len(sources) > 1 else ''}. "
                        "Each statement is quoted from the cited source, so it can be checked in the original.")
        else:
            overview = "I found too little relevant evidence to support an answer to this question."
        return {"intent": intent, "heading": self.HEADINGS[intent], "overview": overview, "summary": summary,
                "findings": findings, "sources": sources, "leads": leads, "notice": notice}


class GeminiWriter:
    """Response Generation: a language model writes the answer in plain prose from the retrieved papers.

    The model sees only the numbered paper summaries, is told to cite them as [1], [2], and any citation
    number that does not match a supplied paper is removed before the answer is shown.
    """

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    # Tried in turn when the main model is busy; Google reports high demand on individual models at times.
    FALLBACKS = ("gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3-flash-preview")
    STYLE = {
        "definition": "Begin with a one-sentence definition, then explain the idea in simple terms.",
        "mechanism": "Explain step by step how it works.",
        "comparison": "Compare the items point by point and say when each is preferable.",
        "limitation": "Explain the main limitations and why they matter.",
        "application": "Describe the main applications with brief examples.",
        "general": "Answer the question directly, then add the most useful supporting detail.",
    }

    def __init__(self, key, model=None, timeout=45):
        self.key = key
        self.model = model or os.getenv("GEMINI_MODEL") or "gemini-flash-latest"
        self.timeout = timeout

    def prompt(self, question, history, papers, intent):
        earlier = "\n".join(f"Q: {turn['question']}\nA: {turn['answer'].get('text', '')[:600]}"
                            for turn in history[-2:])
        evidence = "\n\n".join(f"[{number}] {paper.title} ({paper.authors}, {paper.year}). {paper.abstract[:1500]}"
                               for number, paper in enumerate(papers, start=1))
        return (
            "You are Lumen, a research assistant for students and researchers. Write a clear, accurate answer "
            "that a non-specialist can follow, in formal academic English.\n"
            f"{self.STYLE.get(intent, self.STYLE['general'])}\n"
            "Rules:\n"
            "- Use only the numbered papers below as your evidence. End every factual sentence with the number "
            "of the paper that supports it in square brackets, e.g. [1] or [2][3].\n"
            "- Do not state facts that the papers do not support, do not cite a paper for anything it does not "
            "say, and never invent papers, numbers, or results.\n"
            "- If the papers do not cover part of the question, say so in one short sentence without a citation.\n"
            "- Start directly with the answer. Do not greet the reader, introduce yourself, or restate the question.\n"
            "- Write 2 to 4 short paragraphs (about 120 to 250 words). Plain text only: no headings, no bold, "
            "no markdown. Separate paragraphs with a blank line.\n\n"
            + (f"Earlier in this conversation:\n{earlier}\n\n" if earlier else "")
            + f"Papers:\n{evidence}\n\n"
            + f"Question: {question}"
        )

    def deadline(self, limit=None):
        """This writer's own time limit, shortened to the request's overall budget when one is given."""
        own = time.monotonic() + self.timeout
        return min(own, limit) if limit else own

    @staticmethod
    def remaining(deadline, cap):
        left = deadline - time.monotonic()
        if left < 5:
            raise TimeoutError("no time left in the answer budget")
        return min(cap, left)

    def write(self, question, history, papers, intent, deadline=None):
        body = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": self.prompt(question, history, papers, intent)}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4000},
        }).encode()
        # Busy or rate-limited responses move on to the next model in the chain.
        attempts = [self.model, *(name for name in self.FALLBACKS if name != self.model)]
        deadline = self.deadline(deadline)
        for number, model in enumerate(attempts):
            request = Request(self.ENDPOINT.format(model=model), data=body, method="POST",
                              headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
            try:
                with urlopen(request, timeout=self.remaining(deadline, 25)) as response:
                    payload = json.loads(response.read(1_000_000))
                break
            except (HTTPError, TimeoutError) as error:
                busy = isinstance(error, TimeoutError) or error.code in (429, 500, 503)
                if not busy or number == len(attempts) - 1 or deadline - time.monotonic() < 8:
                    raise
                logging.getLogger(__name__).warning("Gemini model %s busy (%s); trying the next model", model, error)
        parts = payload["candidates"][0]["content"]["parts"]
        return self.clean("".join(part.get("text", "") for part in parts if not part.get("thought")), len(papers))

    @staticmethod
    def clean(text, count):
        # Some models write non-breaking hyphens and narrow spaces, which break search and copying.
        text = text.translate({0x2010: "-", 0x2011: "-", 0x2012: "-", 0x00A0: " ", 0x202F: " ", 0x2009: " "})
        # Full-width citation marks such as 【1】 or 【1†source】 become [1].
        text = re.sub(r"[【［]\s*(\d+)[^】］\]]{0,40}[】］]", r"[\1]", text)
        text = re.sub(r"[*#`_]{1,3}", "", text)
        text = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= count else "", text)
        text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
        paragraphs = [re.sub(r"\s+", " ", block).strip() for block in re.split(r"\n\s*\n", text)]
        paragraphs = [block for block in paragraphs if block]
        # An answer cut off by the length limit ends mid-sentence; keep it only up to its last full sentence.
        while paragraphs and not re.search(r"[.!?)\]]$", paragraphs[-1]):
            last = paragraphs.pop()
            complete = re.match(r"(.*[.!?](?:\s*\[\d+\])*)\s", last + " ", re.S)
            if complete and len(complete.group(1)) > 40:
                paragraphs.append(complete.group(1).strip())
                break
        return paragraphs


class GroqWriter(GeminiWriter):
    """The same instructions and checks as GeminiWriter, sent to open-weight models served by Groq."""

    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    FALLBACKS = ("qwen/qwen3.8-27b",)

    def __init__(self, key, model=None, timeout=30):
        super().__init__(key, model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b", timeout)

    def write(self, question, history, papers, intent, deadline=None):
        prompt = self.prompt(question, history, papers, intent)
        deadline = self.deadline(deadline)
        attempts = [self.model, *(name for name in self.FALLBACKS if name != self.model)]
        for number, model in enumerate(attempts):
            options = {"reasoning_effort": "low", "include_reasoning": False} if model.startswith("openai/") else {}
            body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                               "temperature": 0.3, "max_completion_tokens": 3000, **options}).encode()
            request = Request(self.ENDPOINT, data=body, method="POST", headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.key}", "User-Agent": "Lumen/1.0"})
            try:
                with urlopen(request, timeout=self.remaining(deadline, 20)) as response:
                    payload = json.loads(response.read(1_000_000))
                break
            except (HTTPError, TimeoutError) as error:
                busy = isinstance(error, TimeoutError) or error.code in (429, 500, 503)
                if not busy or number == len(attempts) - 1 or deadline - time.monotonic() < 6:
                    raise
                logging.getLogger(__name__).warning("Groq model %s busy (%s); trying the next model", model, error)
        text = payload["choices"][0]["message"].get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return self.clean(text, len(papers))


class WriterChain:
    """Tries each configured answer writer in order, so one provider's outage does not stop answers."""

    def __init__(self, writers):
        self.writers = writers
        self.model = " / ".join(writer.model for writer in writers)

    def write(self, question, history, papers, intent, deadline=None):
        for number, writer in enumerate(self.writers):
            try:
                return writer.write(question, history, papers, intent, deadline)
            except (OSError, ValueError, KeyError, IndexError) as error:
                if number == len(self.writers) - 1:
                    raise
                logging.getLogger(__name__).warning("%s failed (%s); trying the next provider",
                                                    type(writer).__name__, error)


def configured_writer():
    writers = [GroqWriter(os.environ["GROQ_API_KEY"])] if os.getenv("GROQ_API_KEY") else []
    if os.getenv("GEMINI_API_KEY"):
        writers.append(GeminiWriter(os.environ["GEMINI_API_KEY"]))
    return WriterChain(writers) if writers else None


class UnsupportedAnswer(ValueError):
    """The written answer, once checked, cites none of the supplied papers."""


class ResearchAssistant:
    def __init__(self, encoder, external, writer=None):
        processor = QueryProcessor()
        self.processor = processor
        self.writer = writer
        self.external = external
        self.context = ConversationContext(processor)
        self.retriever = SemanticRetriever(encoder, LocalCorpus(), external)
        self.composer = EvidenceComposer(processor, encoder)
        self.expansions = {}

    def interpret(self, resolved):
        """Expand abbreviations so that, for example, RAG is searched as retrieval-augmented generation."""
        notes = []
        for acronym in self.processor.acronyms(resolved):
            texts = [paper.title + ". " + paper.abstract for paper in self.retriever.local.search(acronym)]
            expansion = self.processor.expansion(acronym, texts)
            if not expansion and acronym in self.expansions:
                expansion = self.expansions[acronym]
            elif not expansion and self.external:
                try:
                    texts = [paper.title + ". " + paper.abstract for paper in self.external.search(acronym)]
                    expansion = self.processor.expansion(acronym, texts)
                except (OSError, ValueError, TypeError, AttributeError, KeyError):
                    expansion = None
                if expansion is not None:
                    # Remember the result, including "no expansion", so the abbreviation is looked up only once.
                    if len(self.expansions) >= 512:
                        self.expansions.pop(next(iter(self.expansions)), None)
                    self.expansions[acronym] = expansion
            if expansion and expansion not in resolved.lower():
                resolved = re.sub(r"\b" + re.escape(acronym) + r"\b", f"{acronym} ({expansion})", resolved, count=1)
                notes.append(f"{acronym} was read as \u201c{expansion}\u201d.")
        return resolved, " ".join(notes)

    # Total time one answer may take, kept well inside the web server's 120-second worker timeout.
    BUDGET_SECONDS = 80
    # Cosine similarity a cited sentence needs with some sentence of the paper it cites.
    SUPPORT = 0.30
    # How far below the best-matching paper a cited paper may score and still keep its citation.
    MARGIN = 0.10

    def answer(self, question, history):
        deadline = time.monotonic() + self.BUDGET_SECONDS
        contextual = self.context.resolve(question, history)
        resolved, interpretation = self.interpret(contextual)
        if contextual != question:
            # Show the reader which earlier topic the follow-up was linked to.
            topic = ConversationContext.topic(contextual)
            linked = f"Read as a follow-up to “{topic[:120]}”."
            interpretation = " ".join(filter(None, [linked, interpretation]))
        intent = self.processor.intent(question)
        ranked, notice = self.retriever.retrieve(resolved, self.context.search_text(resolved))
        answer = None
        # With no matching paper there is nothing to cite, so no answer is written from general knowledge.
        if self.writer and ranked:
            try:
                answer = self.generate(question, resolved, history, ranked, notice, intent, deadline)
            except UnsupportedAnswer:
                notice = " ".join(filter(None, [notice, "The written answer could not be supported by the papers, "
                                                        "so the answer is made of quoted evidence."]))
            except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
                logging.getLogger(__name__).warning("Answer writer failed: %s", error)
                notice = " ".join(filter(None, [notice, "The answer writer is temporarily unavailable, "
                                                        "so the answer is made of quoted evidence."]))
        if answer is None:
            answer = self.composer.compose(resolved, ranked, notice, intent)
        answer["interpretation"] = interpretation
        return resolved, answer

    def supported(self, paragraphs, papers):
        """Remove citation marks whose sentence is not close in meaning to any sentence of the cited paper.

        The writer may cite the wrong paper for a claim; a range check alone would let that through.
        Returns the checked paragraphs and the number of citation marks removed.
        """
        pieces = [self.split(block) for block in paragraphs]
        claims = [(b, s, int(n)) for b, block in enumerate(pieces) for s, sentence in enumerate(block)
                  for n in dict.fromkeys(re.findall(r"\[(\d+)\]", sentence))]
        if not claims:
            return paragraphs, 0
        evidence = {number: [text for text in self.processor.sentences(paper.abstract) if len(text) >= 20][:30]
                    or [paper.title] for number, paper in enumerate(papers, start=1)}
        sentences = [re.sub(r"\[\d+\]", "", pieces[b][s]) for b, s, _ in claims]
        flat = [text for number in sorted(evidence) for text in evidence[number]]
        vectors = self.retriever.encoder.encode([*sentences, *flat])
        claim_vectors, paper_vectors = vectors[:len(sentences)], vectors[len(sentences):]
        spans, start = {}, 0
        for number in sorted(evidence):
            spans[number] = (start, start + len(evidence[number]))
            start += len(evidence[number])
        removed, dropped = 0, set()
        for (b, s, number), vector in zip(claims, claim_vectors):
            scores = {other: float((paper_vectors[low:high] @ vector).max()) for other, (low, high) in spans.items()}
            best = scores.get(number, 0.0)
            # A citation stands only if its sentence is close to the cited paper, and that paper is the best
            # or nearly the best match among the supplied papers; otherwise another paper says it better.
            if best < self.SUPPORT or best < max(scores.values()) - self.MARGIN:
                pieces[b][s] = re.sub(r"\s*\[" + str(number) + r"\]", "", pieces[b][s])
                removed += 1
                if not re.search(r"\[\d+\]", pieces[b][s]):
                    dropped.add((b, s))  # nothing supports the sentence any more, so it is not shown
        blocks = [" ".join(text for s, text in enumerate(block) if (b, s) not in dropped)
                  for b, block in enumerate(pieces)]
        return [block for block in blocks if block.strip()], removed

    def split(self, paragraph):
        """Sentences of a written paragraph, with a citation that follows a full stop kept on its sentence."""
        sentences = []
        for sentence in self.processor.sentences(paragraph):
            marks = re.match(r"((?:\s*\[\d+\])+)\s*(.*)", sentence, re.S)
            if marks and sentences:
                sentences[-1] += marks.group(1)
                sentence = marks.group(2)
            if sentence.strip():
                sentences.append(sentence.strip())
        return sentences

    def generate(self, question, resolved, history, ranked, notice, intent, deadline=None):
        papers = [paper for paper, _ in ranked if paper.abstract][:5]
        if not papers:
            raise ValueError("no papers to write from")
        question_text = question if resolved == question else f"{question} (context: {resolved.split(chr(10))[0]})"
        paragraphs = self.writer.write(question_text, history, papers, intent, deadline)
        if not paragraphs:
            raise ValueError("empty answer")
        paragraphs, unsupported = self.supported(paragraphs, papers)
        # Keep only the papers the answer cites, numbered in order of first citation.
        order = []
        for number in re.findall(r"\[(\d+)\]", " ".join(paragraphs)):
            if 1 <= int(number) <= len(papers) and int(number) not in order:
                order.append(int(number))
        renumber = {old: new for new, old in enumerate(order, start=1)}
        paragraphs = [re.sub(r"\[(\d+)\]", lambda m: f"[{renumber[int(m.group(1))]}]" if int(m.group(1)) in renumber
                             else "", block) for block in paragraphs]
        cited = [papers[old - 1] for old in order]
        if not cited:
            raise UnsupportedAnswer("the written answer cites none of the papers")
        # The quote shown for a source is the sentence of that paper closest to the claim that cites it.
        claim_for = {}
        for sentence in self.split(" ".join(paragraphs)):
            for number in re.findall(r"\[(\d+)\]", sentence):
                claim_for.setdefault(int(number), re.sub(r"\[\d+\]", "", sentence))
        findings = []
        for number, paper in enumerate(cited, start=1):
            chosen = self.composer.best_sentence(claim_for.get(number, resolved), paper, intent)
            if chosen and len(chosen[1].split()) >= 10:  # skip linking sentences such as "This is where X steps in."
                findings.append({"text": chosen[1], "citation": number})
        cited_urls = {paper.url for paper in cited}
        overview = (f"Written from {len(cited)} paper{'s' if len(cited) > 1 else ''}. Numbers in brackets link "
                    "each cited statement to its source; the key evidence below quotes the papers directly.")
        if unsupported:
            plural = unsupported > 1
            overview += (f" {unsupported} citation{'s' if plural else ''} that did not match the cited paper "
                         f"closely enough {'were' if plural else 'was'} removed.")
        return {"intent": intent, "heading": EvidenceComposer.HEADINGS[intent], "text": "\n\n".join(paragraphs),
                "overview": overview, "summary": [], "findings": findings,
                "sources": [paper.reference() for paper in cited],
                "leads": [paper.reference() for paper, _ in ranked if paper.url not in cited_urls],
                "notice": notice, "generated": True, "unsupported_citations_removed": unsupported}


if __name__ == "__main__":
    HuggingFaceEncoder().load()
