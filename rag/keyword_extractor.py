import spacy

nlp = spacy.load("en_core_sci_sm")

class KeywordExtractor:

    def extract_keywords(self, text: str):
        doc = nlp(text)
        keywords = []

        for ent in doc.ents:
            keywords.append(ent.text.lower())

        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                keywords.append(chunk.text.lower())

        final = []
        seen = set()
        for k in keywords:
            k = k.strip()
            if k not in seen and len(k) > 2:
                seen.add(k)
                final.append(k)

        return final[:15]
