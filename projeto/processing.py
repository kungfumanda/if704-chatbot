import json
import spacy
from spacy.tokens import Span

# Carregar o modelo do spacy
nlp = spacy.load("en_core_web_sm")
Span.set_extension("type", default="SOMETHING_ELSE", force=True)
all_tokens = []

# Carregar os dados do JSON
with open('../external data/ccpe_example_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Classificar intents
def classify_intent_user(annotation_type, entity_type):
    # Majoritariamente valido para speaker user    
    if entity_type == "MOVIE_OR_SERIES":
        if annotation_type == "ENTITY_NAME":
            return "NAME_MOVIE_SERIES"
        elif annotation_type == "ENTITY_PREFERENCE":
            return "PREFERENCE_MOVIE_SERIES"
        elif annotation_type == "ENTITY_DESCRIPTION":
            return "DESCRIBE_MOVIE_SERIES"
        elif annotation_type == "ENTITY_OTHER":
            return "OTHER_INFO_MOVIE_SERIES"
        
    elif entity_type == "MOVIE_GENRE_OR_CATEGORY":
        if annotation_type == "ENTITY_NAME" :
            return "NAME_MOVIE_GENRE"
        elif annotation_type == "ENTITY_PREFERENCE" :
            return "PREFERENCE_MOVIE_GENRE"
        elif annotation_type == "ENTITY_DESCRIPTION" :
            return "DESCRIBE_MOVIE_GENRE"
        elif annotation_type == "ENTITY_OTHER" :
            return "OTHER_INFO_MOVIE_GENRE"

    elif entity_type == "PERSON":
        if annotation_type == "ENTITY_NAME":
            return "NAME_PERSON"
        elif annotation_type == "ENTITY_PREFERENCE":
            return "PREFERENCE_PERSON"
        elif annotation_type == "ENTITY_DESCRIPTION":
            return "DESCRIBE_PERSON"
        elif annotation_type == "ENTITY_OTHER":
            return "OTHER_INFO_PERSON"
    
    elif entity_type == "SOMETHING_ELSE":
        if annotation_type == "ENTITY_NAME" :
            return "NAME_SOMETHING_ELSE"
        elif annotation_type == "ENTITY_PREFERENCE":
            return "PREFERENCE_SOMETHING_ELSE"
        elif annotation_type == "ENTITY_DESCRIPTION":
            return "DESCRIBE_SOMETHING_ELSE"
        elif annotation_type == "ENTITY_OTHER":
            return "OTHER_INFO_SOMETHING_ELSE"
        
    else:
        return "UNKOWN_INTENT"

# Função para tokenização e lematização
def tokenize_and_lemmatize(doc):
    tokens = [token for token in doc]
    lemmas = [token.lemma_ for token in doc]
    tokens_lemmas = [f"{token.text} ===> {token.lemma_}" for token in doc]
    return tokens, lemmas, tokens_lemmas

# Retorna entidades com texto e tipo
def extract_entities(text):
    doc = nlp(text)
    entities = ((ent.text, ent._.type) for ent in doc.ents)
    return entities

# Iterar sobre as utterances no JSON
for conversation in data:
    for utterance in conversation["utterances"]:
        text = utterance["text"]
        doc = nlp(text)

        # speaker = utterance["speaker"]
        # speaker == user || speaker == assistant

        # se houver segments, aplicar a tokenização e lematização a cada segmento
        if "segments" in utterance:
            for segment in utterance["segments"]:
                segment_text = segment["text"]

                # detalhes necessarios pra extracao de intents e entidades
                segment_annotation_type = segment["annotations"][0]["annotationType"]
                # annotation type == ENTITY_NAME || ENTITY_PREFERENCE || ENTITY_DESCRIPTION || ENTITY_OTHER
                segment_entity_type = segment["annotations"][0]["annotationType"]
                # entity type == MOVIE_GENRE_OR_CATEGORY || MOVIE_OR_SERIES || PERSON || SOMETHING_ELSE
                

                # melhorar depois - retirar o for e incluir outras annotations
                # add tipo de entidade                
                for ent in doc.ents:
                    if ent.text.lower() == segment_text.lower():
                        ent._.type = segment_entity_type

                tokens, lemmas, tokens_lemmas = tokenize_and_lemmatize(nlp(segment_text))

                print(f"Text: {text}")
                print(f"Tokenized: {tokens}")
                print(f"Lemmatized: {lemmas}")
                all_tokens.extend(tokens_lemmas)
                
                
        else:
            # Se não houver segments, aplicar a tokenização e lematização à utterance como um todo
            tokens, lemmas, tokens_lemmas = tokenize_and_lemmatize(doc)
            print(f"Text: {text}")
            print(f"Tokenized: {tokens}")
            print(f"Lemmatized: {lemmas}")
            all_tokens.extend(tokens_lemmas)


        # Imprimir entidades
        entities = extract_entities(doc)
        for ent in entities:
            print("\n")
            print("Text:", ent[0])
            print("Type:", ent[1])

        print("\n" + "="*50 + "\n")  # Separador para facilitar a leitura


    # Remove repetidos
    all_tokens = list(set(all_tokens))
    # Printa combinações de tokens e lemmas
    print("\n".join(all_tokens))
