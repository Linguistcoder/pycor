def remove_end(string):
    string = string.split()
    if '...' in string:
        string = string[:-2]
    return ' '.join(string)


def expand_synsets(synsets: dict, s=True):
    for synset_id, synset in synsets.items():
        hypernyms = [synsets[syn if s else syn.synset_id] for syn in synset.hypernyms if syn]
        synsets[synset_id].hypernyms = hypernyms

        hyponyms = [synsets[syn if s else syn.synset_id] for syn in synset.hyponyms if syn]
        synsets[synset_id].hyponyms = hyponyms

        holonyms = [synsets[syn if s else syn.synset_id] for syn in synset.holonyms if syn]
        synsets[synset_id].holonyms = holonyms

        meronyms = [synsets[syn if s else syn.synset_id] for syn in synset.meronyms if syn]
        synsets[synset_id].meronyms = meronyms

        other_relations = [synsets[syn if s else syn.synset_id] for syn in synset.other_relations if syn]
        synsets[synset_id].other_relations = other_relations

        synonyms = [synsets[syn if s else syn.synset_id] for syn in synset.synonyms if syn]
        synsets[synset_id].synonyms = synonyms

    return synsets


def retrieve_words(syn, criteria: list):
    """Types of criteria:
        - 'local' = local synset members
        - 'hyp_members' = members in hypernyms and hyponyms
        - 'definition' = definition of synset
        - 'hyp_definition' = definition of hypernyms and hyponyms
        - 'example_sentence' = example sentence(s) from gloss """
    syn_repr = []
    if 'local' in criteria:
        syn_repr += syn.get_members()

    if 'definition' in criteria:
        syn_repr += syn.get_definition()

    if 'hyp_members' in criteria or 'hyp_definition' in criteria:
        for hyp in syn.hypernyms + syn.hyponyms:
            if 'hyp_members' in criteria:
                syn_repr += hyp.get_members()

            if 'hyp_definition' in criteria:
                syn_repr += hyp.get_definition()

    if 'example_sentence' in criteria:
        syn_repr += syn.get_example_sentence()

    if 'all_example_sentences' in criteria:
        examples = syn.get_all_example_sentences(hypernyms=True)
        examples_tokens = [s.split() for s in examples]
        syn_repr += [token for tokens in examples_tokens for token in tokens if token]

    syn_repr = [s for s in syn_repr if s]
    return syn_repr


def retrieve_sentences(syn, criteria: list):
    syn_repr = []
    if 'local' in criteria:
        syn_repr = syn.get_all_example_sentences()

    elif 'hyper' in criteria:
        syn_repr = syn.get_all_example_sentences(hypernyms=True)

    elif 'hypo' in criteria:
        syn_repr = syn.get_all_example_sentences(hyponyms=True)

    elif 'all_hyp' in criteria:
        syn_repr = syn.get_all_example_sentences(hypernyms=True, hyponyms=True)

    if 'def' in criteria:
        syn_repr += syn.get_definition(preprocess=False)

    return syn_repr
