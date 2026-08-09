from legal_tc_bench.terms import extract_defined_terms


def test_defined_term_extraction_and_occurrences():
    paragraphs = [
        '"Purchaser" means Alpha Holdings Ltd.',
        'The Purchaser shall pay the Purchase Price.',
        'The Purchaser may assign its rights.',
        'Notices to the Purchaser must be in writing.',
    ]
    terms = extract_defined_terms(paragraphs, min_occurrences=3)
    assert len(terms) == 1
    assert terms[0].source_term == "Purchaser"
    assert len(terms[0].occurrences) == 4
