# v1.6.1: All-caps heading boundary fix

This patch resolves the remaining pilot SOFT occurrence:

```text
POLICY Tanger
```

HTML-to-text extraction can flatten a heading and the following sentence into one whitespace-separated string. `POLICY` is therefore not treated as a capitalised modifier of the entity alias `Tanger`.

The occurrence is now classified as:

```json
{
  "control_eligibility": "HARD",
  "usage_status": "DEFINED_TERM_USAGE",
  "context_classification": "HEADING_BOUNDARY",
  "validation_reasons": ["ALL_CAPS_HEADING_BOUNDARY"]
}
```

The rule is deliberately narrow: the previous token must be an all-caps known heading label and the source term must look like an entity alias. It does not affect unresolved semantic compounds such as `Sanctioned Person`.
