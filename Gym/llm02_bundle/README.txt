Copy prepare.py to benchmarks/owasp_llm02/prepare.py and run:
  python benchmarks/owasp_llm02/prepare.py
  gym eval prepare --benchmark owasp_llm02
  gym env validate --benchmark owasp_llm02
Read benchmarks/owasp_llm02/data/preparation_report.json for included/excluded counts.
IMPORTANT: the current verifier expects password and must be updated to protected_values and scenario_type before model evaluation. The converter intentionally excludes unscorable datasets instead of fabricating labels.
