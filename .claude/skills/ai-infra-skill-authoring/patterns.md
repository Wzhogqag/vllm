# Skill Patterns

## Good Patterns

### Trigger-First Design

Spend disproportionate effort on the description. It is the routing key.

### Progressive Disclosure

Put the always-needed workflow in SKILL.md and branch-specific detail into
separate files.

### Evidence-Shaped Outputs

Force the model to emit facts, hypotheses, unknowns, and next checks separately
when the domain is uncertainty-heavy.

### Deterministic Assist

Add scripts only when code materially improves correctness or cost, such as log
parsing, benchmark summarization, or schema validation.

## Anti-Patterns

### Giant Expert Skill

One skill that tries to cover incidents, benchmarking, design, review, and
roadmapping will trigger poorly and become expensive.

### Policy Hidden In Sidecars

If the main workflow depends on a rule, keep it in SKILL.md. Sidecars should be
optional depth.

### No Evaluation Harness

If you cannot name prompts that should and should not trigger the skill, the
boundary is not ready.

### Script Without Guardrails

Do not ship scripts that fetch external data, install risky dependencies, or
mutate files broadly unless the workflow truly requires it and the risk is
documented.
