# Watermark Lab usability revision

Style: Framed sheet, seed 867065199. No redraws. Keep Segoe UI and the existing teal. A pale green-gray frame surrounds a white work surface; teal primarily identifies favored tokens and plotted measurements.

Screen question: Can I generate a watermark and see whether the detector finds it?
Signature: the ordinary and watermarked token sequences, actual measured scores, and an interpretation of the pair.

Inventory: paired completions; a shared prompt and random seed; numeric offline tokens; readable pretrained-model output; a private watermark key; per-position green lists; scored token pairs; z-score and cutoff; three paraphrases; French back-translation; archived CSV measurements.

Screen questions:
- Generate: What changes when the watermark is added?
- Check: Does this sample match my watermark key?
- Rewrite: Does the detector still find the pattern after rewriting?
- Batch: How does the pattern behave across topics and transformations?
- Archive: What did a previous experiment measure?

Changes: remove ignored prompt and model inputs from the offline demo; explain offline output before starting; show model inputs only for readable text; explain every advanced setting; hide settings irrelevant to the current task; guard language-only actions; summarize observed outcomes, including failed detection and false positives; preserve original configuration when handing samples to detection or rewriting; retain task-specific results; show signed archive scores on a common scale; fold navigation into a menu on small screens.

Validation:
- JavaScript syntax passes.
- Existing Python API, boundary, offline integration and watermark tests: 10 pass.
- A DOM simulation against the live local API passes mode switching, ignored-prompt hiding, attack guards, missing input, actual generation, detector handoff with the original key, independent detection, per-task results and archive rendering. This is functional verification, not a browser layout check.
- Source design lint: zero failures and zero warnings.
- Local dashboard returns HTTP 200.

Visual critique: not scored. Both before/after inspector attempts fail at browser launch with spawn EPERM. The computer-use browser inventory is empty. No screenshots or responsive visual measurements were available; visual quality remains unverified. Do not claim browser verification.

No sample data, decorative images, new runtime dependencies or external assets were added. All result narratives depend on the actual detector output.

## Paragraph display correction

Readable paragraph generation is now the default. Each sample shows its exact fully decoded text before metrics. Paragraph breaks and Unicode are preserved, the prose has no internal scroll limit, and token-by-token inspection is collapsed separately. The archive uses the same prose typography. Added exact-passage copy actions and an explicit numeric-test label.

Verification: JavaScript and download script syntax pass; source lint reports zero failures/warnings. DOM flow checks cover exact paragraph/Unicode preservation independently of token fragments, paragraph-before-metrics order, optional highlighting, exact clipboard copying, and actionable blocked-download messaging, alongside the existing offline integration workflow.

A real Qwen generation attempt failed: no cached model and a refused network proxy connection at 127.0.0.1:9. No readable generation success is claimed. download_model.ps1 lets the user cache the model in the expected project directory from a normal network-enabled PowerShell session. Browser visual inspection remains unavailable.
