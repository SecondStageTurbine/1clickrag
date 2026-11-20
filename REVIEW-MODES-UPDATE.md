# Review modes update (v12)

General document Q&A remains the default. Maintenance tracing remains a separate option. This update adds Diagnostic artifact review, Symptom–failure matrix, Diagnostic peer review, and Source-marking peer review.

## Use

1. Restart the application after updating, then refresh the browser.
2. Choose General, Diagnostic artifact review, Symptom–failure matrix, or Supplier fault-code maintenance trace under **Task**. Select the appropriate source folder/revision with the existing filters and ask your question.
3. After a complete answer, select a peer-review mode and ask what to review. The latest completed non-peer answer and the exact retrieved evidence from its final generation round are attached automatically. Peer reviews do not replace that original snapshot.
4. Original references use `[P1]`, etc., and can be read in the **Original review input** panel. Supplemental search references use `[1]`, etc., and retain the usual source preview. Peer review begins with the original evidence; if searching is enabled, the model can request bounded supplemental searches.

Reloading or **New** clears chat history and the review snapshot. They are not stored in localStorage; only mode/backend preferences are. An interrupted response does not replace the last complete snapshot. Answers exceeding 60,000 characters or evidence exceeding 100,000 characters cannot become a browser review snapshot. The UI explains when no review input is available.

## API

`GET /chat/modes` returns the six supported modes and `peer_review` metadata. Existing `/chat` clients still default to `general`.

Peer-review requests require an explicit `review` object:

```json
{
  "mode": "diagnostic_peer_review",
  "messages": [{"role": "user", "content": "Check grounding and unresolved branches."}],
  "review": {
    "analysis": "The prior answer with its original [1] references.",
    "evidence": "[1] supplier.txt:1-5\nThe exact original retrieved passage."
  },
  "retrieve": false
}
```

If evidence is absent, the policy requires a limited review. Missing analysis returns HTTP 400; oversized inputs return 422. The snapshot is client-supplied content, not an authenticated audit record. Existing path/language filters apply to supplemental searches, not retroactively to original evidence. Both original evidence and new passages go to the selected backend; use an approved local backend for sensitive work.

## Policy changes

The partner's artifact and matrix policies were adapted under `app/policies`; originals in FileShare were not changed. The unsupported program-specific example mapping was removed. Candidate inference is separate from confirmed mappings; every relationship needs source support. Matrix output is bounded with a sparse format for large results, and blank cells do not mean a link is disproven. Peer-review policies now require claim-level findings, evidence, corrections, and uncertainty.

Source markings are reported with attribution, not assigned to generated content. Derived or unmarked text is not automatically unclassified. This feature does not establish classification, release authority, or TS accreditation. Prompt rules are safeguards, not guarantees of model compliance. Human engineering/security review remains necessary.

## Installation and rollback

The v12 ZIP is a cumulative **source overlay for an existing compatible install**, not a new offline runtime/model bundle. It contains the complete first-party `app` tree (including preview/cache dependencies), policies, tests, and runtime source/configuration templates. No `.env`, `.data`, corpus documents, databases, credentials, model weights, or virtual environment are included.

Stop RAG and back up the existing install before extracting the ZIP into its root (the folder containing `Rag.bat`), allowing packaged source files to be replaced. If your partner has separate custom backend/frontend changes, merge those files instead of blindly overwriting them. Restart and refresh the browser. To roll back, stop RAG and restore the backed-up source tree; retain your `.env` and data.

The new modes and evidence handoff alone require no re-index. Upgrading from a build without the earlier extraction/relationship changes may require re-indexing to populate those features. Dependency and local-model availability must still be checked on the target; this source-only ZIP cannot install missing wheels or models on a dark network. It does not update the separate RHEL GPU bundle.

## Verification and remaining acceptance test

Run `.venv\Scripts\python.exe -m unittest discover -s tests -q` on Windows, or the equivalent environment's Python on Linux. Node is needed only for the optional UI-handler integration test, not to run RAG.

The suite includes real XLSX extraction, Qdrant/SQLite hybrid retrieval, policy selection, the Ollama HTTP adapter, SSE delivery, original-evidence handoff, supplemental retrieval, context-budget preservation, input validation, and UI handlers calling a real local HTTP server. The UI test uses a minimal DOM, not browser visual rendering. Its answer endpoint is scripted and its embedding vectors are deterministic test doubles. These tests verify application plumbing, not real model comprehension or aircraft-manual accuracy.

On the development machine, Ollama exposes only `nomic-embed-text:latest`; no answer-generating Ollama model is installed. No models were downloaded and no supplier material was sent to an external service. Final acceptance still requires your approved local LLM and representative supplier documents: trace the chosen fault code to independently verified preparations, figure/block/sheet transitions, all supported branches, and maintenance actions; verify each citation and ensure unresolved branches stay unresolved. Repeat with artifact, matrix, and both peer-review modes. Bounded retrieval is not an exhaustive typed-graph traversal, and policies do not create missing evidence or complete unresolved OCR labels.
