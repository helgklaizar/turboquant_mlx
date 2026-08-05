import Foundation

struct AnalysisRequest {
    var transcript: String
    /// What the model already summarised in this session, so realtime passes stay coherent.
    var previousSummary: String?
    /// Insight texts already shown to the user — the model is told not to repeat them.
    var knownInsights: [String]
    var language: String
    var mode: CaptureMode
}

struct AnalysisResult {
    var summary: String
    var insights: [Insight]

    static let empty = AnalysisResult(summary: "", insights: [])
}

protocol Analyzer {
    func analyze(_ request: AnalysisRequest) async throws -> AnalysisResult
}

// MARK: - Prompt

enum AnalysisPrompt {

    static let system = """
    You are a note-taking assistant. You receive a raw, imperfect speech-to-text \
    transcript and pull out only what matters. Ignore filler, small talk and \
    transcription noise.

    Answer with a single JSON object and nothing else. No prose, no markdown fences.

    Schema:
    {
      "summary": "two or three sentences",
      "items": [
        {
          "kind": "task|decision|fact|question|date|contact|idea",
          "text": "one short sentence",
          "who": "person responsible, or null",
          "due": "deadline as stated, or null",
          "confidence": 0.0
        }
      ]
    }

    Rules:
    - Write summary and text in the same language as the transcript.
    - Never invent anything that is not in the transcript. When the transcript is \
      too short or says nothing of substance, return an empty items array.
    - confidence is your own 0.0-1.0 estimate that the item is real and correctly read.
    - At most 12 items.
    """

    static func user(for request: AnalysisRequest) -> String {
        var parts: [String] = []

        if let previous = request.previousSummary, !previous.isEmpty {
            parts.append("Summary of the conversation so far:\n\(previous)")
        }
        if !request.knownInsights.isEmpty {
            let known = request.knownInsights.prefix(30).map { "- \($0)" }.joined(separator: "\n")
            parts.append("Already captured — do not repeat these:\n\(known)")
        }
        if request.mode == .realtime {
            parts.append("This is a live fragment of an ongoing recording. Extract only what is new in it.")
        }
        parts.append("Transcript:\n\(request.transcript)")

        return parts.joined(separator: "\n\n")
    }
}

// MARK: - Response decoding

/// What the server is asked to return. Kept separate from `Insight` so a model that
/// omits optional fields still decodes.
struct AnalysisPayload: Decodable {
    /// Every field is optional: a model that omits one should cost us that item,
    /// not the whole analysis pass.
    struct Item: Decodable {
        var kind: String?
        var text: String?
        var who: String?
        var due: String?
        var confidence: Double?
    }

    var summary: String?
    var items: [Item]?

    func toResult() -> AnalysisResult {
        let insights = (items ?? []).compactMap { item -> Insight? in
            let text = (item.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            return Insight(
                kind: InsightKind(lenient: item.kind ?? "fact"),
                text: text,
                who: item.who?.isEmpty == false ? item.who : nil,
                due: item.due?.isEmpty == false ? item.due : nil,
                confidence: min(max(item.confidence ?? 0.5, 0), 1)
            )
        }
        return AnalysisResult(
            summary: summary?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "",
            insights: insights
        )
    }

    /// Models wrap JSON in prose or fences more often than they should.
    /// Pull the outermost object out before decoding.
    static func decode(fromModelOutput raw: String) throws -> AnalysisPayload {
        let cleaned = raw
            .replacingOccurrences(of: "```json", with: "```")
            .components(separatedBy: "```")
            .first(where: { $0.contains("{") }) ?? raw

        guard
            let start = cleaned.firstIndex(of: "{"),
            let end = cleaned.lastIndex(of: "}"),
            start < end
        else {
            throw AnalyzerError.unparseable(String(raw.prefix(200)))
        }

        let json = String(cleaned[start...end])
        guard let data = json.data(using: .utf8) else {
            throw AnalyzerError.unparseable(String(json.prefix(200)))
        }
        return try JSONDecoder().decode(AnalysisPayload.self, from: data)
    }
}

enum AnalyzerError: LocalizedError {
    case badURL(String)
    case http(Int, String)
    case unparseable(String)
    case notConfigured

    var errorDescription: String? {
        switch self {
        case .badURL(let value):
            return "\"\(value)\" is not a valid server URL."
        case .http(let code, let body):
            return "Server returned HTTP \(code). \(body.prefix(160))"
        case .unparseable(let snippet):
            return "Could not read the model's answer as JSON: \(snippet)"
        case .notConfigured:
            return "No analysis backend is configured. Set the server URL in Settings."
        }
    }
}
