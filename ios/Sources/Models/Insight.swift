import Foundation

/// The kinds of things the model is asked to pull out of a transcript.
enum InsightKind: String, Codable, CaseIterable, Identifiable {
    case task
    case decision
    case fact
    case question
    case date
    case contact
    case idea

    var id: String { rawValue }

    /// Unknown kinds coming back from a model are folded into `.fact`
    /// instead of failing the whole decode.
    init(lenient raw: String) {
        self = InsightKind(rawValue: raw.lowercased()) ?? .fact
    }

    var symbol: String {
        switch self {
        case .task: return "checkmark.circle"
        case .decision: return "flag"
        case .fact: return "info.circle"
        case .question: return "questionmark.circle"
        case .date: return "calendar"
        case .contact: return "person.crop.circle"
        case .idea: return "lightbulb"
        }
    }

    var title: String { rawValue.capitalized }
}

struct Insight: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var kind: InsightKind
    var text: String
    var who: String?
    var due: String?
    var confidence: Double
    var createdAt: Date = Date()

    /// Key used to drop repeats when the rolling realtime window re-reports
    /// something we already have.
    var dedupeKey: String {
        let normalized = text
            .lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        return "\(kind.rawValue)|\(normalized)"
    }
}
