import Foundation

/// One recording, its transcript and everything the model extracted from it.
struct Session: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var startedAt: Date = Date()
    var endedAt: Date?
    var mode: CaptureMode
    var transcript: String = ""
    var summary: String = ""
    var insights: [Insight] = []
    /// File name inside the app's `Recordings` directory, if audio was kept.
    var audioFileName: String?

    var duration: TimeInterval {
        (endedAt ?? Date()).timeIntervalSince(startedAt)
    }

    var title: String {
        if !summary.isEmpty {
            return String(summary.prefix(60))
        }
        if !transcript.isEmpty {
            return String(transcript.prefix(60))
        }
        return "Session"
    }
}
