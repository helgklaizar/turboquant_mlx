import Foundation

/// The two ways the assistant can be driven.
enum CaptureMode: String, Codable, CaseIterable, Identifiable {
    /// One big button. Tap to start, tap to stop, analysis runs once at the end.
    case button
    /// Continuous listening. The transcript is analysed in a rolling window while you speak.
    case realtime

    var id: String { rawValue }

    var title: String {
        switch self {
        case .button: return "Button"
        case .realtime: return "Realtime"
        }
    }

    var subtitle: String {
        switch self {
        case .button: return "Tap to record, analyse on stop"
        case .realtime: return "Analyse continuously while listening"
        }
    }
}
