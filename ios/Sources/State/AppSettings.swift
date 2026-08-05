import Foundation

/// Persisted preferences.
///
/// Deliberately not `@AppStorage`: that property wrapper only publishes changes
/// when it lives inside a `View`. In an `ObservableObject` it would write through
/// to `UserDefaults` without ever firing `objectWillChange`, so the mode picker
/// would look stuck. Plain `@Published` + `didSet` is boring and correct.
@MainActor
final class AppSettings: ObservableObject {

    @Published var serverURL: String { didSet { defaults.set(serverURL, forKey: Key.serverURL) } }
    @Published var modelName: String { didSet { defaults.set(modelName, forKey: Key.modelName) } }
    @Published var mode: CaptureMode { didSet { defaults.set(mode.rawValue, forKey: Key.mode) } }
    @Published var localeIdentifier: String { didSet { defaults.set(localeIdentifier, forKey: Key.locale) } }
    /// Keep the raw `.caf` next to the transcript. Off by default — audio is the
    /// bulkiest and most sensitive thing this app touches.
    @Published var keepAudio: Bool { didSet { defaults.set(keepAudio, forKey: Key.keepAudio) } }

    private let defaults: UserDefaults

    private enum Key {
        static let serverURL = "serverURL"
        static let modelName = "modelName"
        static let mode = "captureMode"
        static let locale = "localeIdentifier"
        static let keepAudio = "keepAudio"
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        serverURL = defaults.string(forKey: Key.serverURL) ?? "http://mac.local:8080"
        modelName = defaults.string(forKey: Key.modelName) ?? "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
        mode = CaptureMode(rawValue: defaults.string(forKey: Key.mode) ?? "") ?? .button
        localeIdentifier = defaults.string(forKey: Key.locale) ?? "ru-RU"
        keepAudio = defaults.bool(forKey: Key.keepAudio)
    }

    static let supportedLocales: [(id: String, name: String)] = [
        ("ru-RU", "Русский"),
        ("en-US", "English (US)"),
        ("en-GB", "English (UK)"),
        ("de-DE", "Deutsch"),
        ("es-ES", "Español"),
        ("fr-FR", "Français")
    ]

    func makeAnalyzer() -> Analyzer? {
        try? RemoteAnalyzer(baseURLString: serverURL, model: modelName)
    }
}
