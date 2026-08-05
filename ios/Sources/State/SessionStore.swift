import Foundation

/// Flat JSON file on disk. Sessions are small; a database would be overkill until
/// transcripts need full-text search.
@MainActor
final class SessionStore: ObservableObject {

    @Published private(set) var sessions: [Session] = []

    private let fileURL: URL
    let recordingsDirectory: URL

    init() {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        fileURL = support.appendingPathComponent("sessions.json")

        recordingsDirectory = support.appendingPathComponent("Recordings", isDirectory: true)
        try? FileManager.default.createDirectory(at: recordingsDirectory, withIntermediateDirectories: true)

        load()
    }

    func save(_ session: Session) {
        if let index = sessions.firstIndex(where: { $0.id == session.id }) {
            sessions[index] = session
        } else {
            sessions.insert(session, at: 0)
        }
        persist()
    }

    func delete(at offsets: IndexSet) {
        for index in offsets {
            if let name = sessions[index].audioFileName {
                try? FileManager.default.removeItem(at: recordingsDirectory.appendingPathComponent(name))
            }
        }
        sessions.remove(atOffsets: offsets)
        persist()
    }

    func audioURL(for name: String) -> URL {
        recordingsDirectory.appendingPathComponent(name)
    }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        sessions = (try? decoder.decode([Session].self, from: data)) ?? []
    }

    private func persist() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted]
        guard let data = try? encoder.encode(sessions) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}
