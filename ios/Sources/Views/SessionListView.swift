import SwiftUI

struct SessionListView: View {
    @EnvironmentObject private var store: SessionStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(store.sessions) { session in
                    NavigationLink {
                        SessionDetailView(session: session)
                    } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(session.title)
                                .lineLimit(2)
                            HStack(spacing: 8) {
                                Text(session.startedAt, style: .date)
                                Text(session.mode.title)
                                Text("\(session.insights.count) items")
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                    }
                }
                .onDelete { store.delete(at: $0) }
            }
            .overlay {
                if store.sessions.isEmpty {
                    ContentUnavailableView(
                        "No sessions yet",
                        systemImage: "waveform",
                        description: Text("Recordings you finish show up here.")
                    )
                }
            }
            .navigationTitle("Sessions")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

struct SessionDetailView: View {
    let session: Session

    var body: some View {
        List {
            if !session.summary.isEmpty {
                Section("Summary") {
                    Text(session.summary).textSelection(.enabled)
                }
            }
            if !session.insights.isEmpty {
                Section("Extracted") {
                    ForEach(session.insights) { InsightRow(insight: $0) }
                }
            }
            if !session.transcript.isEmpty {
                Section("Transcript") {
                    Text(session.transcript)
                        .font(.footnote)
                        .textSelection(.enabled)
                }
            }
        }
        .navigationTitle(session.startedAt.formatted(date: .abbreviated, time: .shortened))
        .navigationBarTitleDisplayMode(.inline)
    }
}
